"""Train an ACT policy on a converted LeRobot dataset. Runs in `.venv-lerobot`.

.. code-block:: bash

    ./.venv-lerobot/bin/python scripts/train_act.py --dataset grasp_cube_dev \
        --steps 2000 --ceiling-minutes 45
    ./.venv-lerobot/bin/python scripts/train_act.py --resume train__grasp_cube_dev__...

Everything about the policy — the config class, the feature spec, the
normalization statistics, the checkpoint layout — comes from the installed
lerobot (0.6.1) rather than from this file's memory of it: ``make_policy``
derives the input/output features from the dataset metadata,
``make_pre_post_processors`` builds the normalize/unnormalize pipeline from the
dataset's own stats, and both pipelines are saved *next to* the weights so
``scripts/policy_server.py`` can reload the exact same preprocessing without
ever seeing the dataset. lerobot's own ``lerobot_train.py`` is not reused: it
requires the ``training`` extra (accelerate, wandb) that this venv deliberately
does not install, and it has no notion of the VRAM ceiling or the wall-clock
ceiling this shared machine runs under.

**The validation split is ours to honour.** ``scripts/convert_dataset.py``
wrote ``datasets/lerobot/<name>/val_split.json`` from the raw manifest
(``attempt_index % 20 == 0``). The train loader is built with
``episodes=train_episode_indices`` so a val episode can never enter a gradient
step, and val loss is computed here — over a *fixed* batch order, in eval mode,
under ``no_grad`` — every ``--val-every`` steps. Val loss is therefore the pure
L1 action error: ACT only runs its VAE encoder while ``policy.training`` is
true, so the KL term that dominates the training loss is absent by
construction. The two curves are each comparable with themselves across steps,
never with each other.

**Chunk size 50, not lerobot's default 100.** Episodes here average 197 frames
at 30 Hz (6.6 s). A 100-step chunk is 3.3 s — more than half a demonstration —
so a single forward pass would have to predict across the DESCEND/CLOSE/LIFT
boundaries from one wrist frame that cannot tell them apart, and every chunk
sampled in the last half of an episode would be mostly padding. 50 steps is
1.67 s: longer than any single expert phase transition, short enough that the
whole chunk is real supervision, and long enough for the eval client's
temporal ensembling (m=0.1 ⇒ weight e^-5 ≈ 0.7% at the far end) to have
saturated. The choice is recorded in ``run.json`` next to the episode-length
statistics it was made from.

**Shared-GPU contract** (plan §Environment facts): pre-flight ≥6500 MiB free,
≤5500 MiB ours. The batch size is not asserted, it is *measured* — a 20-step
probe on a throwaway policy reports peak allocated, peak reserved and the
nvidia-smi figure for this process, and the batch halves until the last of
those three (the number other users of the card actually see) fits. Both the
probe ladder and the peak of the real run land in ``run.json``.

**Checkpoints** go to ``runs/train/<run>/checkpoints/<step>/`` in lerobot's own
layout (``pretrained_model/`` + ``training_state/``), which is what makes
``--resume`` restore the optimizer moments and the step counter rather than
just the weights. Retention is enforced at save time — last, best-by-val, and
every 10 000 steps — because a 60 000-step run at ~350 MB a checkpoint would
otherwise blow the 60 GB disk budget long before it finished.

**Stopping is a feature.** ``--ceiling-minutes`` and SIGTERM both set the same
flag: the current step finishes, a checkpoint is written, ``run.json`` records
which ceiling was hit, and the process exits 0. That is what makes a run on a
borrowed GPU resumable rather than lost.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the src-layout package importable without installing it; only the
# sim-free half (manus.recorder, for the raw manifest) is ever touched here.
sys.path.insert(0, str(REPO_ROOT / "src"))

KIND = "train"
"""Run kind; names the ``runs/<kind>/`` directory and the run-name prefix."""

DEFAULT_CHUNK = 50
"""Action-chunk length in control steps (1.67 s at 30 Hz). See the module docstring."""

DEFAULT_BATCH = 8
"""Batch size the VRAM probe starts from, per the plan."""

VRAM_CEILING_MIB = 5500
"""Our share of the card (plan §Environment facts). Measured, not assumed."""

MIN_FREE_VRAM_MIB = 6500
"""Pre-flight floor on free VRAM before the process is allowed to start."""

PROBE_STEPS = 20
"""Training steps the batch-size probe runs before reading the peak."""

VAL_EVERY = 500
"""Steps between validation passes."""

SAVE_EVERY = 2000
"""Steps between checkpoints."""

RETAIN_EVERY = 10_000
"""Checkpoints at multiples of this step count are kept forever."""

GRAD_CLIP_NORM = 10.0
"""Gradient-norm clip, matching lerobot's own training default."""

LOSS_WINDOW = 100
"""Steps averaged at each end of the run for the loss-decreased report."""

TASK_FPS_KEY = "action"
"""The feature ``delta_timestamps`` slices into a chunk."""

_DECODE_RETRIES = multiprocessing.Value("i", 0)
"""Shared counter for video-decode retries; incremented inside dataloader workers."""


# --- Provenance ------------------------------------------------------------------


def _nvidia_smi(*arguments: str) -> str | None:
    """First line of an ``nvidia-smi`` query, or None if it cannot be reached."""
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments, "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip().splitlines()[0].strip()


def free_vram_mib() -> int | None:
    """Free VRAM in MiB, or None when nvidia-smi is unavailable."""
    value = _nvidia_smi("--query-gpu=memory.free")
    if value is None:
        return None
    return int(value.split()[0])


def process_vram_mib(pid: int) -> int | None:
    """VRAM this process holds according to nvidia-smi, in MiB.

    The honest number on a shared card: it includes the CUDA context and the
    caching allocator's reserve, neither of which
    :func:`torch.cuda.max_memory_allocated` counts but both of which are
    unavailable to anyone else while we hold them.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            return int(fields[1].split()[0])
    return None


def git_provenance(cwd: Path) -> dict[str, Any]:
    """Commit sha and dirty flag of the repository containing `cwd`."""

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=cwd, capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "dirty": None}
    status = run("status", "--porcelain")
    return {"sha": sha, "dirty": None if status is None else bool(status)}


def stamped_run_name(kind: str, dataset: str, sha: str | None) -> str:
    """``<kind>__<dataset>__<YYYYMMDD-HHMM>__<gitsha8>`` (plan §Design decisions)."""
    return "__".join(
        (kind, dataset, time.strftime("%Y%m%d-%H%M", time.localtime()), (sha or "nogit")[:8])
    )


def env_block() -> dict[str, Any]:
    """Versions and hardware read out of this interpreter, never declared."""
    import numpy as np
    import torch

    import lerobot
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": __import__("torchvision").__version__,
        "lerobot": lerobot.__version__,
        "lerobot_codebase_version": CODEBASE_VERSION,
        "gpu": _nvidia_smi("--query-gpu=name"),
        "driver": _nvidia_smi("--query-gpu=driver_version"),
        "cuda_available": bool(torch.cuda.is_available()),
    }


# --- Dataset ---------------------------------------------------------------------


def read_val_split(lerobot_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Load ``val_split.json`` and check it describes *this* raw dataset.

    The sidecar carries the ``dataset_id`` of the raw dataset it was converted
    from. If that does not match the manifest on disk, the converted copy is
    stale — training would silently learn from one dataset while the run's
    provenance claimed another, which is precisely the failure the content hash
    exists to catch.
    """
    path = lerobot_dir / "val_split.json"
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing; re-run scripts/convert_dataset.py --dataset {lerobot_dir.name}"
        )
    split = json.loads(path.read_text(encoding="utf-8"))
    expected, found = manifest.get("dataset_id"), split.get("dataset_id")
    if expected and found and expected != found:
        raise SystemExit(
            f"{path} was written from dataset_id {found[:12]} but the raw manifest now says "
            f"{expected[:12]}; re-convert before training"
        )
    if not split.get("train_episode_indices"):
        raise SystemExit(f"{path} lists no training episodes")
    return split


class ResilientDataset:
    """A dataset wrapper that survives a single bad video decode.

    torchcodec can fail a random-access frame read with ``Could not push packet
    to decoder: Invalid data found``. One reproducible cause is fixed by
    ``--num-workers 0`` (see :func:`main`); this wrapper covers the residue,
    because a twelve-hour run must not die at a random step for a frame it
    could have read on the second try. Retries land on the same index first;
    only a genuinely unreadable frame falls through to its neighbour, and every
    retry is counted into ``run.json`` so a *rising* count is visible rather
    than silently absorbed.
    """

    def __init__(self, inner: Any, retries: int = 2) -> None:
        self.inner = inner
        self.retries = retries

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> Any:
        for attempt in range(self.retries + 1):
            try:
                return self.inner[index]
            except RuntimeError as error:
                if "decod" not in str(error).lower() and "packet" not in str(error).lower():
                    raise
                with _DECODE_RETRIES.get_lock():
                    _DECODE_RETRIES.value += 1
                print(f"  decode retry {attempt + 1} at index {index}: {error}", file=sys.stderr)
        return self.inner[(index + 1) % len(self.inner)]


def make_dataset(
    repo_id: str, lerobot_dir: Path, episodes: list[int], chunk: int, fps: int
) -> Any:
    """One :class:`LeRobotDataset` restricted to `episodes`, sliced into chunks.

    ``delta_timestamps`` is what turns a per-frame dataset into ACT's
    supervision: every item carries the next `chunk` actions plus the
    ``action_is_pad`` mask that ``ACTPolicy.forward`` needs to ignore the tail
    of an episode.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id,
        root=lerobot_dir,
        episodes=sorted(episodes),
        delta_timestamps={TASK_FPS_KEY: [index / fps for index in range(chunk)]},
    )


def fixed_val_batches(size: int, batch_size: int, cap: int) -> list[list[int]]:
    """Validation batches as a deterministic, contiguous index partition.

    Fixed order and fixed membership: the val number has to move only because
    the policy moved. The trailing short batch is kept (dropping it would throw
    away real frames), and `cap` bounds the pass for datasets where a full
    sweep would cost more than the training steps between two of them.
    """
    batches = [
        list(range(start, min(start + batch_size, size)))
        for start in range(0, size, batch_size)
    ]
    return batches[:cap] if cap else batches


# --- Training --------------------------------------------------------------------


def build_policy(config: Any, dataset: Any) -> tuple[Any, Any, Any]:
    """Policy plus its pre/post-processing pipelines, from the dataset metadata.

    ``make_policy`` fills ``input_features``/``output_features`` from
    ``ds_meta`` (so the state and image shapes are the dataset's, not a
    literal), and ``make_pre_post_processors`` bakes that dataset's statistics
    into the normalize/unnormalize steps. Saving the pair alongside the weights
    is what lets the CPU policy server own normalization later without the
    dataset being present at all.
    """
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        config, dataset_stats=dataset.meta.stats
    )
    return policy, preprocessor, postprocessor


def train_step(policy: Any, preprocessor: Any, optimizer: Any, batch: Any) -> dict[str, float]:
    """One forward/backward/step. Returns the loss terms as plain floats."""
    import torch

    policy.train()
    loss, parts = policy.forward(preprocessor(batch))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach()), **{key: float(value) for key, value in parts.items()}}


def validate(
    policy: Any, preprocessor: Any, dataset: Any, batches: list[list[int]]
) -> dict[str, Any]:
    """Mean loss over the validation episodes, in eval mode, without gradients.

    Batches are collated here rather than through a ``DataLoader`` so the order
    is exactly the one :func:`fixed_val_batches` computed, with no worker
    scheduling in between.
    """
    import torch
    from torch.utils.data import default_collate

    was_training = policy.training
    policy.eval()
    resilient = ResilientDataset(dataset)
    totals: dict[str, float] = {}
    frames = 0
    with torch.no_grad():
        for indices in batches:
            batch = default_collate([resilient[index] for index in indices])
            loss, parts = policy.forward(preprocessor(batch))
            weight = len(indices)
            totals["loss"] = totals.get("loss", 0.0) + float(loss) * weight
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value) * weight
            frames += weight
    if was_training:
        policy.train()
    return {key: value / max(frames, 1) for key, value in totals.items()} | {"frames": frames}


def probe_batch_size(
    config: Any, dataset: Any, start: int, ceiling: int, device: str, steps: int
) -> tuple[int, list[dict[str, Any]]]:
    """Halve the batch size until `steps` real training steps fit under `ceiling`.

    Deliberately a throwaway policy and optimizer: the point is to learn what
    the *peak* costs, and the peak of a run includes the optimizer's Adam
    moments, so a probe that skipped them would under-report by a third. Every
    rung of the ladder is returned, including the ones that did not fit, so
    ``run.json`` shows why the chosen batch size is the one it is.
    """
    import torch
    from torch.utils.data import DataLoader

    ladder: list[dict[str, Any]] = []
    batch_size = start
    while batch_size >= 1:
        policy, preprocessor, _ = build_policy(config, dataset)
        optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
        loader = DataLoader(
            ResilientDataset(dataset), batch_size=batch_size, shuffle=True, drop_last=True
        )
        rung: dict[str, Any] = {"batch_size": batch_size}
        try:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            iterator = iter(loader)
            for _ in range(steps):
                train_step(policy, preprocessor, optimizer, next(iterator))
            torch.cuda.synchronize()
            rung |= {
                "fits": None,
                "seconds_per_step": (time.perf_counter() - started) / steps,
                "allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "nvidia_smi_mib": process_vram_mib(os.getpid()),
            }
            # nvidia-smi is the binding number: it counts the CUDA context and
            # the allocator's reserve, which are just as unavailable to the
            # machine's other user as the tensors themselves.
            reported = rung["nvidia_smi_mib"] or rung["reserved_mib"]
            rung["fits"] = reported <= ceiling
        except torch.OutOfMemoryError as error:
            rung |= {"fits": False, "oom": str(error).splitlines()[0]}
        finally:
            del loader
            del optimizer
            del policy
            torch.cuda.empty_cache()
        ladder.append(rung)
        print(
            f"  probe batch {batch_size}: "
            + (
                f"{rung.get('nvidia_smi_mib')} MiB nvidia-smi / "
                f"{rung.get('allocated_mib', 0):.0f} MiB allocated, "
                f"{rung.get('seconds_per_step', 0) * 1e3:.0f} ms/step"
                if "oom" not in rung
                else "OOM"
            )
            + f" -> {'fits' if rung['fits'] else 'too big'}"
        )
        if rung["fits"]:
            return batch_size, ladder
        batch_size //= 2
    raise SystemExit(f"even batch size 1 does not fit under {ceiling} MiB")


# --- Checkpoints -----------------------------------------------------------------


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    """Directory one checkpoint lives in; zero-padded so the names sort."""
    return run_dir / "checkpoints" / f"{step:08d}"


def checkpoint_steps(run_dir: Path) -> list[int]:
    """Steps that currently have a checkpoint on disk, ascending."""
    root = run_dir / "checkpoints"
    if not root.is_dir():
        return []
    return sorted(
        int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit()
    )


def retention_keep(steps: list[int], last: int, best: int | None, every: int) -> set[int]:
    """The plan's retention set: last, best-by-val, and every `every` steps."""
    keep = {last} | {step for step in steps if step % every == 0}
    if best is not None:
        keep.add(best)
    return keep


def _link(path: Path, target: Path) -> None:
    """Point `path` at sibling `target`, replacing whatever was there."""
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target.name)


def save_checkpoint(
    run_dir: Path,
    step: int,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    optimizer: Any,
    best_step: int | None,
) -> Path:
    """Write one checkpoint and immediately enforce retention.

    lerobot's own layout — ``pretrained_model/`` (weights, policy config, and
    the two processor pipelines) beside ``training_state/`` (optimizer moments,
    RNG, step counter) — because ``load_training_state`` is what makes
    ``--resume`` restore Adam's moments rather than restarting them at zero,
    which for a mid-run resume is the difference between continuing and taking
    a hundred steps of damage.

    Retention runs *here*, not at the end: a run that is killed has already
    pruned everything it was going to prune, so the disk budget holds even for
    a run that never reaches its last line.
    """
    from lerobot.common.train_utils import save_training_state
    from lerobot.utils.constants import PRETRAINED_MODEL_DIR

    directory = checkpoint_dir(run_dir, step)
    pretrained = directory / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained)
    preprocessor.save_pretrained(pretrained)
    postprocessor.save_pretrained(pretrained)
    save_training_state(directory, step, optimizer, None)

    keep = retention_keep(checkpoint_steps(run_dir), step, best_step, RETAIN_EVERY)
    for existing in checkpoint_steps(run_dir):
        if existing not in keep:
            shutil.rmtree(checkpoint_dir(run_dir, existing))
    _link(run_dir / "checkpoints" / "last", directory)
    # Only ever point `best` at something that is actually there. It always is,
    # because a validation improvement is itself a save trigger — but a
    # dangling symlink is exactly the kind of thing an eval run would discover
    # an hour later, so the invariant is checked rather than assumed.
    if best_step is not None and checkpoint_dir(run_dir, best_step).is_dir():
        _link(run_dir / "checkpoints" / "best", checkpoint_dir(run_dir, best_step))
    print(
        f"  checkpoint {step} -> {directory.relative_to(run_dir)} "
        f"(keeping {sorted(keep)}, best {best_step})"
    )
    return directory


def load_config(pretrained: Path, device: str) -> Any:
    """The policy config as the checkpoint recorded it, retargeted at `device`.

    Reloaded rather than rebuilt: ``config.json`` carries the
    ``input_features``/``output_features`` that ``make_policy`` derived from
    the dataset metadata at run start, and a bare ``ACTConfig()`` has none of
    them (``validate_features`` rejects it outright). Only the device is ours
    to change — that is what lets the same checkpoint resume on the GPU and
    serve on the CPU.
    """
    from lerobot.policies.act.configuration_act import ACTConfig

    config = ACTConfig.from_pretrained(pretrained)
    config.device = device
    return config


def load_checkpoint(directory: Path, device: str) -> tuple[Any, Any, Any, Any, Any, int]:
    """Restore config, policy, processors, optimizer *and* step from a checkpoint.

    The optimizer is constructed **here**, from the freshly loaded policy's own
    parameters, and only then filled from disk. Building it earlier — against a
    policy that ``from_pretrained`` then replaces — produces a run that looks
    healthy and trains nothing at all: ``optimizer.step()`` updates tensors no
    longer referenced by the model, and the only visible symptom is a
    validation loss that never moves again. That failure was observed here
    before this function owned the optimizer, which is why it does.
    """
    from lerobot.common.train_utils import load_training_state
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import PRETRAINED_MODEL_DIR

    pretrained = directory / PRETRAINED_MODEL_DIR
    config = load_config(pretrained, device)
    policy = ACTPolicy.from_pretrained(pretrained, config=config)
    policy.to(config.device)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(pretrained))
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    step, optimizer, _ = load_training_state(directory, optimizer, None)
    assert_optimizer_owns(policy, optimizer)
    return config, policy, preprocessor, postprocessor, optimizer, step


def assert_optimizer_owns(policy: Any, optimizer: Any) -> None:
    """Fail loudly if the optimizer is not stepping this policy's tensors."""
    owned = {id(tensor) for tensor in policy.parameters()}
    stepped = {
        id(tensor) for group in optimizer.param_groups for tensor in group["params"]
    }
    if not stepped or not stepped <= owned:
        raise RuntimeError(
            f"optimizer holds {len(stepped - owned)} parameters the policy does not own; "
            "training would silently update nothing"
        )


# --- Reporting -------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: Any, spec: str = ".3f") -> str:
    """Format a number, or an em dash when the run never produced one."""
    return format(value, spec) if isinstance(value, (int, float)) else "—"


def loss_summary(curve: list[dict[str, Any]], window: int) -> dict[str, Any]:
    """First-window versus last-window mean of the training loss.

    ``decreased`` is only a claim when the two windows are disjoint: a 20-step
    shake-out run would otherwise compare a window with itself and report a
    tie as a failure to learn.
    """
    total = [point["loss"] for point in curve]
    l1 = [point.get("l1_loss") for point in curve if point.get("l1_loss") is not None]
    first, last = _mean(total[:window]), _mean(total[-window:])
    disjoint = len(total) >= 2 * window
    return {
        "window": window,
        "steps": len(total),
        "windows_disjoint": disjoint,
        "first_mean": first,
        "last_mean": last,
        "decreased": bool(last < first) if disjoint and first is not None else None,
        "l1_first_mean": _mean(l1[:window]),
        "l1_last_mean": _mean(l1[-window:]),
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON atomically, indented so a diff is readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_report(run_dir: Path, run: dict[str, Any]) -> Path:
    """A human-readable sibling of ``run.json`` (tracked by the gitignore negation)."""
    val = run["val"]
    lines = [
        f"# {run['run_name']}",
        "",
        f"ACT on `{run['dataset_name']}` (`{run['dataset_id'][:12]}`), "
        f"{run['progress']['steps_done']}/{run['progress']['steps_target']} steps in "
        f"{run['progress']['total_wall_clock_s'] / 60:.1f} min"
        + (
            f" across two processes (resumed at step {run['progress']['resumed_from_step']})."
            if run["progress"]["resumed_from_step"] is not None
            else "."
        ),
        "",
        "| | |",
        "|---|---|",
        f"| batch / chunk | {run['config']['batch_size']} / {run['config']['chunk_size']} |",
        f"| seconds per step | {run['progress']['seconds_per_step']:.3f} |",
        f"| VRAM peak (nvidia-smi) | {run['vram']['peak_nvidia_smi_mib']} MiB "
        f"(ceiling {run['vram']['ceiling_mib']}) |",
        f"| VRAM peak (allocated) | {_fmt(run['vram']['peak_allocated_mib'], '.0f')} MiB |",
        f"| train loss {run['train_loss']['window']}-step mean | "
        f"{_fmt(run['train_loss']['first_mean'])} -> {_fmt(run['train_loss']['last_mean'])}"
        f"{'' if run['train_loss']['windows_disjoint'] else ' (windows overlap)'} |",
        f"| best val loss | {_fmt(val['best']['loss'], '.4f')} at step {val['best']['step']} |"
        if val.get("best")
        else "| best val loss | — |",
        f"| stopped because | {run['progress']['stop_reason']} |",
        "",
        "## Validation curve",
        "",
        "| step | val loss (L1) | train loss (100-step mean) |",
        "|---|---|---|",
    ]
    for point in val["curve"]:
        lines.append(
            f"| {point['step']} | {_fmt(point['loss'], '.4f')} | "
            f"{_fmt(point.get('train_loss_mean'))} |"
        )
    lines += ["", "Provenance: `run.json` here. Generated by `scripts/train_act.py`."]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- Entry point -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default=None, help="dataset name, e.g. grasp_cube_dev")
    parser.add_argument("--steps", type=int, default=2000, help="total training steps to reach")
    parser.add_argument("--run-name", default="auto", help="run directory name, or 'auto'")
    parser.add_argument(
        "--ceiling-minutes",
        type=float,
        default=45.0,
        help="wall-clock ceiling; stop cleanly at it",
    )
    parser.add_argument("--resume", default=None, help="run name under runs/train/ to continue")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="probe starts here")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="action chunk length")
    parser.add_argument("--seed", type=int, default=1000, help="torch/numpy/random seed")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument(
        "--val-every", type=int, default=VAL_EVERY, help="steps between val passes"
    )
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="steps between saves")
    parser.add_argument(
        "--val-batches", type=int, default=0, help="cap the val pass to N batches (0 = all)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="dataloader workers; 0 is the safe default (see the note in main)",
    )
    parser.add_argument("--probe-steps", type=int, default=PROBE_STEPS, help="VRAM probe length")
    parser.add_argument(
        "--no-probe", action="store_true", help="trust --batch-size, skip the probe"
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    return parser


class Stopper:
    """Turns SIGTERM/SIGINT into a clean stop after the current step.

    A shared GPU means somebody else may need the card back mid-run, and a
    twelve-hour job that dies without a checkpoint has to start over. The
    signal only sets a flag; the loop notices it, writes a checkpoint and
    reports the run as interrupted. A second signal is left to the default
    handler, so an operator who really means *now* still gets it.
    """

    def __init__(self) -> None:
        self.reason: str | None = None
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, self._handle)

    def _handle(self, number: int, _frame: Any) -> None:
        self.reason = f"signal_{signal.Signals(number).name.lower()}"
        signal.signal(number, signal.SIG_DFL)
        print(f"\n{self.reason}: finishing the current step, then checkpointing", flush=True)

    def __bool__(self) -> bool:
        return self.reason is not None


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - one linear run script
    """Train (or resume) one run. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from manus import recorder

    resumed_run: dict[str, Any] | None = None
    if args.resume:
        run_dir = args.root / "runs" / KIND / args.resume
        if not run_dir.is_dir():
            raise SystemExit(f"no such run: {run_dir}")
        resumed_run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        # The resumed run's shape is not negotiable: a different chunk size or
        # batch size would silently invalidate the optimizer state being
        # restored, so the recorded values win over the CLI defaults.
        args.dataset = resumed_run["dataset_name"]
        args.chunk = resumed_run["config"]["chunk_size"]
        args.batch_size = resumed_run["config"]["batch_size"]
        args.seed = resumed_run["config"]["seed"]

    if not args.dataset:
        raise SystemExit("--dataset is required (or --resume a run that recorded one)")

    dataset_dir = args.root / "datasets" / "raw" / args.dataset
    lerobot_dir = args.root / "datasets" / "lerobot" / args.dataset
    if not lerobot_dir.is_dir():
        raise SystemExit(
            f"{lerobot_dir} is missing; run scripts/convert_dataset.py --dataset {args.dataset}"
        )
    manifest = recorder.read_manifest(dataset_dir)
    split = read_val_split(lerobot_dir, manifest)
    git = git_provenance(args.root)

    if not args.resume:
        name = (
            stamped_run_name(KIND, args.dataset, git["sha"])
            if args.run_name == "auto"
            else args.run_name
        )
        run_dir = args.root / "runs" / KIND / name
        run_dir.mkdir(parents=True, exist_ok=True)
    name = run_dir.name

    preflight_free = None
    if args.device.startswith("cuda"):
        preflight_free = free_vram_mib()
        if preflight_free is not None and preflight_free < MIN_FREE_VRAM_MIB:
            raise SystemExit(
                f"pre-flight: only {preflight_free} MiB of VRAM free, need >= "
                f"{MIN_FREE_VRAM_MIB} MiB (shared GPU: one process at a time)"
            )
        print(f"pre-flight: {preflight_free} MiB free VRAM, ceiling {VRAM_CEILING_MIB} MiB ours")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_id = f"manus/{args.dataset}"
    fps = int(manifest["temporal"]["control_hz"])
    train_data = make_dataset(
        repo_id, lerobot_dir, split["train_episode_indices"], args.chunk, fps
    )
    val_data = make_dataset(repo_id, lerobot_dir, split["val_episode_indices"], args.chunk, fps)
    print(
        f"{args.dataset}: train {train_data.num_episodes} episodes / "
        f"{train_data.num_frames} frames, "
        f"val {val_data.num_episodes} / {val_data.num_frames} (chunk {args.chunk} @ {fps} Hz)"
    )

    from lerobot.policies.act.configuration_act import ACTConfig

    config = ACTConfig(chunk_size=args.chunk, n_action_steps=args.chunk, device=args.device)

    ladder: list[dict[str, Any]] = resumed_run["vram"]["probe"] if resumed_run else []
    if not args.resume and not args.no_probe and args.device.startswith("cuda"):
        print(f"VRAM probe ({args.probe_steps} steps, ceiling {VRAM_CEILING_MIB} MiB):")
        args.batch_size, ladder = probe_batch_size(
            config, train_data, args.batch_size, VRAM_CEILING_MIB, args.device, args.probe_steps
        )
        print(f"batch size {args.batch_size}")

    start_step = 0
    if args.resume:
        last = (run_dir / "checkpoints" / "last").resolve()
        if not last.is_dir():
            raise SystemExit(f"{run_dir}/checkpoints/last does not resolve to a checkpoint")
        config, policy, preprocessor, postprocessor, optimizer, start_step = load_checkpoint(
            last, args.device
        )
        if config.chunk_size != args.chunk:
            raise SystemExit(
                f"{last} was trained with chunk {config.chunk_size}, not {args.chunk}"
            )
        print(f"resumed {name} from {last.name} at step {start_step}")
    else:
        policy, preprocessor, postprocessor = build_policy(config, train_data)
        optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
        assert_optimizer_owns(policy, optimizer)

    val_batches = fixed_val_batches(len(val_data), args.batch_size, args.val_batches)
    if args.num_workers:
        # lerobot.datasets.dataset_reader._query_videos says it outright: "When
        # using data workers ... do not call this function in the main process
        # ... It will result in a Segmentation Fault." Validation reads happen
        # here in the main process, and train and val share one video file and
        # one module-global VideoDecoderCache, so a forked worker and the
        # parent end up seeking the same file description. Observed as
        # torchcodec's "Could not push packet to decoder: Invalid data found".
        print(
            f"WARNING: --num-workers {args.num_workers} with main-process validation is the "
            "configuration lerobot warns segfaults; 0 costs ~35% of a step and is safe",
            file=sys.stderr,
        )
    generator = torch.Generator().manual_seed(args.seed + start_step)
    loader = DataLoader(
        ResilientDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )

    stopper = Stopper()
    curve: list[dict[str, Any]] = list(resumed_run["train_loss_curve"]) if resumed_run else []
    val_curve: list[dict[str, Any]] = list(resumed_run["val"]["curve"]) if resumed_run else []
    best = min(val_curve, key=lambda point: point["loss"]) if val_curve else None
    peak_smi = (resumed_run["vram"]["peak_nvidia_smi_mib"] or 0) if resumed_run else 0
    saved: list[int] = []

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    ceiling_seconds = args.ceiling_minutes * 60.0
    started = time.time()
    step = start_step
    iterator = iter(loader)
    stop_reason = "steps_reached"
    window: list[float] = []

    validated_at = start_step if resumed_run else None

    def do_validate() -> bool:
        """Run one validation pass. Returns whether it is a new best."""
        nonlocal best, validated_at
        point = validate(policy, preprocessor, val_data, val_batches)
        record = {
            "step": step,
            "loss": point["loss"],
            "l1_loss": point.get("l1_loss"),
            "frames": point["frames"],
            "train_loss_mean": _mean(window),
        }
        val_curve.append(record)
        validated_at = step
        improved = best is None or record["loss"] < best["loss"]
        if improved:
            best = record
        print(
            f"  step {step}: val {record['loss']:.4f} "
            f"(best {best['loss']:.4f} @ {best['step']})"
        )
        return improved

    def do_save() -> None:
        """Checkpoint this step, at most once. Idempotent within a step."""
        nonlocal peak_smi
        if saved and saved[-1] == step:
            return
        save_checkpoint(
            run_dir, step, policy, preprocessor, postprocessor, optimizer,
            best["step"] if best else None,
        )
        saved.append(step)
        peak_smi = max(peak_smi, process_vram_mib(os.getpid()) or 0)

    while step < args.steps:
        if time.time() - started > ceiling_seconds:
            stop_reason = "ceiling_minutes"
            break
        if stopper:
            stop_reason = stopper.reason or "signal"
            break
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        parts = train_step(policy, preprocessor, optimizer, batch)
        step += 1
        curve.append({"step": step, **parts})
        window.append(parts["loss"])
        if len(window) > LOSS_WINDOW:
            window.pop(0)
        if step % 50 == 0:
            elapsed = time.time() - started
            print(
                f"step {step}/{args.steps}  loss {parts['loss']:.3f} "
                f"(l1 {parts.get('l1_loss', float('nan')):.4f})  "
                f"{elapsed / max(step - start_step, 1):.3f} s/step  {elapsed / 60:.1f} min",
                flush=True,
            )
        # A new best validation loss is itself a save trigger. Without that,
        # "retention keeps the best-by-val checkpoint" is a promise about a
        # directory that may never have been written: val runs every 500 steps
        # and checkpoints land every 2000, so three out of four best-val steps
        # would have nothing on disk to keep. Retention prunes the extras
        # immediately, so the disk cost stays at last + best + every-10k.
        improved = do_validate() if step % args.val_every == 0 else False
        if improved or step % args.save_every == 0:
            do_save()

    # A run always ends on a checkpoint, whatever ended it: the ceiling, a
    # signal, or the step count. Validating first (unless this very step was
    # just validated) means the final checkpoint can itself be best-by-val.
    if step != start_step:
        if validated_at != step:
            do_validate()
        do_save()
    wall_clock = time.time() - started

    on_cuda = args.device.startswith("cuda")
    peak_allocated = torch.cuda.max_memory_allocated() / 2**20 if on_cuda else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / 2**20 if on_cuda else 0.0
    # Across every process that contributed, not just this one: a run that was
    # killed and resumed cost what all of its halves cost.
    total_wall_clock = (
        resumed_run["progress"]["total_wall_clock_s"] if resumed_run else 0.0
    ) + wall_clock
    run = {
        "kind": KIND,
        "run_name": name,
        "dataset_name": args.dataset,
        "dataset_id": manifest.get("dataset_id"),
        "lerobot": {
            "path": str(lerobot_dir.relative_to(args.root)),
            "repo_id": repo_id,
            "fps": fps,
            "train_episodes": len(split["train_episode_indices"]),
            "val_episodes": len(split["val_episode_indices"]),
            "train_frames": int(train_data.num_frames),
            "val_frames": int(val_data.num_frames),
        },
        "converter": {
            "script": "scripts/convert_dataset.py",
            "val_split_rule": split.get("rule"),
            "val_split_source": split.get("source"),
            "val_split_dataset_id": split.get("dataset_id"),
            "raw_manifest_dataset_id": manifest.get("dataset_id"),
            "raw_numpy": (manifest.get("env") or {}).get("numpy"),
        },
        "git": git,
        "env": env_block(),
        "config": {
            "policy": config.type,
            "chunk_size": args.chunk,
            "n_action_steps": config.n_action_steps,
            "chunk_rationale": (
                "50 steps = 1.67 s at 30 Hz; episodes average "
                f"{manifest['counts']['frames'] / max(manifest['counts']['successes'], 1):.0f} "
                "frames, so lerobot's default 100 would span over half a demonstration and "
                "pad the tail of every late chunk"
            ),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
            "optimizer_lr": config.optimizer_lr,
            "optimizer_lr_backbone": config.optimizer_lr_backbone,
            "optimizer_weight_decay": config.optimizer_weight_decay,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "kl_weight": config.kl_weight,
            "use_vae": config.use_vae,
            "use_amp": config.use_amp,
            "vision_backbone": config.vision_backbone,
            "num_workers": args.num_workers,
            "val_every": args.val_every,
            "save_every": args.save_every,
            "retention": {"last": True, "best_by_val": True, "every": RETAIN_EVERY},
        },
        "vram": {
            "ceiling_mib": VRAM_CEILING_MIB,
            # Measured before the process touched the card, not at write time:
            # by the end we are holding most of what we would have reported.
            "preflight_free_mib": preflight_free,
            "free_mib_at_finish": free_vram_mib(),
            "probe": ladder,
            "probe_steps": args.probe_steps,
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
            "peak_nvidia_smi_mib": max(peak_smi, process_vram_mib(os.getpid()) or 0),
        },
        "progress": {
            "resumed_from_step": start_step if args.resume else None,
            "steps_done": step,
            "steps_this_process": step - start_step,
            "steps_target": args.steps,
            "wall_clock_s": wall_clock,
            "total_wall_clock_s": total_wall_clock,
            "seconds_per_step": wall_clock / max(step - start_step, 1),
            "ceiling_minutes": args.ceiling_minutes,
            "ceiling_hit": stop_reason == "ceiling_minutes",
            "stop_reason": stop_reason,
            "decode_retries": _DECODE_RETRIES.value,
        },
        "train_loss": loss_summary(curve, LOSS_WINDOW),
        "train_loss_curve": curve,
        "val": {
            "every": args.val_every,
            "batches": len(val_batches),
            "mode": "eval() + no_grad; ACT skips its VAE encoder outside training, so this is L1",
            "curve": val_curve,
            "best": best,
        },
        "checkpoints": {
            "dir": "checkpoints",
            "saved_this_process": saved,
            "on_disk": checkpoint_steps(run_dir),
            "best_step": best["step"] if best else None,
            "last": f"checkpoints/{step:08d}",
        },
        "result": (
            f"{step} steps, val {_fmt(best['loss'], '.4f')} @ {best['step']}"
            if best
            else f"{step} steps"
        )
        + f", {max(peak_smi, 0)} MiB, {total_wall_clock / 60:.0f} min"
        + (" (resumed)" if args.resume else "")
        + (" (ceiling)" if stop_reason == "ceiling_minutes" else "")
        + (" (interrupted)" if stop_reason.startswith("signal") else ""),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(run_dir / "run.json", run)
    write_report(run_dir, run)

    summary = run["train_loss"]
    print(
        f"\n{name}: {step} steps in {wall_clock / 60:.1f} min "
        f"({run['progress']['seconds_per_step']:.3f} s/step), stopped on {stop_reason}"
    )
    verdict = (
        "windows overlap, not a claim"
        if not summary["windows_disjoint"]
        else "decreased"
        if summary["decreased"]
        else "DID NOT DECREASE"
    )
    print(
        f"train loss {LOSS_WINDOW}-step mean: {_fmt(summary['first_mean'])} -> "
        f"{_fmt(summary['last_mean'])} ({verdict})"
    )
    if best:
        print(
            f"val: {len(val_curve)} points, best {_fmt(best['loss'], '.4f')} "
            f"at step {best['step']}"
        )
    print(
        f"vram peak: {run['vram']['peak_nvidia_smi_mib']} MiB nvidia-smi / "
        f"{peak_allocated:.0f} MiB allocated (ceiling {VRAM_CEILING_MIB})"
    )
    print(f"checkpoints on disk: {checkpoint_steps(run_dir)}")
    print(f"STATUS: {'complete' if step >= args.steps else 'pending'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
