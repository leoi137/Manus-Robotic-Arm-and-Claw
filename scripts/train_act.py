"""Train an ACT policy on one or more converted LeRobot datasets. Runs in `.venv-lerobot`.

.. code-block:: bash

    # Stage 1: the five-object mixture, the run this script exists for
    ./.venv-lerobot/bin/python scripts/train_act.py --datasets all \
        --steps 100000 --ceiling-minutes 900 --vram-ceiling-mib 13000

    # a single dataset, the original mode
    ./.venv-lerobot/bin/python scripts/train_act.py --dataset grasp_cube_dev \
        --steps 2000 --ceiling-minutes 45
    ./.venv-lerobot/bin/python scripts/train_act.py --resume train__grasp_cube_dev__...

--------------------------------------------------------------------------------
A LESSON IN WHAT THIS SCRIPT IS ACTUALLY DOING
--------------------------------------------------------------------------------

If you know transformers but not robot imitation learning, five ideas carry the
whole file. Read these once and the rest of the code is bookkeeping.

**1. What one training example is.** In language modelling an example is a span
of tokens. Here an example is a *(observation, action chunk)* pair sampled at a
random frame *t* of a random episode:

    inputs   observation.images.wrist  (1, 3, 240, 320)  one RGB frame at time t
             observation.state         (6,)              six joint positions at t
    target   action                    (chunk, 6)        the joints at t..t+chunk-1
             action_is_pad             (chunk,)          True where the episode ran out

There is no history and no recurrence: ACT looks at a *single* moment and
predicts the next ~1.7 seconds of motion. That is the entire "action chunking"
idea. `delta_timestamps` in :func:`make_dataset` is what turns the flat
per-frame dataset into these pairs — it asks LeRobot for frames at
`t + 0/fps, t + 1/fps, ... t + (chunk-1)/fps` of the `action` column.

**2. Why chunks at all.** A policy that predicts one step at a time must be
queried at 30 Hz and drifts: tiny per-step errors compound, and it stutters when
the demonstrations were multi-modal (two equally good ways to reach the cube).
Predicting a whole chunk in one shot makes the policy commit to *one* of those
modes for 1.7 s, and cuts the effective horizon the policy must be accurate over
by 50x. The eval client then blends overlapping chunks (temporal ensembling) so
the executed trajectory is still smooth.

**3. What the CVAE latent is for.** Human/expert demonstrations are noisy: for
the same image the recorded action can differ run to run. A plain regressor
averages those alternatives into a mushy middle. ACT is a *conditional VAE*: at
TRAIN time only, an extra transformer encoder reads the ground-truth action
chunk and emits a latent `z` (dim 32) capturing "which style of motion this
particular demonstration was". The decoder gets image + state + `z`, so the
variation it cannot predict from pixels is explained by `z` instead of being
smeared into the mean. A KL term pulls `z` toward N(0, I). At INFERENCE `z = 0`
— the mode of the prior — so you get the single most typical motion. This is
why train loss (L1 + kl_weight*KL) and val loss (pure L1, VAE encoder off) are
*not* comparable numbers; only each with itself over time.

**4. Why an ImageNet-pretrained ResNet-18.** We have ~1.2M frames, which sounds
like a lot but they are 5000 near-identical tabletop scenes. Training a vision
backbone from scratch on that would learn our exact lighting and nothing about
edges, corners or shading. ImageNet weights arrive already knowing generic
low-level vision; fine-tuning them gently adapts that to our scene instead of
relearning it. lerobot 0.6.1 sets `optimizer_lr` and `optimizer_lr_backbone`
both to 1e-5 (the original ACT paper's values; the separate backbone group
exists so you *can* freeze or slow it, and DETR-lineage code often runs it 10x
lower) — we keep the proven defaults rather than inventing our own, because
Stage 1 is a diagnostic of the DATA and every hyperparameter we change is a
confounder if it fails. Note also `norm_layer=FrozenBatchNorm2d`: BatchNorm
statistics are garbage at batch sizes this small, so the ImageNet running
stats are frozen and used as constants. The transformer on top is trained from
scratch, because "where is the gripper relative to the cube" is not a thing
ImageNet ever had to answer.

**5. What normalization stats are and why they matter here.** The dataset ships
`meta/stats.json`: per-feature mean/std/min/max computed over every frame. The
preprocessor standardizes state and actions with them; the postprocessor undoes
it on the way out. Without this, the loss would be dominated by whichever joint
happens to have the largest numerical range and the gripper (small range, but
the joint that decides success or failure) would be ignored. **When mixing
datasets you must aggregate the stats across all of them** — see
:class:`MixtureDataset`. Training on cube-only statistics while feeding pingpong
frames is a silent, hard-to-debug way to poison a run.

Bonus, **why the action space is joint positions** and not end-effector poses or
torques: the SO-101 is a chain of position-controlled Hiwonder servos, so joint
targets are literally the wire format of the real robot. Predicting them means
no IK at inference (no solver failures, no branch flips) and the sim action
space is byte-identical to the real one — the one part of sim-to-real we get for
free. `observation.state` is the *measured* joint positions and `action` is the
*commanded* ones; they differ by servo lag, which is exactly the signal the
policy needs to learn to push through.

**Reading the loss curve.** Expect `l1_loss` (normalized units) to fall fast for
~2k steps, then grind down slowly for the rest of the run; total loss is
dominated early by the KL term. Healthy: val L1 tracking train L1 downward and
flattening around 0.02–0.05 by 50k–100k steps. Bad, and what each means:
  * flat from step 0 -> the optimizer is not stepping the policy's tensors
    (see :func:`assert_optimizer_owns`) or the LR is wrong;
  * train falls, val flat/rising -> overfitting; more objects in the mix or
    fewer steps;
  * both plateau high (> ~0.15) -> the observation cannot explain the action.
    That is the v1 image-blindness failure mode: if the wrist camera cannot see
    the object at episode start, no amount of training removes the ambiguity.
    Watch for this one specifically; it is why Stage 1 exists.

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

DEFAULT_DATASETS = [
    "grasp_cube_v2",
    "grasp_die_v2",
    "grasp_domino_v2",
    "grasp_duplo_v2",
    "grasp_pingpong_v2",
]
"""The Stage-1 mixture: ``--datasets all`` expands to this."""

DEFAULT_BATCH = 8
"""Batch size the VRAM probe starts from, per the plan.

The probe only ever *halves* this, so it is a ceiling, not a target. For the
Stage-1 run it is raised to 64 on the command line: ACT at 240x320 with a single
camera is small (~51M params, one ResNet-18 forward per sample), so on a
dedicated 16 GB 5080 the card is bandwidth-bound long before it is memory-bound
and a bigger batch buys real throughput. See :func:`probe_batch_size` — the
number that ends up in run.json is measured, never assumed.
"""

VRAM_CEILING_MIB = 5500
"""Default share of the card, sized for the *shared*-GPU contract (plan
§Environment facts). Override with ``--vram-ceiling-mib`` when the card is ours
alone, as it is for the Stage-1 run (13000 of the 5080's 16303 MiB, leaving
headroom for allocator fragmentation and the display context)."""

MIN_FREE_VRAM_MIB = 6500
"""Pre-flight floor on free VRAM before the process is allowed to start.
``--min-free-vram-mib`` overrides it."""

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
        # Walk to a *different* index on each retry, and protect every attempt
        # including the last. Retrying the same index is close to useless: when
        # torchcodec fails this way the decoder for that file is already in a
        # bad state, so attempt 2 and 3 fail identically. Observed exactly that
        # on the first Stage-1 launch — three retries on index 506501, then an
        # unprotected fall-through to its neighbour, which also raised and took
        # the whole run down at step ~1. (Both indices decode perfectly in a
        # single process, which is what proves this is decoder state and not
        # corrupt data; the real fix is the spawn start method, see main().)
        size = len(self.inner)
        for attempt in range(self.retries + 1):
            candidate = (index + attempt) % size
            try:
                return self.inner[candidate]
            except RuntimeError as error:
                if "decod" not in str(error).lower() and "packet" not in str(error).lower():
                    raise
                with _DECODE_RETRIES.get_lock():
                    _DECODE_RETRIES.value += 1
                print(
                    f"  decode retry {attempt + 1} at index {candidate}: {error}", file=sys.stderr
                )
        raise RuntimeError(
            f"{self.retries + 1} consecutive frames from index {index} failed to decode; "
            "this is no longer a transient error"
        )


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


class MixtureDataset:
    """Several :class:`LeRobotDataset` objects presented as one, with merged stats.

    **Why mix at all.** Stage 1 asks one policy to grasp five different objects
    from one wrist camera. Training five separate policies would answer a
    different question (and could not later take a language command). Because
    all five sets share the same robot, camera, fps and feature schema, mixing
    is just concatenation: index *i* of the mixture is index *i - offset* of
    whichever dataset that offset lands in.

    **The mixing ratio is implicit and that is deliberate.** Sampling is uniform
    over *frames*, so a dataset contributes in proportion to its frame count
    (pingpong's 294k frames get ~1.5x the gradient share of cube's 198k). That
    is the right default here: the longer episodes are longer precisely because
    those objects are harder, so they deserve more supervision. If a future run
    needs equal-per-object sampling, that is a WeightedRandomSampler on the
    DataLoader, not a change here.

    **The one thing you cannot concatenate naively is the statistics.**
    Normalization must be computed over the *mixture*, not over any one member
    (see the module docstring, point 5). :func:`aggregate_stats` does it
    properly — count-weighted means, a pooled std (not a mean of stds), and
    min/max as true extrema over all five.

    ``meta`` is borrowed from the first member because ``make_policy`` only
    reads the *schema* from it (which features exist and what shape they are),
    which :func:`assert_uniform_schema` has already proved identical across the
    mixture. The numbers that actually enter the model come from ``stats``.
    """

    def __init__(self, parts: list[Any], names: list[str]) -> None:
        from torch.utils.data import ConcatDataset

        from lerobot.datasets.compute_stats import aggregate_stats

        self.parts = parts
        self.names = names
        self.inner = ConcatDataset(parts)
        self.meta = parts[0].meta
        self.stats = aggregate_stats([part.meta.stats for part in parts])
        self.num_episodes = sum(int(part.num_episodes) for part in parts)
        self.num_frames = sum(int(part.num_frames) for part in parts)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> Any:
        return self.inner[index]

    def per_dataset(self) -> list[dict[str, Any]]:
        """Frame/episode counts per member, for ``run.json``."""
        return [
            {
                "name": name,
                "episodes": int(part.num_episodes),
                "frames": int(part.num_frames),
                "frame_share": int(part.num_frames) / max(self.num_frames, 1),
            }
            for name, part in zip(self.names, self.parts, strict=True)
        ]


def assert_uniform_schema(parts: list[Any], names: list[str]) -> None:
    """Refuse to mix datasets whose features or fps disagree.

    Concatenation silently "works" on mismatched sets — the collate function
    would only complain about shapes, not about, say, one dataset recorded at
    20 Hz. A mixture with a different fps in it is a mixture where the same
    chunk length means a different amount of *time* per member, which is a bug
    you would only notice as a policy that moves too fast on one object.
    """
    reference = {key: tuple(value["shape"]) for key, value in parts[0].meta.features.items()}
    for name, part in zip(names[1:], parts[1:], strict=True):
        found = {key: tuple(value["shape"]) for key, value in part.meta.features.items()}
        if found != reference:
            raise SystemExit(
                f"{name} has feature schema {found}, which differs from {names[0]}'s "
                f"{reference}; the mixture would be inconsistent"
            )
        if part.meta.fps != parts[0].meta.fps:
            raise SystemExit(
                f"{name} is {part.meta.fps} Hz but {names[0]} is {parts[0].meta.fps} Hz"
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

    # For a MixtureDataset ``.stats`` is the count-weighted aggregate over all
    # five members; for a plain LeRobotDataset it is that one dataset's own.
    # Either way the model is normalized by the statistics of exactly the data
    # it is about to be trained on, which is the only invariant that matters.
    stats = getattr(dataset, "stats", None)
    if stats is None:
        stats = dataset.meta.stats
    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
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


class CurveCSV:
    """Append-only CSV of the loss curve, flushed as the run goes.

    ``run.json`` is only written when the process *ends*, which is exactly the
    moment you cannot rely on during a twelve-hour job on a rented box. These
    two files exist so that ``tail -f runs/train/<run>/train_curve.csv`` is a
    live progress bar, and so the curve survives even a SIGKILL. Plain CSV on
    purpose: no wandb account, no server, and it opens in anything.
    """

    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path = path
        self.columns = columns
        self.handle = path.open("a", encoding="utf-8")
        if path.stat().st_size == 0:
            self.handle.write(",".join(columns) + "\n")
            self.handle.flush()

    def append(self, record: dict[str, Any], flush: bool = False) -> None:
        row = [record.get(column) for column in self.columns]
        self.handle.write(
            ",".join(
                ""
                if value is None
                else (f"{value:.6g}" if isinstance(value, float) else str(value))
                for value in row
            )
            + "\n"
        )
        if flush:
            self.handle.flush()

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


def plot_curves(run_dir: Path, curve: list[dict[str, Any]], val_curve: list[dict[str, Any]]) -> Path | None:
    """Write ``loss_curve.png``: train L1 (smoothed) against val L1.

    Deliberately L1 and not the total loss. Total loss includes the KL term,
    which is large and falls fast at the start, so a total-loss plot is a
    picture of the VAE settling down and hides the thing you care about — the
    action error. Log-y because the interesting part of the run is the last
    decade of improvement, which a linear axis flattens into a straight line.

    Returns None (never raises) if matplotlib is absent: a missing plot must not
    kill a run that has been training for nine hours. The CSVs are the source of
    truth; this is a convenience.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display on a headless rented box
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not curve:
        return None

    def smooth(values: list[float], window: int = 200) -> list[float]:
        # A running mean: per-step loss is dominated by which frames the random
        # sampler happened to draw, and that noise is bigger than the trend.
        out, total = [], 0.0
        for index, value in enumerate(values):
            total += value
            if index >= window:
                total -= values[index - window]
            out.append(total / min(index + 1, window))
        return out

    steps = [point["step"] for point in curve]
    l1 = [point.get("l1_loss") or point["loss"] for point in curve]
    figure, axes = plt.subplots(figsize=(8, 4.5), dpi=120)
    axes.plot(steps, l1, color="0.8", linewidth=0.5, label="train L1 (raw)")
    axes.plot(steps, smooth(l1), color="tab:blue", linewidth=1.5, label="train L1 (200-step mean)")
    if val_curve:
        axes.plot(
            [point["step"] for point in val_curve],
            [point.get("l1_loss") or point["loss"] for point in val_curve],
            color="tab:red", marker="o", markersize=3, linewidth=1.2, label="val L1 (held-out)",
        )
    axes.set_yscale("log")
    axes.set_xlabel("training step")
    axes.set_ylabel("L1 action error (normalized units)")
    axes.set_title(run_dir.name, fontsize=9)
    axes.grid(True, which="both", alpha=0.25)
    axes.legend(fontsize=8)
    figure.tight_layout()
    path = run_dir / "loss_curve.png"
    figure.savefig(path)
    plt.close(figure)
    return path


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
    parser.add_argument(
        "--datasets",
        default=None,
        help="comma-separated dataset names to mix, or 'all' for the five v2 grasp sets",
    )
    parser.add_argument(
        "--mix-name",
        default="grasp_v2_mix",
        help="short name for the mixture, used in the run directory name",
    )
    parser.add_argument(
        "--vram-ceiling-mib",
        type=int,
        default=VRAM_CEILING_MIB,
        help="our share of the card; the batch probe fits under this",
    )
    parser.add_argument(
        "--min-free-vram-mib",
        type=int,
        default=MIN_FREE_VRAM_MIB,
        help="pre-flight floor on free VRAM",
    )
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
    parser.add_argument(
        "--mp-context",
        default="spawn",
        choices=["spawn", "forkserver", "fork"],
        help="dataloader worker start method; 'fork' corrupts the video decoder cache",
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
        args.datasets = ",".join(resumed_run.get("dataset_names") or [resumed_run["dataset_name"]])
        args.chunk = resumed_run["config"]["chunk_size"]
        args.batch_size = resumed_run["config"]["batch_size"]
        args.seed = resumed_run["config"]["seed"]

    # One code path for one dataset and for five: `names` is always a list, and
    # a single-element list behaves exactly as the original single-dataset mode
    # did (same run-name, same run.json fields).
    if args.datasets:
        names = DEFAULT_DATASETS if args.datasets == "all" else [
            part.strip() for part in args.datasets.split(",") if part.strip()
        ]
    elif args.dataset:
        names = [args.dataset]
    else:
        raise SystemExit("--dataset or --datasets is required (or --resume a run that recorded one)")

    # The mixture's display name. A run directory called
    # `train__grasp_cube_v2+grasp_die_v2+...` would be unusable, so five sets
    # collapse to --mix-name while run.json keeps the exact list.
    args.dataset = names[0] if len(names) == 1 else args.mix_name

    manifests: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    lerobot_dirs: dict[str, Path] = {}
    for name in names:
        lerobot_dir = args.root / "datasets" / "lerobot" / name
        if not lerobot_dir.is_dir():
            raise SystemExit(
                f"{lerobot_dir} is missing; run scripts/convert_dataset.py --dataset {name}"
            )
        lerobot_dirs[name] = lerobot_dir
        # The raw manifest is the provenance anchor: it carries the dataset_id
        # that read_val_split checks the converted copy against, so a stale
        # conversion is caught here rather than discovered as a weird loss curve.
        manifests[name] = recorder.read_manifest(args.root / "datasets" / "raw" / name)
        splits[name] = read_val_split(lerobot_dir, manifests[name])
    manifest = manifests[names[0]]
    split = splits[names[0]]
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
        if preflight_free is not None and preflight_free < args.min_free_vram_mib:
            raise SystemExit(
                f"pre-flight: only {preflight_free} MiB of VRAM free, need >= "
                f"{args.min_free_vram_mib} MiB (shared GPU: one process at a time)"
            )
        print(
            f"pre-flight: {preflight_free} MiB free VRAM, "
            f"ceiling {args.vram_ceiling_mib} MiB ours"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_id = f"manus/{args.dataset}"
    fps = int(manifest["temporal"]["control_hz"])

    # Two mixtures, built from the SAME per-dataset episode split. The val
    # episodes of every member are held out of every member's training half, so
    # "held out" means held out of the whole mixture, not just of its own set.
    train_parts = [
        make_dataset(
            f"manus/{name}", lerobot_dirs[name], splits[name]["train_episode_indices"],
            args.chunk, fps,
        )
        for name in names
    ]
    val_parts = [
        make_dataset(
            f"manus/{name}", lerobot_dirs[name], splits[name]["val_episode_indices"],
            args.chunk, fps,
        )
        for name in names
    ]
    assert_uniform_schema(train_parts, names)
    train_data = MixtureDataset(train_parts, names)
    val_data = MixtureDataset(val_parts, names)
    for row in train_data.per_dataset():
        print(
            f"  {row['name']:<20} {row['episodes']:>5} eps  {row['frames']:>8} frames  "
            f"{row['frame_share'] * 100:5.1f}% of gradient share"
        )
    print(
        f"{args.dataset}: train {train_data.num_episodes} episodes / "
        f"{train_data.num_frames} frames, "
        f"val {val_data.num_episodes} / {val_data.num_frames} (chunk {args.chunk} @ {fps} Hz)"
    )

    from lerobot.policies.act.configuration_act import ACTConfig

    config = ACTConfig(chunk_size=args.chunk, n_action_steps=args.chunk, device=args.device)

    ladder: list[dict[str, Any]] = resumed_run["vram"]["probe"] if resumed_run else []
    if not args.resume and not args.no_probe and args.device.startswith("cuda"):
        print(f"VRAM probe ({args.probe_steps} steps, ceiling {args.vram_ceiling_mib} MiB):")
        args.batch_size, ladder = probe_batch_size(
            config, train_data, args.batch_size, args.vram_ceiling_mib,
            args.device, args.probe_steps,
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
    # Workers are worth a lot here: measured on the 5080, batch 64 costs
    # 0.94 s/step with num_workers=0 and 0.14 s/step with 6 workers — 6.8x,
    # because decoding AV1 video for 64 samples is far more expensive than the
    # forward/backward pass. Without workers the GPU sits at 0% utilization.
    #
    # But the naive version of this crashes, and the reason is worth knowing.
    # lerobot.datasets.dataset_reader._query_videos warns: "When using data
    # workers ... do not call this function in the main process ... It will
    # result in a Segmentation Fault." The mechanism is a module-global
    # VideoDecoderCache. With the default `fork` start method, each worker
    # inherits a *copy of the parent's already-open decoders* — same file
    # offsets, same internal state — and both ends then seek the same
    # descriptors. It surfaces as torchcodec's "Could not push packet to
    # decoder: Invalid data found", which is what killed the first launch of
    # this run at step 1.
    #
    # `spawn` fixes it at the root: a spawned worker starts a fresh
    # interpreter, re-imports lerobot, and builds its own decoder cache from
    # nothing. There is nothing to inherit, so there is nothing to corrupt.
    # Costs a few seconds of startup once, since persistent_workers keeps them.
    multiprocessing_context = None
    if args.num_workers:
        multiprocessing_context = args.mp_context
        print(
            f"dataloader: {args.num_workers} workers via '{args.mp_context}' "
            "(fork corrupts lerobot's shared video-decoder cache)"
        )
    generator = torch.Generator().manual_seed(args.seed + start_step)
    loader = DataLoader(
        ResilientDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        multiprocessing_context=multiprocessing_context,
        generator=generator,
    )

    stopper = Stopper()
    curve: list[dict[str, Any]] = list(resumed_run["train_loss_curve"]) if resumed_run else []
    val_curve: list[dict[str, Any]] = list(resumed_run["val"]["curve"]) if resumed_run else []
    best = min(val_curve, key=lambda point: point["loss"]) if val_curve else None
    peak_smi = (resumed_run["vram"]["peak_nvidia_smi_mib"] or 0) if resumed_run else 0
    saved: list[int] = []

    # Live, crash-proof logs (see CurveCSV). Opened in append mode so a --resume
    # continues the same two files rather than truncating the first half.
    # `kld_loss` is lerobot's own key for the KL term (ACTPolicy.forward), not
    # `kl_loss`; getting it wrong silently writes an empty column.
    train_csv = CurveCSV(run_dir / "train_curve.csv", ["step", "loss", "l1_loss", "kld_loss"])
    val_csv = CurveCSV(run_dir / "val_curve.csv", ["step", "loss", "l1_loss", "frames"])

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
        val_csv.append(record, flush=True)
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
        # Refresh the plot whenever we checkpoint: cheap, and it means the PNG
        # on disk is never more than --save-every steps stale while watching.
        plot_curves(run_dir, curve, val_curve)

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
        # Flush every 50 steps rather than every step: one fsync per gradient
        # step would be a measurable share of a 100k-step run's wall clock.
        train_csv.append({"step": step, **parts}, flush=step % 50 == 0)
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
        "dataset_names": names,
        "dataset_id": manifest.get("dataset_id"),
        "dataset_ids": {name: manifests[name].get("dataset_id") for name in names},
        "mixture": train_data.per_dataset(),
        "lerobot": {
            "path": str(lerobot_dirs[names[0]].parent.relative_to(args.root)),
            "repo_id": repo_id,
            "fps": fps,
            "train_episodes": sum(len(splits[n]["train_episode_indices"]) for n in names),
            "val_episodes": sum(len(splits[n]["val_episode_indices"]) for n in names),
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
                f"{args.chunk} steps = {args.chunk / fps:.2f} s at {fps} Hz; episodes average "
                f"{sum(m['counts']['frames'] for m in manifests.values()) / max(sum(m['counts']['successes'] for m in manifests.values()), 1):.0f} "
                "frames across the mixture, so lerobot's default 100 would span over half a "
                "demonstration and pad the tail of every late chunk"
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
            "ceiling_mib": args.vram_ceiling_mib,
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
    train_csv.close()
    val_csv.close()
    plot_path = plot_curves(run_dir, curve, val_curve)
    write_json(run_dir / "run.json", run)
    write_report(run_dir, run)
    print(f"curves: {run_dir / 'train_curve.csv'}, {run_dir / 'val_curve.csv'}")
    if plot_path is None:
        print("loss_curve.png not written (matplotlib missing; CSVs are the source of truth)")
    else:
        print(f"plot: {plot_path}")

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
        f"{peak_allocated:.0f} MiB allocated (ceiling {args.vram_ceiling_mib})"
    )
    print(f"checkpoints on disk: {checkpoint_steps(run_dir)}")
    print(f"STATUS: {'complete' if step >= args.steps else 'pending'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
