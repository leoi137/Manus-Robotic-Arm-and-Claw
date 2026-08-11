"""Closed-loop evaluation of an ACT policy in Isaac Sim. Runs in `~/isaaclab-env`.

.. code-block:: bash

    # shell A -- the policy, on the CPU, in the other interpreter
    ./.venv-lerobot/bin/python scripts/policy_server.py \
        --ckpt runs/train/<run>/checkpoints/best --warmup

    # shell B -- the simulator, which owns the GPU alone
    ~/isaaclab-env/bin/python scripts/eval_policy.py \
        --ckpt-run <run> --episodes 200 --namespace eval_dev --video-every 20 --headless

The client half of the cross-venv loop. Isaac Sim drives the arm and holds the
GPU; the policy answers over a unix socket from a CPU process
(``scripts/policy_server.py``). Nothing about the policy is imported here — no
torch, no lerobot, no normalization statistics — so the two halves cannot drift
into disagreeing about preprocessing: there is only one implementation of it,
and it lives on the server.

**Held-out placements, asserted.** Draws come from
``randomize.draw_episode(namespace, 10_000_000 + i)``. The plan reserves
``attempt_index >= 10_000_000`` for evaluation, and this script refuses to run
below that bar rather than trusting the caller to remember. Combined with a
distinct namespace string (the seed is a hash of *both*), a placement seen in
training cannot be re-drawn here.

**Temporal ensembling**, Algorithm 2 of the ACT paper. Every control tick sends
one observation and receives a whole ``K``-step chunk, so at tick *t* there are
up to ``K`` overlapping predictions for the action to execute *now*. They are
combined with exponential weights ``wᵢ = exp(-m·i)``, ``w₀`` on the oldest —
the direction the paper specifies, and the one lerobot's own
``ACTTemporalEnsembler`` implements, mirrored here in numpy because the ensemble
belongs on the side of the wire that knows what "now" means. The alternative
(execute the chunk open-loop, re-query every K steps) throws away K-1 of every
K observations and visibly jerks at the chunk boundary.

**Success is the expert's predicate, unchanged.**
:class:`manus.expert.GraspSuccessMonitor` — object centre 5 cm above its spawn
height for 30 consecutive control steps with the jaws closed — is imported, not
reimplemented, so the number this reports is comparable with the Step 8 expert
gate by construction. Region cells match ``scripts/gen_workspace_map.py``'s 3x6
grid for the same reason.

**The interval is the result, not the point estimate.** A Wilson 95% score
interval is computed over the episodes actually run; the plan's gate is on its
lower bound. Wilson rather than the normal approximation because at rates near
0 or 1 — exactly where a good policy lives — the normal interval runs off the
end of [0, 1] and reports a lower bound that cannot be right.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# The src-layout package, plus scripts/ for the wire format. policy_server's
# top-level imports are stdlib only, which is what makes importing it here --
# in an interpreter with no torch and no lerobot -- legal.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import policy_server  # noqa: E402

KIND = "eval"
"""Run kind; names the ``runs/<kind>/`` directory and the run-name prefix."""

EVAL_SEED_BASE = 10_000_000
"""First attempt index of the held-out namespace (plan §Design decisions)."""

MAX_CONTROL_STEPS = 450
"""Per-episode ceiling: 15 s at 30 Hz, more than twice the expert's demonstrations."""

SETTLE_STEPS = 30
WARMUP_RENDERS = 6
"""Reset schedule, identical to ``scripts/gen_dataset.py`` so the first
observation of an eval episode is drawn from the same distribution as the first
observation of a training episode."""

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
"""Wrist frame size on the wire, downscaled from the camera's native 640x480."""

ENSEMBLE_COEFF = 0.1
"""ACT temporal-ensemble coefficient ``m``. Positive weights older chunks more."""

RADIUS_BINS = 3
AZIMUTH_BINS = 6
"""Report cells, matching ``scripts/gen_workspace_map.py`` exactly."""

VIDEO_FPS = 30
"""One frame per control step, so playback is real time."""

MIN_FREE_VRAM_MIB = 6500
"""Pre-flight floor on free VRAM (shared GPU; the plan's ceiling contract)."""

NUDGE_M = 0.02
"""Object rise that separates "the jaws moved it" from "the jaws missed"."""


# --- Statistics ------------------------------------------------------------------


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    The score interval inverts the test ``|p̂ - p| / sqrt(p(1-p)/n) <= z``,
    solving for ``p`` rather than substituting ``p̂`` into the standard error.
    That is what keeps it inside [0, 1] and usable at 0 and 1 successes, where
    the Wald interval collapses to a point and claims certainty it has not
    earned.

    Args:
        successes: Number of successes observed.
        total: Number of trials.
        z: Standard normal quantile; the default is the two-sided 95% value.

    Returns:
        ``(lower, upper)``, clipped to [0, 1]. An empty sample returns
        ``(0.0, 1.0)`` — no trials, no information.
    """
    if total <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= total:
        raise ValueError(f"{successes} successes out of {total} trials")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z
        / denominator
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4 * total * total))
    )
    # p = 0 and p = 1 are *exact* roots of the score equation when p̂ is 0 or 1
    # — substitute and the whole expression vanishes. The closed form gets them
    # to within one ulp, which is enough to print "upper bound 0.9999999999999999"
    # for a flawless run. Pin the algebra rather than the arithmetic.
    lower = 0.0 if successes == 0 else max(0.0, centre - spread)
    upper = 1.0 if successes == total else min(1.0, centre + spread)
    return (lower, upper)


def percentile_ms(samples: list[float]) -> dict[str, Any]:
    """p50/p95/max/mean over per-request seconds, reported in milliseconds."""
    return policy_server.latency_summary(samples)


# --- Temporal ensembling ---------------------------------------------------------


class TemporalEnsembler:
    """Algorithm 2 of the ACT paper, as an online average, in numpy.

    At every tick the policy returns a ``(K, D)`` chunk covering ticks
    ``t .. t+K-1``. The entry for tick ``t+j`` is folded into a running weighted
    average that already holds the predictions made for it at ticks
    ``t-K+1 .. t-1``; the entry for tick ``t`` is then complete and is popped.
    Weights are ``wᵢ = exp(-coeff·i)`` with ``i`` counting from the *oldest*
    prediction, so a positive coefficient trusts the chunk that was planned
    with more context over the one planned just now — the paper's choice, and
    the one lerobot's ``ACTTemporalEnsembler`` implements.

    Kept online (a running mean plus a per-entry count) rather than as a
    history buffer: the arithmetic is identical and the state is ``O(K·D)``
    instead of ``O(K²·D)``.
    """

    def __init__(self, coeff: float, chunk_size: int) -> None:
        import numpy as np

        self.coeff = float(coeff)
        self.chunk_size = int(chunk_size)
        self.weights = np.exp(-self.coeff * np.arange(self.chunk_size, dtype=np.float64))
        self.weights_cumsum = np.cumsum(self.weights)
        self.reset()

    def reset(self) -> None:
        """Forget every prediction; call between episodes."""
        self.actions: Any = None
        self.counts: Any = None

    def update(self, chunk: Any) -> Any:
        """Fold in one ``(K, D)`` chunk and pop the action for the current tick."""
        import numpy as np

        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.shape[0] != self.chunk_size:
            raise ValueError(f"expected a {self.chunk_size}-step chunk, got {chunk.shape[0]}")
        if self.actions is None:
            self.actions = chunk.copy()
            self.counts = np.ones((self.chunk_size, 1), dtype=np.int64)
        else:
            self.actions *= self.weights_cumsum[self.counts - 1]
            self.actions += chunk[:-1] * self.weights[self.counts]
            self.actions /= self.weights_cumsum[self.counts]
            self.counts = np.clip(self.counts + 1, None, self.chunk_size)
            self.actions = np.concatenate([self.actions, chunk[-1:]], axis=0)
            self.counts = np.concatenate([self.counts, np.ones_like(self.counts[-1:])], axis=0)
        action = self.actions[0].copy()
        self.actions = self.actions[1:]
        self.counts = self.counts[1:]
        return action


# --- Region cells ----------------------------------------------------------------


def cell_of(x: float, y: float) -> str:
    """Report cell a placement falls in, e.g. ``"r1_az2"``.

    The same 3x6 grid ``scripts/gen_workspace_map.py`` reports the expert gate
    over, so the two breakdowns can be read side by side.
    """
    from manus.kinematics import GRASP_REGION

    radius, azimuth = GRASP_REGION.polar(x, y)
    azimuth = math.degrees(azimuth)
    low, high = GRASP_REGION.radius
    span = GRASP_REGION.azimuth_max_deg
    radius_bin = min(RADIUS_BINS - 1, max(0, int((radius - low) / (high - low) * RADIUS_BINS)))
    azimuth_bin = min(AZIMUTH_BINS - 1, max(0, int((azimuth + span) / (2 * span) * AZIMUTH_BINS)))
    return f"r{radius_bin}_az{azimuth_bin}"


def cell_label(name: str) -> str:
    """Human-readable bounds of a cell name from :func:`cell_of`."""
    from manus.kinematics import GRASP_REGION

    radius_bin, azimuth_bin = (
        int(part[1:]) if part[0] == "r" else int(part[2:]) for part in name.split("_")
    )
    low, high = GRASP_REGION.radius
    span = GRASP_REGION.azimuth_max_deg
    r0 = low + (high - low) * radius_bin / RADIUS_BINS
    r1 = low + (high - low) * (radius_bin + 1) / RADIUS_BINS
    a0 = -span + 2 * span * azimuth_bin / AZIMUTH_BINS
    a1 = -span + 2 * span * (azimuth_bin + 1) / AZIMUTH_BINS
    return f"r {r0:.3f}-{r1:.3f} m, az {a0:+.0f}..{a1:+.0f} deg"


def classify(success: bool, peak_z: float, spawn_z: float, threshold_z: float) -> str:
    """Name a policy episode's outcome from the object's height trace.

    A policy has no plan to interrogate, so the expert's
    :func:`manus.expert.classify_outcome` taxonomy does not transfer whole;
    these are its height-derived clauses, which are the ones that survive
    without privileged knowledge of intent.
    """
    if success:
        return "success"
    if peak_z >= threshold_z:
        return "slipped"
    if peak_z >= spawn_z + NUDGE_M:
        return "short_lift"
    return "no_lift"


# --- Provenance ------------------------------------------------------------------


def _nvidia_smi(query: str) -> str | None:
    """First row of ``nvidia-smi --query-gpu=<query>``, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
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
    """Free VRAM in MiB, or None when nvidia-smi cannot be reached."""
    value = _nvidia_smi("memory.free")
    return None if value is None else int(value.split()[0])


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


def file_digest(path: Path) -> str | None:
    """SHA-256 of a file, or None when it is not there."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- The client ------------------------------------------------------------------


class PolicyClient:
    """One connection to the CPU policy server, framed per its wire format."""

    def __init__(self, socket_path: Path, connect_timeout: float = 120.0) -> None:
        import socket as socket_module

        deadline = time.time() + connect_timeout
        last: Exception | None = None
        while time.time() < deadline:
            try:
                self.socket = socket_module.socket(
                    socket_module.AF_UNIX, socket_module.SOCK_STREAM
                )
                self.socket.connect(str(socket_path))
                break
            except OSError as error:  # the server may still be loading the checkpoint
                last = error
                self.socket.close()
                time.sleep(1.0)
        else:
            raise SystemExit(
                f"could not connect to {socket_path} within {connect_timeout:.0f} s ({last}); "
                "is scripts/policy_server.py running in .venv-lerobot?"
            )
        self.roundtrip: list[float] = []
        self.server_ms: list[float] = []

    def act(self, joint_pos: Any, jpeg: bytes) -> Any:
        """Send one observation, return the ``(K, D)`` action chunk as a list."""
        started = time.perf_counter()
        self.socket.sendall(policy_server.pack_request(joint_pos, jpeg))
        reply = policy_server.recv_reply(self.socket)
        self.roundtrip.append(time.perf_counter() - started)
        if "server_ms" in reply:
            self.server_ms.append(float(reply["server_ms"]) / 1e3)
        return reply["actions"]

    def close(self) -> None:
        """Hang up, which is what tells the server to print its own latency."""
        try:
            self.socket.close()
        except OSError:
            pass


# --- The runner ------------------------------------------------------------------


def annotate(frame: Any, lines: list[str]) -> Any:
    """Burn a few lines of status text into the top-left of an RGB frame."""
    from PIL import Image, ImageDraw
    import numpy as np

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        position = (6, 6 + 13 * index)
        for offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((position[0] + offset[0], position[1] + offset[1]), line, fill=(0, 0, 0))
        draw.text(position, line, fill=(255, 255, 255))
    return np.asarray(image)


class EvalRunner:
    """Drives one live scene through policy-controlled episodes."""

    def __init__(self, sim: Any, scene: Any, spec: Any, client: PolicyClient) -> None:
        import torch

        from manus import specs

        self.sim = sim
        self.scene = scene
        self.spec = spec
        self.client = client
        self.robot = scene["robot"]
        self.object = scene["object"]
        self.camera = scene["wrist_cam"]
        self.dt = sim.get_physics_dt()
        self.device = self.robot.data.joint_pos.torch.device
        assert self.robot.joint_names == list(specs.JOINT_NAMES), (
            f"joint order mismatch: {self.robot.joint_names} != {list(specs.JOINT_NAMES)}"
        )
        self.gripper_column = specs.JOINT_NAMES.index("gripper")
        self.home = torch.tensor(
            [[specs.HOME_POSE[name] for name in specs.JOINT_NAMES]],
            dtype=torch.float32,
            device=self.device,
        )
        self.lower = torch.tensor(
            [[specs.JOINT_LIMITS[name][0] for name in specs.JOINT_NAMES]],
            dtype=torch.float32,
            device=self.device,
        )
        self.upper = torch.tensor(
            [[specs.JOINT_LIMITS[name][1] for name in specs.JOINT_NAMES]],
            dtype=torch.float32,
            device=self.device,
        )
        self.ensembler: TemporalEnsembler | None = None
        self.native_hw: tuple[int, int] | None = None

    # -- plumbing ------------------------------------------------------------------

    def advance(self, render: bool) -> None:
        """One physics step, refreshing only the buffers this script reads."""
        self.sim.step(render=render)
        self.robot.update(self.dt)
        self.object.update(self.dt)

    def measured(self) -> Any:
        """Six measured joint positions (radians), in ``specs.JOINT_NAMES`` order."""
        return self.robot.data.joint_pos.torch[0].detach().cpu().numpy().astype(float)

    def object_z(self) -> float:
        """Object body-origin height above the ground plane, metres."""
        return float(self.object.data.root_link_pos_w.torch[0][2] - self.scene.env_origins[0][2])

    def capture(self) -> Any:
        """The wrist frame for the current state, at the recorded resolution.

        The same downscale the dataset was built with — PIL LANCZOS from the
        camera's native 640x480 to 320x240 — because a policy trained on
        LANCZOS-resampled pixels must be shown LANCZOS-resampled pixels.
        """
        import numpy as np
        import torch
        from PIL import Image

        self.camera.update(4 * self.dt)
        rgb = self.camera.data.output["rgb"].torch[0, ..., :3].to(torch.uint8).cpu().numpy()
        self.native_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        if rgb.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
            rgb = np.asarray(
                Image.fromarray(rgb).resize((FRAME_WIDTH, FRAME_HEIGHT), Image.LANCZOS),
                dtype=np.uint8,
            )
        return np.ascontiguousarray(rgb)

    def reset_episode(self, draw: Any) -> None:
        """Home the arm, stamp the draw onto the scene, and let it settle."""
        import torch

        from manus.task_scene import apply_randomization

        self.robot.write_joint_state_to_sim_index(
            position=self.home, velocity=torch.zeros_like(self.home), full_data=True
        )
        self.robot.set_joint_position_target_index(target=self.home)
        apply_randomization(self.scene, draw, self.spec)
        self.scene.reset()
        self.scene.write_data_to_sim()
        for step in range(SETTLE_STEPS):
            self.advance(render=step >= SETTLE_STEPS - WARMUP_RENDERS)

    # -- one episode ---------------------------------------------------------------

    def run(self, episode_index: int, seed_index: int, draw: Any, record: bool) -> dict[str, Any]:
        """Run one policy-controlled episode. Returns its outcome dictionary.

        The temporal contract is the recorder's, unchanged: the frame and the
        measured joints are read *before* the step whose resulting state they
        precede, the action derived from them is written, and only then does
        physics advance. That is the loop the demonstrations were recorded
        under, so the policy sees the phase it was trained in.
        """
        import numpy as np
        import torch

        from manus import recorder
        from manus.expert import GraspSuccessMonitor

        self.reset_episode(draw)
        measured = self.measured()
        frame = self.capture()
        home = np.array([float(value) for value in self.home[0].cpu().numpy()])
        # Every episode starts with an empty ensemble: carrying chunks across a
        # reset would drive the first ticks of a new placement with actions
        # planned for the previous one.
        self.ensembler = None

        monitor = GraspSuccessMonitor(self.spec.spawn_z)
        rest_z = self.object_z()
        jpegs: list[bytes] = []
        path_length = np.zeros(6)
        peak_excursion = np.zeros(6)
        previous = measured.copy()
        chunk_size: int | None = None
        stop_reason = "max_steps"
        started = time.perf_counter()

        for step in range(MAX_CONTROL_STEPS):
            jpeg = recorder.encode_jpeg(frame)
            if record:
                jpegs.append(jpeg)
            chunk = self.client.act(measured, jpeg)
            if self.ensembler is None:
                # K is whatever the server's checkpoint was trained with; the
                # client never has to be told, and cannot be told wrong.
                chunk_size = len(chunk)
                self.ensembler = TemporalEnsembler(ENSEMBLE_COEFF, chunk_size)
            action = self.ensembler.update(chunk)

            target = torch.from_numpy(
                np.asarray(action, dtype=np.float32).reshape(1, -1)
            ).to(self.device)
            # The policy is free to emit anything; the servo is not. Clamping
            # here rather than trusting the network keeps an out-of-range
            # target from being silently absorbed by PhysX at a joint stop.
            target = torch.clamp(target, self.lower, self.upper)
            self.robot.set_joint_position_target_index(target=target)
            self.scene.write_data_to_sim()
            for substep in range(recorder.DECIMATION):
                self.advance(render=substep == recorder.DECIMATION - 1)

            measured = self.measured()
            frame = self.capture()
            path_length += np.abs(measured - previous)
            peak_excursion = np.maximum(peak_excursion, np.abs(measured - home))
            previous = measured.copy()
            monitor.update(self.object_z(), measured[self.gripper_column])
            if monitor.success:
                stop_reason = "success"
                break

        outcome = classify(monitor.success, monitor.peak_z, self.spec.spawn_z, monitor.threshold_z)
        return {
            "episode": episode_index,
            "seed_index": seed_index,
            "draw": draw.to_dict(),
            "cell": cell_of(draw.object_x, draw.object_y),
            "success": bool(monitor.success),
            "outcome": outcome,
            "stop_reason": stop_reason,
            "steps": step + 1,
            "rest_z": rest_z,
            "monitor": monitor.to_dict(),
            "chunk_size": chunk_size,
            "motion": {
                "path_length_rad": [float(value) for value in path_length],
                "peak_excursion_rad": [float(value) for value in peak_excursion],
                "total_path_rad": float(path_length.sum()),
                "max_excursion_rad": float(peak_excursion.max()),
            },
            "wall_clock_s": time.perf_counter() - started,
            "frames": jpegs,
        }


def write_video(path: Path, jpegs: list[bytes], caption: list[str]) -> Path:
    """Write the frames the policy actually saw to an mp4.

    Deliberately the wire JPEGs and not a re-render: the video is then a record
    of the policy's own input, so a failure that turns out to be "the camera
    saw nothing useful" is visible in the artefact rather than only in the
    numbers.
    """
    import imageio.v3 as iio
    import numpy as np

    from manus import recorder

    frames = [recorder.decode_jpeg(blob) for blob in jpegs]
    frames = [annotate(frame, caption + [f"t={index:3d}"]) for index, frame in enumerate(frames)]
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), fps=VIDEO_FPS)
    return path


# --- Reporting -------------------------------------------------------------------


def summarise(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Success rate, Wilson interval, outcome taxonomy and the per-cell table."""
    from collections import Counter

    total = len(episodes)
    successes = sum(1 for episode in episodes if episode["success"])
    low, high = wilson_interval(successes, total)
    cells: dict[str, dict[str, int]] = {}
    for episode in episodes:
        cell = cells.setdefault(episode["cell"], {"n": 0, "successes": 0})
        cell["n"] += 1
        cell["successes"] += int(episode["success"])
    for cell in cells.values():
        cell["rate"] = cell["successes"] / cell["n"]
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "wilson95": {"lower": low, "upper": high, "z": 1.959963984540054},
        "outcomes": dict(Counter(episode["outcome"] for episode in episodes)),
        "cells": dict(sorted(cells.items())),
    }


def motion_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Did the actions actually move the arm? The mini-eval's mechanical gate."""
    paths = [episode["motion"]["total_path_rad"] for episode in episodes]
    excursions = [episode["motion"]["max_excursion_rad"] for episode in episodes]
    return {
        "episodes": len(episodes),
        "total_path_rad": {
            "min": min(paths) if paths else None,
            "mean": sum(paths) / len(paths) if paths else None,
            "max": max(paths) if paths else None,
        },
        "max_excursion_rad": {
            "min": min(excursions) if excursions else None,
            "mean": sum(excursions) / len(excursions) if excursions else None,
            "max": max(excursions) if excursions else None,
        },
        "episodes_with_no_motion": sum(1 for value in excursions if value < 1e-6),
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON atomically, indented so a diff is readable."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _num(value: Any, spec: str = ".1f") -> str:
    """Format a number, or an em dash when the run never produced one."""
    return format(value, spec) if isinstance(value, (int, float)) else "—"


def write_report(run_dir: Path, run: dict[str, Any]) -> Path:
    """The human-readable half of the run's provenance."""
    result = run["result_detail"]
    wilson = result["wilson95"]
    latency = run["latency"]
    motion = run["motion"]
    lines = [
        f"# {run['run_name']}",
        "",
        f"ACT policy `{run['checkpoint']['run']}` (step {run['checkpoint']['step']}), "
        f"closed loop in Isaac Sim over {result['episodes']} held-out placements from "
        f"namespace `{run['seeds']['namespace']}` "
        f"({run['seeds']['first']}..{run['seeds']['last']}).",
        "",
        f"**{result['successes']}/{result['episodes']} = {result['success_rate']:.1%}** — "
        f"Wilson 95% interval [{wilson['lower']:.3f}, {wilson['upper']:.3f}], "
        f"lower bound **{wilson['lower']:.3f}**.",
        "",
        run["gate_note"],
        "",
        "## Outcomes",
        "",
        "| outcome | n |",
        "| --- | --- |",
    ]
    for name, count in sorted(result["outcomes"].items(), key=lambda item: -item[1]):
        lines.append(f"| {name} | {count} |")

    lines += [
        "",
        "## By region cell",
        "",
        "| cell | bounds | n | successes | rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, cell in result["cells"].items():
        lines.append(
            f"| `{name}` | {cell_label(name)} | {cell['n']} | {cell['successes']} | "
            f"{cell['rate']:.0%} |"
        )

    lines += [
        "",
        "## Loop mechanics",
        "",
        "| | |",
        "| --- | --- |",
        f"| control rate | {run['loop']['control_hz']} Hz "
        f"(physics {run['loop']['physics_dt']:.6f} s, decimation {run['loop']['decimation']}) |",
        f"| action chunk K | {run['loop']['chunk_size']} |",
        f"| temporal ensemble m | {run['loop']['ensemble_coeff']} "
        f"(w_i = exp(-m i), w_0 oldest) |",
        f"| episode ceiling | {run['loop']['max_control_steps']} control steps |",
        f"| requests | {latency['roundtrip']['requests']} |",
        f"| round-trip latency | p50 {_num(latency['roundtrip']['p50_ms'])} ms, "
        f"p95 {_num(latency['roundtrip']['p95_ms'])} ms, "
        f"max {_num(latency['roundtrip']['max_ms'])} ms |",
        f"| server inference latency | p50 {_num(latency['server']['p50_ms'])} ms, "
        f"p95 {_num(latency['server']['p95_ms'])} ms |",
        f"| joint path per episode | mean {_num(motion['total_path_rad']['mean'], '.2f')} rad, "
        f"min {_num(motion['total_path_rad']['min'], '.2f')} rad |",
        f"| peak excursion from home | mean "
        f"{_num(motion['max_excursion_rad']['mean'], '.2f')} rad, "
        f"min {_num(motion['max_excursion_rad']['min'], '.2f')} rad |",
        f"| episodes with no motion | {motion['episodes_with_no_motion']} |",
        "",
    ]
    if run["videos"]:
        lines += ["## Videos", ""]
        for video in run["videos"]:
            lines.append(
                f"- `{video['path']}` — episode {video['episode']}, {video['outcome']}, "
                f"{video['frames']} frames"
            )
        lines.append("")
    lines.append("Provenance: `run.json` here. Generated by `scripts/eval_policy.py`.")
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- Entry point -----------------------------------------------------------------


def build_parser(with_app_launcher: bool = True) -> argparse.ArgumentParser:
    """The CLI. `with_app_launcher` is False in tests, which have no simulator."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Deliberately not `required`: AppLauncher.add_app_launcher_args runs
    # parse_known_args() on the real argv to check for name collisions, and a
    # missing required argument makes that call abort before the launcher's own
    # flags are even registered -- which turns `--help` into an error message
    # about --ckpt-run. Validated in main() instead, where the message can say
    # something useful.
    parser.add_argument("--ckpt-run", default=None, help="training run name under runs/train/")
    parser.add_argument(
        "--ckpt", default="best", help="checkpoint within the run: best, last, or a step directory"
    )
    parser.add_argument("--episodes", type=int, default=200, help="held-out placements to run")
    parser.add_argument("--namespace", default="eval_dev", help="held-out seed namespace")
    parser.add_argument(
        "--seed-base", type=int, default=EVAL_SEED_BASE, help="first attempt index (>= 10 000 000)"
    )
    parser.add_argument("--video-every", type=int, default=20, help="record every Nth episode")
    parser.add_argument("--run-name", default="auto", help="run directory name, or 'auto'")
    parser.add_argument(
        "--socket", type=Path, default=policy_server.DEFAULT_SOCKET, help="policy server socket"
    )
    parser.add_argument(
        "--connect-timeout", type=float, default=180.0, help="seconds to wait for the server"
    )
    parser.add_argument("--object", default="cube_3cm", help="catalogue key of the object")
    parser.add_argument(
        "--gate-lower-bound", type=float, default=None, help="fail below this LB95"
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    if with_app_launcher:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def resolve_checkpoint(root: Path, ckpt_run: str, which: str) -> tuple[Path, int]:
    """The checkpoint directory to serve, and the step it was saved at."""
    run_dir = root / "runs" / "train" / ckpt_run
    if not run_dir.is_dir():
        raise SystemExit(f"no such training run: {run_dir}")
    directory = (run_dir / "checkpoints" / which).resolve()
    if not directory.is_dir():
        raise SystemExit(f"{run_dir}/checkpoints/{which} does not resolve to a checkpoint")
    return directory, int(directory.name)


def main(args: Any) -> int:  # noqa: PLR0915 - one linear run script
    """Run the evaluation. Returns the process exit code."""
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    from manus import expert as expert_mod
    from manus import recorder
    from manus.objects import OBJECTS
    from manus.randomize import draw_episode, stable_hash64
    from manus.task_scene import GraspSceneCfg

    if not args.ckpt_run:
        raise SystemExit(
            "--ckpt-run is required: the name of a directory under runs/train/, e.g.\n"
            "  --ckpt-run train__grasp_cube_dev__20260811-0615__d0e26f30"
        )
    if args.seed_base < EVAL_SEED_BASE:
        raise SystemExit(
            f"--seed-base {args.seed_base} is inside the training namespace; the plan reserves "
            f"attempt_index >= {EVAL_SEED_BASE} for evaluation"
        )

    checkpoint, step = resolve_checkpoint(args.root, args.ckpt_run, args.ckpt)
    train_run = json.loads(
        (args.root / "runs" / "train" / args.ckpt_run / "run.json").read_text(encoding="utf-8")
    )
    dataset_name = train_run["dataset_name"]
    git = git_provenance(args.root)
    name = (
        stamped_run_name(KIND, dataset_name, git["sha"])
        if args.run_name == "auto"
        else args.run_name
    )
    run_dir = args.root / "runs" / KIND / name
    run_dir.mkdir(parents=True, exist_ok=True)

    spec = OBJECTS[args.object]
    client = PolicyClient(args.socket, args.connect_timeout)
    print(f"connected to the policy server on {args.socket}")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=recorder.PHYSICS_DT, device=args.device)
    )
    scene = InteractiveScene(GraspSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    runner = EvalRunner(sim, scene, spec, client)

    episodes: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    have: set[str] = set()
    started = time.time()
    for index in range(args.episodes):
        seed_index = args.seed_base + index
        draw = draw_episode(args.namespace, seed_index)
        # Record on the schedule, plus whichever of success/failure we have yet
        # to see: a report with no failure video is a report you cannot debug.
        record = (args.video_every and index % args.video_every == 0) or len(have) < 2
        result = runner.run(index, seed_index, draw, record=record)
        frames = result.pop("frames")
        result["seed"] = stable_hash64(args.namespace, seed_index)
        episodes.append(result)

        label = "success" if result["success"] else "failure"
        if frames and (label not in have or (args.video_every and index % args.video_every == 0)):
            path = write_video(
                run_dir / "videos" / f"ep{index:04d}_{result['outcome']}.mp4",
                frames,
                [f"ep{index} {result['outcome']}", f"cell {result['cell']}"],
            )
            videos.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "episode": index,
                    "outcome": result["outcome"],
                    "success": result["success"],
                    "frames": len(frames),
                }
            )
            have.add(label)

        elapsed = time.time() - started
        print(
            f"  [{index:4d}] {result['outcome']:<11} {result['steps']:4d} steps  "
            f"cell {result['cell']:<9} peak {result['monitor']['peak_z'] * 1e3:6.1f} mm  "
            f"path {result['motion']['total_path_rad']:5.2f} rad  "
            f"({elapsed / (index + 1):4.1f} s/ep)",
            flush=True,
        )

    wall_clock = time.time() - started
    client.close()

    result_detail = summarise(episodes)
    motion = motion_summary(episodes)
    lower = result_detail["wilson95"]["lower"]
    gated = args.gate_lower_bound is not None
    gate_note = (
        (
            f"Gate: Wilson LB95 >= {args.gate_lower_bound:.2f} — "
            f"**{'PASS' if lower >= args.gate_lower_bound else 'FAIL'}**"
        )
        if gated
        else (
            "No success gate on this run: the mini-eval checks the loop mechanically "
            "(actions move the arm, metrics and videos are written), and the smoke policy "
            "is weak by construction."
        )
    )

    run = {
        "kind": KIND,
        "run_name": name,
        "dataset_name": dataset_name,
        "dataset_id": train_run.get("dataset_id"),
        "checkpoint": {
            "run": args.ckpt_run,
            "which": args.ckpt,
            "path": str(checkpoint.relative_to(args.root)),
            "step": step,
            "sha256": file_digest(checkpoint / "pretrained_model" / "model.safetensors"),
            "train_steps": train_run["progress"]["steps_done"],
            "train_val_best": train_run["val"].get("best"),
        },
        "seeds": {
            "namespace": args.namespace,
            "base": args.seed_base,
            "first": args.seed_base,
            "last": args.seed_base + args.episodes - 1,
            "held_out_floor": EVAL_SEED_BASE,
            "overlap_with_training": False,
            "note": (
                "draw_episode hashes (namespace, attempt_index); the namespace differs from the "
                "training dataset's and every index is at or above the held-out floor"
            ),
        },
        "loop": {
            "control_hz": recorder.CONTROL_HZ,
            "physics_dt": recorder.PHYSICS_DT,
            "decimation": recorder.DECIMATION,
            "max_control_steps": MAX_CONTROL_STEPS,
            "settle_steps": SETTLE_STEPS,
            "warmup_renders": WARMUP_RENDERS,
            "chunk_size": episodes[0]["chunk_size"] if episodes else None,
            "ensemble_coeff": ENSEMBLE_COEFF,
            "ensemble": "ACT Algorithm 2, w_i = exp(-m i), w_0 on the oldest chunk",
            "frame_hw": [FRAME_HEIGHT, FRAME_WIDTH],
            "native_hw": list(runner.native_hw) if runner.native_hw else None,
            "resample": "PIL LANCZOS",
            "jpeg_quality": recorder.JPEG_QUALITY,
        },
        "success_predicate": {
            "source": "manus.expert.GraspSuccessMonitor (the expert's, unchanged)",
            "lift_m": expert_mod.SUCCESS_LIFT_M,
            "sustain_steps": expert_mod.SUCCESS_SUSTAIN_STEPS,
            "gripper_held_max_rad": expert_mod.GRIPPER_HELD_MAX_RAD,
        },
        "result_detail": result_detail,
        "motion": motion,
        "latency": {
            "roundtrip": percentile_ms(client.roundtrip),
            "server": percentile_ms(client.server_ms),
            "note": "round-trip includes JPEG encode, socket IPC and the server's inference",
        },
        "episodes": episodes,
        "videos": videos,
        "gate": {
            "lower_bound_required": args.gate_lower_bound,
            "lower_bound_observed": lower,
            "passed": None if not gated else bool(lower >= args.gate_lower_bound),
        },
        "gate_note": gate_note,
        "env": {
            "python": sys.version.split()[0],
            "gpu": _nvidia_smi("name"),
            "driver": _nvidia_smi("driver_version"),
            "device": args.device,
            "free_vram_mib_at_start": args.preflight_free_mib,
            "object": spec.name,
        },
        "git": git,
        "wall_clock_s": wall_clock,
        "seconds_per_episode": wall_clock / max(len(episodes), 1),
        "result": (
            f"{result_detail['successes']}/{result_detail['episodes']} = "
            f"{result_detail['success_rate']:.0%}, LB95 {lower:.3f}"
        ),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(run_dir / "run.json", run)
    write_report(run_dir, run)

    print(
        f"\n{name}: {result_detail['successes']}/{result_detail['episodes']} = "
        f"{result_detail['success_rate']:.1%}, Wilson 95% "
        f"[{lower:.3f}, {result_detail['wilson95']['upper']:.3f}]"
    )
    print(f"outcomes: {result_detail['outcomes']}")
    print(
        f"motion: peak excursion mean {motion['max_excursion_rad']['mean']:.3f} rad, "
        f"min {motion['max_excursion_rad']['min']:.3f} rad, "
        f"{motion['episodes_with_no_motion']} episodes with none"
    )
    latency = run["latency"]["roundtrip"]
    print(
        f"latency: {latency['requests']} requests, p50 {latency['p50_ms']:.1f} ms, "
        f"p95 {latency['p95_ms']:.1f} ms"
    )
    print(f"videos: {len(videos)} -> {run_dir / 'videos'}")
    print(f"wrote {run_dir / 'run.json'} and report.md")
    if motion["episodes_with_no_motion"] or (motion["max_excursion_rad"]["max"] or 0) < 1e-3:
        print("FAIL: the policy did not move the arm", file=sys.stderr)
        return 1
    if gated and lower < args.gate_lower_bound:
        print(f"FAIL: Wilson LB95 {lower:.3f} < {args.gate_lower_bound:.2f}", file=sys.stderr)
        return 1
    print("STATUS: complete")
    return 0


if __name__ == "__main__":
    args_cli = build_parser().parse_args()
    # The scene always carries the wrist camera, and Isaac Lab's Camera refuses
    # to initialise unless the app renders sensors.
    args_cli.enable_cameras = True

    free = free_vram_mib()
    if free is not None and free < MIN_FREE_VRAM_MIB:
        raise SystemExit(
            f"pre-flight: only {free} MiB of VRAM free, need >= {MIN_FREE_VRAM_MIB} MiB "
            "(shared GPU: the policy server must be on the CPU)"
        )
    args_cli.preflight_free_mib = free
    print(f"pre-flight: {free} MiB free VRAM; the policy server is a separate CPU process")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        code = main(args_cli)
    finally:
        simulation_app.close()
    sys.exit(code)
