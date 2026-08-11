"""Generate a raw grasping dataset: one Isaac boot, one chunk of attempts.

The data factory. Each attempt draws its own randomization
(:func:`manus.randomize.draw_episode`), stamps it onto the scene, runs
:class:`~manus.expert.ScriptedGraspExpert` to a verdict, and — when the grasp
succeeds — writes one ``episode_<attempt_index>.npz`` through
:class:`manus.recorder.EpisodeRecorder`. Every attempt, successful or not,
appends one line to ``attempts.jsonl``; that ledger is the source of truth and
``manifest.json`` is rebuilt from it at the end of every boot.

.. code-block:: bash

    # one chunk (default 50 attempts), then exit; repeat until it says complete
    ~/isaaclab-env/bin/python scripts/gen_dataset.py \
        --dataset grasp_cube_dev --target-successes 50 --chunk 25 --headless

**Single boot by design.** Isaac Sim is not restartable in-process and this
machine allows one GPU process at a time, so the script runs at most ``--chunk``
attempts and exits; the caller loops. Resumption needs no state beyond the
ledger: the next attempt index is one past the highest recorded, and
``--target-successes`` counts the successes already on disk. The last line of a
run is always ``STATUS: complete`` (target met), ``STATUS: pending`` (more
attempts to run) or ``STATUS: exhausted`` (``--max-attempts`` spent first, exit
code 1), which is what a driver loop should branch on.

Recording follows the recorder's temporal contract exactly — *action[t] is the
joint target written before the step whose resulting state is joint_pos[t+1]* —
so one recorded row is ``(frame, joint_pos, action)`` all observed *before* the
physics step, with the resulting state landing in the next row:

.. code-block:: text

    read joint_pos[t] and frame[t]  ->  action[t] = expert.step(joint_pos[t])
    write action[t]  ->  4 x sim.step()  ->  joint_pos[t+1], frame[t+1]

**Render decimation** is the throughput lever: only the last of each control
step's four physics steps renders (``sim.step(render=...)``), so the wrist
camera runs at the 30 Hz capture rate rather than the 120 Hz physics rate. Both
per-step costs are timed and the measured factor is recorded in the manifest.

Frames are stored at **320x240**, downscaled from the camera's native 640x480
with PIL. That is the resolution the whole downstream chain is specified at
(``scripts/convert_dataset.py``, and ``scripts/verify_dataset.py`` compares a
converted frame against the raw JPEG at the same index *shape for shape*), and
at 92% JPEG it costs ~10 kB a frame instead of ~40.

Episodes are cut the moment the success predicate latches, which is what
:class:`~manus.expert.GraspSuccessMonitor` documents the driver as doing: the
30 held steps that satisfy it are the end of the demonstration, and the frames
after it are a motionless arm that teaches a policy to stop moving.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the src-layout package importable without installing it. The sim-free
# half (recorder, randomize, objects) is imported straight away so the ledger
# can be read before deciding whether to boot Isaac Sim at all.
sys.path.insert(0, str(REPO_ROOT / "src"))

from isaaclab.app import AppLauncher  # noqa: E402 - importing does not start the app

from manus import recorder  # noqa: E402
from manus.randomize import draw_episode, stable_hash64  # noqa: E402

SETTLE_STEPS = 30
"""Physics steps held at home after a reset, so the object comes to rest.

Same as the Step 8 expert gate, and recorded in every episode's meta because
``scripts/replay_check.py`` has to reproduce the initial state exactly.
"""

WARMUP_RENDERS = 6
"""Rendered steps at the end of the settle, before the first frame is captured.

The RTX renderer accumulates temporally: the first frame after a scene edit is
noisier and darker than the ones that follow, and frame 0 of every episode
would otherwise be the odd one out.
"""

MAX_CONTROL_STEPS = 1200
"""Hard ceiling per attempt; the FSM's own per-state budgets stop well short."""

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
"""Recorded frame size, downscaled from the camera's native 640x480."""

FAILURE_KEEP_TOTAL = 50
"""Cap on retained failure episodes per dataset (the plan's number)."""

FAILURE_KEEP_PER_MODE = 10
"""Cap per failure mode, so the retained sample spans modes rather than the
first fifty of whichever one happens to be commonest."""

FAILURES_DIR = "failures"
"""Subdirectory retained failures are written to, relative to the dataset."""

MIN_FREE_VRAM_MIB = 6500
"""Pre-flight floor on free VRAM (shared GPU; the plan's ceiling contract)."""

CONTROL_DT = recorder.DECIMATION * recorder.PHYSICS_DT
"""Simulated seconds per control step: 1/30 s."""

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--dataset", required=True, help="dataset name, e.g. grasp_cube_dev")
parser.add_argument(
    "--target-successes", type=int, required=True, help="successes the dataset should hold"
)
parser.add_argument(
    "--max-attempts",
    type=int,
    default=None,
    help="attempt budget before giving up (default 3x --target-successes)",
)
parser.add_argument("--chunk", type=int, default=50, help="attempts to run in this boot")
parser.add_argument("--object", default="cube_3cm", help="catalogue key of the object to grasp")
parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root holding datasets/")
parser.add_argument(
    "--manifest-only",
    action="store_true",
    help="rebuild manifest.json from the episodes on disk and exit (no GPU)",
)
parser.add_argument(
    "--probe-render",
    type=int,
    default=0,
    help="after the chunk, time N control steps rendered at 30 Hz and at 120 Hz",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The scene always carries the wrist camera, and Isaac Lab's Camera refuses to
# initialise unless the app renders sensors.
args_cli.enable_cameras = True


# --- Provenance ----------------------------------------------------------------


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
    if value is None:
        return None
    return int(value.split()[0])


def _scalars(mapping: Any) -> dict[str, Any]:
    """Keep the JSON-safe entries of a config dictionary: scalars and flat tuples."""
    if not isinstance(mapping, Mapping):
        return {}
    keep = (bool, int, float, str)
    return {
        key: (list(value) if isinstance(value, (tuple, list)) else value)
        for key, value in mapping.items()
        if value is None
        or isinstance(value, keep)
        or (
            isinstance(value, (tuple, list))
            and all(isinstance(item, keep) for item in value)
        )
    }


def solver_block(sim: Any, spec: Any) -> dict[str, Any]:
    """The physics settings the episodes were produced under.

    Two layers, because they live in different places in this stack: the
    scene-wide ``SimulationCfg`` (``physics`` is None here, meaning PhysX engine
    defaults, which is itself worth recording) and the per-object solver
    iteration counts, which :mod:`manus.objects` raises above the default so a
    squeezed cube does not jitter out of the jaws.
    """
    simulation = _scalars(sim.cfg.to_dict()) if hasattr(sim.cfg, "to_dict") else {}
    physics = getattr(getattr(sim, "cfg", None), "physics", None)
    rigid = getattr(spec.make_spawn_cfg(), "rigid_props", None)
    return {
        "simulation": simulation,
        "physics": _scalars(physics.to_dict()) if hasattr(physics, "to_dict") else None,
        "object_rigid_body": _scalars(rigid.to_dict()) if hasattr(rigid, "to_dict") else None,
    }


def env_block(sim: Any, spec: Any, config: Any) -> dict[str, Any]:
    """Everything about *this* process that the data depends on.

    Read out of the running interpreter rather than declared: versions, the
    driver and card the pixels came off, the physics settings, and the expert
    constants the demonstrations embody. ``objects`` is the key
    ``scripts/gen_catalog_md.py`` renders.
    """
    import numpy as np
    import torch

    from manus import expert as expert_mod
    from manus import kinematics, specs
    from manus.scene import WRIST_CAM_HEIGHT, WRIST_CAM_WIDTH

    versions: dict[str, Any] = {}
    try:
        import isaaclab

        versions["isaaclab"] = getattr(isaaclab, "__version__", None)
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        versions["isaaclab"] = None
    try:
        from isaacsim.core.version import get_version

        version = get_version()
        versions["isaacsim"] = version[0] if isinstance(version, (tuple, list)) else str(version)
    except Exception:  # noqa: BLE001
        versions["isaacsim"] = None

    return {
        "objects": [spec.name],
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "isaacsim": versions["isaacsim"],
        "isaaclab": versions["isaaclab"],
        "gpu": _nvidia_smi("name"),
        "driver": _nvidia_smi("driver_version"),
        "device": str(getattr(getattr(sim, "cfg", None), "device", None)),
        "physics_dt": recorder.PHYSICS_DT,
        "control_hz": recorder.CONTROL_HZ,
        "decimation": recorder.DECIMATION,
        "renders_per_control_step": 1,
        "solver": solver_block(sim, spec),
        "actuator": {
            "kp": specs.STS3215_KP,
            "damping": specs.STS3215_DAMPING,
            "effort_limit": specs.STS3215_EFFORT_LIMIT,
            "velocity_limit": specs.SERVO_VELOCITY_LIMIT,
        },
        "camera": {
            "native_hw": [WRIST_CAM_HEIGHT, WRIST_CAM_WIDTH],
            "recorded_hw": [FRAME_HEIGHT, FRAME_WIDTH],
            "resample": "PIL LANCZOS",
            "jpeg_quality": recorder.JPEG_QUALITY,
        },
        "expert": {
            "hover_height": config.hover_height,
            "converge_tol": config.converge_tol,
            "close_ramp": config.close_ramp,
            "close_target_rad": spec.close_target_rad,
            "hold_steps": config.hold_steps,
            "lift_rise": config.lift_rise,
            "droop_gain": config.droop_gain,
            "tcp_to_pad_centre": kinematics.TCP_TO_PAD_CENTRE,
            "jaw_clearance": expert_mod.JAW_CLEARANCE,
        },
        "success_predicate": {
            "lift_m": expert_mod.SUCCESS_LIFT_M,
            "sustain_steps": expert_mod.SUCCESS_SUSTAIN_STEPS,
            "gripper_held_max_rad": expert_mod.GRIPPER_HELD_MAX_RAD,
            "statement": (
                "object centre >= spawn_z + 0.05 m for 30 consecutive control steps "
                "with the jaws at or below 1.0 rad"
            ),
        },
    }


# --- Droop -----------------------------------------------------------------------


def droop_stats(dataset_dir: Path) -> dict[str, Any]:
    """Commanded-versus-measured joint error over every episode on disk.

    ``action[t] - joint_pos[t+1]``: the target that was written, minus the state
    it actually produced. Under the vendored PD gains with no gravity
    feed-forward that difference *is* the droop, and it is the quantity a policy
    trained on these actions inherits — so it belongs in the provenance next to
    the data rather than in a log nobody reads.

    A pure function of the episodes, like the manifest itself, so a resumed run
    reports the whole dataset and not just its last chunk.
    """
    import numpy as np

    from manus import specs

    per_joint_abs: list[np.ndarray] = []
    per_joint_signed: list[np.ndarray] = []
    peaks: list[np.ndarray] = []
    steps = 0
    for path in recorder.episode_paths(dataset_dir):
        episode = recorder.load_episode(path)
        if len(episode.actions) < 2:
            continue
        error = episode.actions[:-1].astype(np.float64) - episode.joint_pos[1:].astype(np.float64)
        per_joint_abs.append(np.abs(error).sum(axis=0))
        per_joint_signed.append(error.sum(axis=0))
        peaks.append(np.abs(error).max(axis=0))
        steps += error.shape[0]
    if not steps:
        return {"episodes": 0, "steps": 0}

    mean_abs = np.sum(per_joint_abs, axis=0) / steps
    mean_signed = np.sum(per_joint_signed, axis=0) / steps
    peak = np.max(peaks, axis=0)
    return {
        "definition": "action[t] - joint_pos[t+1], radians (commanded minus measured)",
        "episodes": len(per_joint_abs),
        "steps": steps,
        "joints": list(specs.JOINT_NAMES),
        "mean_abs_rad": [float(value) for value in mean_abs],
        "mean_signed_rad": [float(value) for value in mean_signed],
        "max_abs_rad": [float(value) for value in peak],
        "worst_joint": specs.JOINT_NAMES[int(np.argmax(mean_abs))],
    }


def episode_droop(actions: Any, joint_pos: Any) -> dict[str, Any]:
    """Per-episode droop summary, recorded into that episode's meta."""
    import numpy as np

    if len(actions) < 2:
        return {"steps": 0}
    error = np.asarray(actions[:-1], dtype=np.float64) - np.asarray(joint_pos[1:], dtype=np.float64)
    return {
        "steps": int(error.shape[0]),
        "mean_abs_rad": [float(value) for value in np.abs(error).mean(axis=0)],
        "max_abs_rad": [float(value) for value in np.abs(error).max(axis=0)],
    }


# --- Ledger bookkeeping ----------------------------------------------------------


def next_attempt_index(attempts: list[dict[str, Any]]) -> int:
    """One past the highest attempt index in the ledger (0 for a new dataset).

    Deliberately not ``len(attempts)``: a ledger with a gap (a chunk killed
    mid-write, an attempt re-run by hand) must never re-issue an index, because
    the index names the episode file and seeds the draw.
    """
    return max((int(record["attempt_index"]) for record in attempts), default=-1) + 1


def retained_failures(attempts: list[dict[str, Any]]) -> Counter:
    """Count of kept failure episodes per outcome, from the ledger."""
    return Counter(
        record["outcome"]
        for record in attempts
        if record["outcome"] != recorder.SUCCESS and record.get("episode_file")
    )


def should_keep_failure(kept: Counter, outcome: str) -> bool:
    """Whether this failure earns a slot under the retention caps.

    Two caps rather than one: a flat "first fifty" fills up with whichever mode
    is commonest, and the point of keeping failures is to have one of *each* to
    look at. With the expert at 100% over the Step 8 gate there may well be
    none of any kind — the caps exist so that the day there are, the sample is
    worth eyeballing.
    """
    return sum(kept.values()) < FAILURE_KEEP_TOTAL and kept[outcome] < FAILURE_KEEP_PER_MODE


# --- The runner ------------------------------------------------------------------


class EpisodeRunner:
    """Runs attempts against one live scene, recording every control step."""

    def __init__(self, sim: Any, scene: Any, spec: Any, config: Any) -> None:
        import torch

        from manus import specs

        self.sim = sim
        self.scene = scene
        self.spec = spec
        self.config = config
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
        # Cost accounting, measured rather than assumed. The physics step and
        # the frame grab are timed apart because they are separately avoidable:
        # ``sim.step(render=True)`` drives the RTX pipeline, and
        # ``Camera.update`` is what pulls the result back across the bus.
        self.render_steps = 0
        self.render_seconds = 0.0
        self.plain_steps = 0
        self.plain_seconds = 0.0
        self.captures = 0
        self.capture_seconds = 0.0
        self.native_hw: tuple[int, int] | None = None

    # -- plumbing -----------------------------------------------------------------

    def advance(self, render: bool) -> None:
        """One physics step, refreshing only the buffers this script reads.

        Deliberately not ``scene.update()``: that ticks the wrist camera on
        every physics step, which is exactly the cost render decimation exists
        to avoid.
        """
        started = time.perf_counter()
        self.sim.step(render=render)
        elapsed = time.perf_counter() - started
        if render:
            self.render_steps += 1
            self.render_seconds += elapsed
        else:
            self.plain_steps += 1
            self.plain_seconds += elapsed
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

        Sensors refresh on ``update()``, not on ``step()``; the buffer this
        pulls was filled by the rendered physics step that ended the previous
        control tick, so it shows the state the joints are in *now*.
        """
        import numpy as np
        import torch
        from PIL import Image

        started = time.perf_counter()
        self.camera.update(CONTROL_DT)
        rgb = (
            self.camera.data.output["rgb"].torch[0, ..., :3].to(torch.uint8).cpu().numpy()
        )
        self.native_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        if rgb.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
            rgb = np.asarray(
                Image.fromarray(rgb).resize((FRAME_WIDTH, FRAME_HEIGHT), Image.LANCZOS),
                dtype=np.uint8,
            )
        frame = np.ascontiguousarray(rgb)
        self.captures += 1
        self.capture_seconds += time.perf_counter() - started
        return frame

    def probe_render(self, draw: Any, steps: int) -> dict[str, Any]:
        """Time `steps` control steps with the camera at 30 Hz and at 120 Hz.

        The render-decimation lever, measured end to end rather than inferred
        from the physics step alone: the same control loop, once rendering only
        on the capture tick (what the generator does) and once rendering every
        physics step and grabbing every frame (what a naive
        ``scene.update()``-driven loop does). Both hold the arm still at the
        post-settle pose, so the only difference between them is the renderer.
        """
        results = {}
        for label, every in (("decimated_30hz", False), ("every_step_120hz", True)):
            self.reset_episode(draw)
            self.capture()
            started = time.perf_counter()
            for _ in range(steps):
                self.scene.write_data_to_sim()
                for substep in range(recorder.DECIMATION):
                    render = every or substep == recorder.DECIMATION - 1
                    self.advance(render=render)
                    if every:
                        self.capture()
                if not every:
                    self.capture()
            results[label] = (time.perf_counter() - started) / steps
        results["speedup"] = results["every_step_120hz"] / results["decimated_30hz"]
        results["control_steps"] = steps
        return results

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

    # -- one attempt ----------------------------------------------------------------

    def run(self, attempt_index: int, draw: Any) -> dict[str, Any]:
        """Run and record one attempt. Returns its outcome dictionary.

        The recording order is the temporal contract: the frame, the measured
        joints and the action derived from them are all recorded *before* the
        physics step they precede, so ``joint_pos[t+1]`` is what ``action[t]``
        produced.
        """
        import torch

        from manus import specs
        from manus.expert import GraspSuccessMonitor, ScriptedGraspExpert, classify_outcome

        self.reset_episode(draw)
        measured = self.measured()
        frame = self.capture()

        expert = ScriptedGraspExpert(self.spec, config=self.config)
        plan = expert.reset(draw, q_current=measured)
        monitor = GraspSuccessMonitor(self.spec.spawn_z)
        episode = recorder.EpisodeRecorder(attempt_index)
        rest_z = self.object_z()
        # Kept alongside the recorder (which does not expose its buffers) so the
        # episode's droop can be summarised into its own meta.
        states: list[Any] = []
        commands: list[list[float]] = []

        stop_reason = "max_steps"
        for step in range(MAX_CONTROL_STEPS):
            if expert.done:
                stop_reason = "expert_done"
                break
            targets = expert.step(measured)
            command = [targets[name] for name in specs.JOINT_NAMES]
            episode.add_step(frame, measured, command, step * CONTROL_DT)
            states.append(measured)
            commands.append(command)
            self.robot.set_joint_position_target_index(
                target=torch.tensor(
                    [[targets[name] for name in specs.JOINT_NAMES]],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self.scene.write_data_to_sim()
            for substep in range(recorder.DECIMATION):
                self.advance(render=substep == recorder.DECIMATION - 1)
            measured = self.measured()
            frame = self.capture()
            monitor.update(self.object_z(), measured[self.gripper_column])
            if monitor.success:
                # The predicate is met: the 30 sustained steps that satisfied it
                # are the end of the demonstration (GraspSuccessMonitor's own
                # docstring: "the episode is cut there by the driver anyway").
                stop_reason = "success"
                break

        outcome = classify_outcome(expert, monitor)
        return {
            "attempt_index": attempt_index,
            "outcome": outcome,
            "success": bool(monitor.success),
            "stop_reason": stop_reason,
            "rest_z": rest_z,
            "episode": episode,
            "droop": episode_droop(commands, states),
            "monitor": monitor.to_dict(),
            "telemetry": expert.telemetry(),
            "plan_ok": plan.ok,
        }


# --- Manifest ---------------------------------------------------------------------


def write_manifest(
    dataset_dir: Path, env: dict[str, Any], generation: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild ``manifest.json`` from the ledger, plus this run's extras.

    ``build_manifest`` is a pure function of the ledger and the episodes and
    stays that way; the droop and generation blocks are attached to the object
    it returns before it is written, exactly as ``scripts/verify_dataset.py``
    attaches ``verify_result``. Regenerating the manifest drops them again,
    which is the honest behaviour: they are measurements of a *run*, not of the
    data, and a rebuild has not measured anything.
    """
    manifest = recorder.build_manifest(dataset_dir, env)
    manifest["droop"] = droop_stats(dataset_dir)
    manifest["generation"] = generation
    recorder.write_manifest(dataset_dir, manifest)
    return manifest


def accumulated_generation(dataset_dir: Path, chunk: dict[str, Any]) -> dict[str, Any]:
    """Fold this chunk's timing into whatever previous chunks recorded."""
    try:
        previous = recorder.read_manifest(dataset_dir).get("generation") or {}
    except FileNotFoundError:
        previous = {}
    totals = previous.get("totals") or {}
    return {
        "last_chunk": chunk,
        "totals": {
            "boots": int(totals.get("boots", 0)) + 1,
            "attempts": int(totals.get("attempts", 0)) + chunk["attempts"],
            "wall_clock_s": float(totals.get("wall_clock_s", 0.0)) + chunk["wall_clock_s"],
        },
    }


# --- Entry point --------------------------------------------------------------------


def plan_chunk(attempts: list[dict[str, Any]]) -> tuple[list[int], str]:
    """Attempt indices to run in this boot, and the status if there are none."""
    successes = sum(1 for record in attempts if record["outcome"] == recorder.SUCCESS)
    max_attempts = args_cli.max_attempts or 3 * args_cli.target_successes
    if successes >= args_cli.target_successes:
        return [], "complete"
    if len(attempts) >= max_attempts:
        return [], "exhausted"
    start = next_attempt_index(attempts)
    # Never plan more attempts than the budget allows, even if the chunk is
    # bigger: the last chunk of an unlucky run has to stop at max_attempts.
    room = min(args_cli.chunk, max_attempts - len(attempts))
    # Nor more than could conceivably be needed: with the expert at 100% a
    # 50-success target must not run 50 more attempts than that.
    room = min(room, args_cli.target_successes - successes)
    return list(range(start, start + room)), "pending"


def main() -> int:
    """Run one chunk. Returns the process exit code."""
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    from manus.expert import ExpertConfig
    from manus.objects import OBJECTS
    from manus.task_scene import grasp_scene_cfg

    dataset_dir = args_cli.root / "datasets" / "raw" / args_cli.dataset
    spec = OBJECTS[args_cli.object]
    config = ExpertConfig()

    attempts = recorder.read_attempts(dataset_dir)
    todo, _ = plan_chunk(attempts)
    successes = sum(1 for record in attempts if record["outcome"] == recorder.SUCCESS)
    print(
        f"{args_cli.dataset}: {successes}/{args_cli.target_successes} successes, "
        f"{len(attempts)} attempts on record"
    )

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=recorder.PHYSICS_DT, device=args_cli.device)
    )
    scene = InteractiveScene(grasp_scene_cfg(args_cli.object, num_envs=1, env_spacing=2.0))
    sim.reset()
    runner = EpisodeRunner(sim, scene, spec, config)
    env = env_block(sim, spec, config)

    kept_failures = retained_failures(attempts)
    started = time.time()
    frames_written = 0
    for position, attempt_index in enumerate(todo, start=1):
        seed = stable_hash64(args_cli.dataset, attempt_index)
        draw = draw_episode(args_cli.dataset, attempt_index)
        result = runner.run(attempt_index, draw)
        episode: recorder.EpisodeRecorder = result["episode"]

        meta = {
            "dataset": args_cli.dataset,
            "object": spec.name,
            "seed": seed,
            "draw": draw.to_dict(),
            "outcome": result["outcome"],
            "success": result["success"],
            "stop_reason": result["stop_reason"],
            "rest_z": result["rest_z"],
            "monitor": result["monitor"],
            "telemetry": result["telemetry"],
            "camera": env["camera"],
            "droop": result["droop"],
            "reset": {
                "settle_steps": SETTLE_STEPS,
                "warmup_renders": WARMUP_RENDERS,
                "home_pose": [float(value) for value in runner.home[0].cpu().numpy()],
            },
            "generator": "scripts/gen_dataset.py",
        }

        episode_file: str | None = None
        if not len(episode):
            print(f"  [{attempt_index:5d}] nothing recorded; keeping the ledger row only")
        elif result["success"]:
            path = episode.write(dataset_dir, meta)
            episode_file = path.name
            frames_written += len(episode)
        elif should_keep_failure(kept_failures, result["outcome"]):
            path = episode.write(dataset_dir / FAILURES_DIR, meta)
            episode_file = f"{FAILURES_DIR}/{path.name}"
            kept_failures[result["outcome"]] += 1

        recorder.append_attempt(
            dataset_dir,
            recorder.AttemptRecord(
                attempt_index=attempt_index,
                seed=seed,
                draw=draw.to_dict(),
                outcome=result["outcome"],
                episode_file=episode_file,
            ),
        )
        elapsed = time.time() - started
        print(
            f"  [{attempt_index:5d}] {result['outcome']:<12} {len(episode):4d} frames  "
            f"stop={result['stop_reason']:<11} "
            f"peak {result['monitor']['peak_z'] * 1e3:6.1f} mm  "
            f"({elapsed / position:5.1f} s/attempt)"
        )

    wall_clock = time.time() - started
    probe = (
        runner.probe_render(
            draw_episode(args_cli.dataset, next_attempt_index(attempts)), args_cli.probe_render
        )
        if args_cli.probe_render
        else None
    )
    render_mean = runner.render_seconds / max(1, runner.render_steps)
    plain_mean = runner.plain_seconds / max(1, runner.plain_steps)
    capture_mean = runner.capture_seconds / max(1, runner.captures)
    chunk = {
        "attempts": len(todo),
        "frames": frames_written,
        "wall_clock_s": wall_clock,
        "seconds_per_attempt": wall_clock / len(todo) if todo else None,
        "render": {
            "native_hw": list(runner.native_hw) if runner.native_hw else None,
            "rendered_steps": runner.render_steps,
            "plain_steps": runner.plain_steps,
            "rendered_step_s": render_mean,
            "plain_step_s": plain_mean,
            "capture_s": capture_mean,
            "captures": runner.captures,
            "probe": probe,
        },
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest = write_manifest(dataset_dir, env, accumulated_generation(dataset_dir, chunk))

    counts = manifest["counts"]
    print(
        f"\nchunk: {len(todo)} attempts in {wall_clock:.0f} s"
        + (f" ({wall_clock / len(todo):.1f} s/attempt)" if todo else "")
    )
    if probe:
        print(
            f"render probe over {probe['control_steps']} control steps: "
            f"{probe['decimated_30hz'] * 1e3:.1f} ms/step capturing at 30 Hz vs "
            f"{probe['every_step_120hz'] * 1e3:.1f} ms/step at 120 Hz -> "
            f"{probe['speedup']:.2f}x throughput from render decimation"
        )
    if runner.render_steps:
        print(
            f"per step: {render_mean * 1e3:.1f} ms rendered physics, "
            f"{plain_mean * 1e3:.1f} ms plain physics, "
            f"{capture_mean * 1e3:.1f} ms frame grab "
            f"({runner.native_hw[0]}x{runner.native_hw[1]} native)"
        )
    print(
        f"dataset: {counts['successes']} successes / {counts['attempts']} attempts, "
        f"{counts['frames']} frames, id {manifest['dataset_id'][:12]}"
    )

    attempts = recorder.read_attempts(dataset_dir)
    _, status = plan_chunk(attempts)
    print(f"STATUS: {status}")
    return 1 if status == "exhausted" else 0


def main_no_gpu() -> int:
    """Handle the cases that need no simulator: nothing pending, or --manifest-only.

    A finished dataset is reported, not rewritten. The manifest is a tracked
    file that later steps add to (``scripts/verify_dataset.py`` writes its
    ``verify_result`` into it), and rebuilding it here would silently drop that
    and restamp ``created`` every time the driver loop asked whether there was
    anything left to do. ``--manifest-only`` is the explicit way to rebuild.
    """
    dataset_dir = args_cli.root / "datasets" / "raw" / args_cli.dataset
    if not dataset_dir.is_dir():
        raise SystemExit(f"no such dataset: {dataset_dir}")
    attempts = recorder.read_attempts(dataset_dir)
    _, status = plan_chunk(attempts)
    try:
        previous = recorder.read_manifest(dataset_dir)
    except FileNotFoundError:
        previous = {}
    if args_cli.manifest_only or not previous:
        manifest = write_manifest(
            dataset_dir,
            previous.get("env") or {},
            previous.get("generation") or {},
        )
        print(f"rebuilt {dataset_dir / recorder.MANIFEST_NAME}")
    else:
        manifest = previous
    counts = manifest["counts"]
    print(
        f"{args_cli.dataset}: {counts['successes']} successes / {counts['attempts']} attempts, "
        f"{counts['frames']} frames, id {manifest['dataset_id'][:12]}"
    )
    print(f"STATUS: {status}")
    return 1 if status == "exhausted" else 0


if __name__ == "__main__":
    _dataset_dir = args_cli.root / "datasets" / "raw" / args_cli.dataset
    _attempts = recorder.read_attempts(_dataset_dir)
    _todo, _status = plan_chunk(_attempts)
    if args_cli.manifest_only or not (_todo or args_cli.probe_render):
        # Nothing to simulate: rebuild the manifest from disk and say where the
        # dataset stands, without paying for (or occupying) the GPU.
        sys.exit(main_no_gpu())

    free = free_vram_mib()
    if free is not None and free < MIN_FREE_VRAM_MIB:
        raise SystemExit(
            f"pre-flight: only {free} MiB of VRAM free, need >= {MIN_FREE_VRAM_MIB} MiB "
            "(shared GPU: one process at a time)"
        )
    print(
        f"pre-flight: {free} MiB free VRAM; "
        + (f"running attempts {_todo[0]}..{_todo[-1]}" if _todo else "probe only")
    )

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
