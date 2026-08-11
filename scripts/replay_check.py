"""Replay gate: re-drive recorded episodes open-loop and see if the arm agrees.

The check that makes the dataset trustworthy. Three episodes are re-run in a
fresh scene from the *same* draw, driven only by their recorded ``actions``
array — no expert, no feedback, nothing that could quietly re-derive what the
data should have been — and the joint trajectory that comes out is compared
against the recorded ``joint_pos``:

.. code-block:: bash

    ~/isaaclab-env/bin/python scripts/replay_check.py --dataset grasp_cube_dev --headless

It is the only test that can see an off-by-one. Every other check reads the
arrays as a self-consistent unit; this one puts them back into the physics that
produced them, so it fails loudly on the mistakes that are invisible on disk and
fatal to a policy: ``action[t]`` recorded against the wrong state, joints in the
wrong column order, degrees stored where radians were meant.

**Tolerance: 0.05 rad, max over every joint and step.** Not equality, because
GPU PhysX is not bit-reproducible — the same commands on the same driver
re-converge to slightly different states, and the recorded run was rendering
while this one is not. 0.05 rad is ~2.9 deg, an order of magnitude above that
drift and an order of magnitude *below* the errors the failures it targets
produce: a one-step index shift replays a command from the wrong point of a
ramp (the pregrasp ramp alone traverses ~1.5 rad over 45 steps, so a single
step of shift is ~0.03 rad of *immediate* error that then compounds over the
episode), a swapped joint pair or a degree/radian confusion is off by radians.

**Ramp-start sanity.** Separately, the first recorded action is checked against
the first recorded state: under the temporal contract they are different rows —
``action[0]`` is the target the expert issued *from* ``joint_pos[0]``, one step
of the PREGRASP ramp towards the planned pregrasp pose — so the two must differ,
and the difference must point at that pose. If the generator had recorded the
action alongside the state it produced, this is where it would show.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from isaaclab.app import AppLauncher  # noqa: E402 - importing does not start the app

from manus import recorder  # noqa: E402 - sim-free

TOLERANCE_RAD = 0.05
"""Gate: max |replayed - recorded| joint error, radians (see the module docstring)."""

RAMP_START_MIN_RAD = 1e-4
"""Smallest ``|action[0] - joint_pos[0]|`` that counts as "these are different rows"."""

RAMP_DIRECTION_MIN_RAD = 0.02
"""Joint travel below which the pregrasp direction check is not meaningful."""

MIN_FREE_VRAM_MIB = 6500
"""Pre-flight floor on free VRAM (shared GPU)."""

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--dataset", required=True, help="dataset name, e.g. grasp_cube_dev")
parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
parser.add_argument(
    "--episodes", type=int, default=3, help="how many episodes to replay (first/middle/last)"
)
parser.add_argument(
    "--tolerance", type=float, default=TOLERANCE_RAD, help="gate on max joint error, radians"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # the scene carries the wrist camera


def chosen_episodes(dataset_dir: Path, wanted: int) -> list[Path]:
    """First, middle and last successful episodes, in ledger order."""
    paths = [
        dataset_dir / record["episode_file"]
        for record in recorder.read_attempts(dataset_dir)
        if record["outcome"] == recorder.SUCCESS and record.get("episode_file")
    ]
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise SystemExit(f"{dataset_dir}: no successful episodes to replay")
    if len(paths) <= wanted:
        return paths
    picks = [0, len(paths) // 2, len(paths) - 1] if wanted == 3 else [
        round(index * (len(paths) - 1) / (wanted - 1)) for index in range(wanted)
    ]
    return [paths[index] for index in dict.fromkeys(picks)]


def ramp_start_check(episode: recorder.Episode, spec: Any, draw: Any) -> dict[str, Any]:
    """Is ``action[0]`` a step of the pregrasp ramp away from ``joint_pos[0]``?

    The plan is re-solved on the CPU from the same draw — ``plan_grasp`` is
    sim-free and deterministic — so the expected direction comes from the
    expert's own planner rather than from anything the recording could have got
    wrong in the same way twice.
    """
    import numpy as np

    from manus import kinematics, specs
    from manus.control import GRIPPER_OPEN
    from manus.expert import plan_grasp

    state = episode.joint_pos[0].astype(float)
    action = episode.actions[0].astype(float)
    delta = action - state

    plan = plan_grasp(spec, draw, q_current=state)
    towards = np.zeros(len(specs.JOINT_NAMES))
    towards[: kinematics.NUM_ARM_JOINTS] = plan.q_pregrasp - state[: kinematics.NUM_ARM_JOINTS]
    # The jaws open during PREGRASP, so their target moves towards gripper_open.
    towards[specs.JOINT_NAMES.index("gripper")] = (
        GRIPPER_OPEN - state[specs.JOINT_NAMES.index("gripper")]
    )

    moved = float(np.abs(delta).max())
    considered = [
        index for index in range(len(specs.JOINT_NAMES))
        if abs(towards[index]) >= RAMP_DIRECTION_MIN_RAD
    ]
    wrong = [
        specs.JOINT_NAMES[index]
        for index in considered
        if np.sign(delta[index]) != np.sign(towards[index])
    ]
    return {
        "differs": moved > RAMP_START_MIN_RAD,
        "max_abs_rad": moved,
        "joints_checked": [specs.JOINT_NAMES[index] for index in considered],
        "wrong_direction": wrong,
        "delta_mrad": [float(value * 1e3) for value in delta],
        "towards_mrad": [float(value * 1e3) for value in towards],
        "passed": moved > RAMP_START_MIN_RAD and not wrong,
    }


class Replayer:
    """Drives recorded action arrays through a live scene, open-loop."""

    def __init__(self, sim: Any, scene: Any, spec: Any) -> None:
        import torch

        from manus import specs

        self.sim = sim
        self.scene = scene
        self.spec = spec
        self.robot = scene["robot"]
        self.object = scene["object"]
        self.dt = sim.get_physics_dt()
        self.device = self.robot.data.joint_pos.torch.device
        assert self.robot.joint_names == list(specs.JOINT_NAMES), self.robot.joint_names
        self.home = torch.tensor(
            [[specs.HOME_POSE[name] for name in specs.JOINT_NAMES]],
            dtype=torch.float32,
            device=self.device,
        )

    def advance(self, render: bool) -> None:
        """One physics step. Renders on the same schedule the recording used.

        Rendering cannot move the arm, but replicating the call pattern costs
        almost nothing (the expensive half is the frame grab, which is skipped
        here) and removes one difference between the two runs.
        """
        self.sim.step(render=render)
        self.robot.update(self.dt)
        self.object.update(self.dt)

    def measured(self) -> Any:
        """Six measured joint positions, radians, in ``specs.JOINT_NAMES`` order."""
        return self.robot.data.joint_pos.torch[0].detach().cpu().numpy().astype(float)

    def replay(self, episode: recorder.Episode) -> dict[str, Any]:
        """Re-drive one episode from its recorded actions and score the result."""
        import numpy as np
        import torch

        from manus.randomize import EpisodeDraw
        from manus.task_scene import apply_randomization

        meta = episode.meta
        draw = EpisodeDraw.from_dict(meta["draw"])
        reset = meta.get("reset") or {}
        settle = int(reset.get("settle_steps", 30))
        warmup = int(reset.get("warmup_renders", 0))
        decimation = int(meta.get("decimation", recorder.DECIMATION))

        self.robot.write_joint_state_to_sim_index(
            position=self.home, velocity=torch.zeros_like(self.home), full_data=True
        )
        self.robot.set_joint_position_target_index(target=self.home)
        apply_randomization(self.scene, draw, self.spec)
        self.scene.reset()
        self.scene.write_data_to_sim()
        for step in range(settle):
            self.advance(render=step >= settle - warmup)

        start_error = float(np.abs(self.measured() - episode.joint_pos[0].astype(float)).max())
        errors = np.zeros((len(episode) - 1, episode.joint_pos.shape[1]))
        for step in range(len(episode) - 1):
            self.robot.set_joint_position_target_index(
                target=torch.from_numpy(np.ascontiguousarray(episode.actions[step : step + 1]))
                .to(dtype=torch.float32, device=self.device)
            )
            self.scene.write_data_to_sim()
            for substep in range(decimation):
                self.advance(render=substep == decimation - 1)
            # The temporal contract: action[t] produced joint_pos[t+1].
            errors[step] = np.abs(self.measured() - episode.joint_pos[step + 1].astype(float))

        per_joint = errors.max(axis=0)
        worst_step = int(np.unravel_index(int(np.argmax(errors)), errors.shape)[0])
        return {
            "attempt_index": episode.attempt_index,
            "steps": len(episode),
            "start_error_rad": start_error,
            "max_abs_rad": float(errors.max()),
            "mean_abs_rad": float(errors.mean()),
            "per_joint_max_rad": [float(value) for value in per_joint],
            "worst_step": worst_step,
        }


def main() -> int:
    """Replay the chosen episodes and report the gate. Returns the exit code."""
    import numpy as np

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    from manus import specs
    from manus.objects import OBJECTS
    from manus.randomize import EpisodeDraw
    from manus.task_scene import GraspSceneCfg

    dataset_dir = args_cli.root / "datasets" / "raw" / args_cli.dataset
    paths = chosen_episodes(dataset_dir, args_cli.episodes)
    episodes = [recorder.load_episode(path) for path in paths]
    object_names = {episode.meta.get("object", "cube_3cm") for episode in episodes}
    if len(object_names) != 1:
        raise SystemExit(f"episodes mix objects {object_names}; replay one object at a time")
    spec = OBJECTS[object_names.pop()]
    physics_dt = float(episodes[0].meta.get("physics_dt", recorder.PHYSICS_DT))

    print(f"{args_cli.dataset}: replaying attempts {[e.attempt_index for e in episodes]}")
    print(f"contract: {episodes[0].meta.get('temporal_contract')}")

    # Ramp-start sanity is pure CPU: do it before spending the GPU.
    ramps = [
        ramp_start_check(episode, spec, EpisodeDraw.from_dict(episode.meta["draw"]))
        for episode in episodes
    ]
    for episode, ramp in zip(episodes, ramps, strict=True):
        print(
            f"  ramp start, attempt {episode.attempt_index}: "
            f"|action[0] - joint_pos[0]| max {ramp['max_abs_rad'] * 1e3:.1f} mrad, "
            f"{len(ramp['joints_checked'])} joints checked, "
            f"wrong direction {ramp['wrong_direction'] or 'none'} -> "
            f"{'PASS' if ramp['passed'] else 'FAIL'}"
        )

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, device=args_cli.device))
    scene = InteractiveScene(GraspSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    replayer = Replayer(sim, scene, spec)

    results = [replayer.replay(episode) for episode in episodes]
    print(f"\n{'attempt':>8} {'steps':>6} {'start':>8} {'max err':>9}  per-joint max [mrad]")
    for result in results:
        print(
            f"{result['attempt_index']:8d} {result['steps']:6d} "
            f"{result['start_error_rad'] * 1e3:7.1f}m {result['max_abs_rad'] * 1e3:8.1f}m  "
            + " ".join(f"{value * 1e3:6.1f}" for value in result["per_joint_max_rad"])
        )
    print(" " * 34 + "  " + " ".join(f"{name[:6]:>6}" for name in specs.JOINT_NAMES))

    worst = max(result["max_abs_rad"] for result in results)
    per_joint = np.max([result["per_joint_max_rad"] for result in results], axis=0)
    ramps_passed = all(ramp["passed"] for ramp in ramps)
    passed = worst < args_cli.tolerance and ramps_passed
    print(
        f"\nworst joint: {specs.JOINT_NAMES[int(np.argmax(per_joint))]} at "
        f"{worst * 1e3:.1f} mrad; tolerance {args_cli.tolerance * 1e3:.0f} mrad"
    )
    print(
        f"{'PASS' if passed else 'FAIL'}: replay gate over {len(results)} episodes "
        f"(open-loop max {worst:.4f} rad < {args_cli.tolerance} rad, "
        f"ramp-start {'ok' if ramps_passed else 'FAILED'})"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    from manus.recorder import episode_paths  # noqa: F401 - fail fast on a broken src path

    free = None
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            free = int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError):
        free = None
    if free is not None and free < MIN_FREE_VRAM_MIB:
        raise SystemExit(
            f"pre-flight: only {free} MiB of VRAM free, need >= {MIN_FREE_VRAM_MIB} MiB"
        )
    print(f"pre-flight: {free} MiB free VRAM")

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
