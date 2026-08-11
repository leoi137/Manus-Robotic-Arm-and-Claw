"""Drive the SO-101 through the scripted demo pose sequence, optionally
recording the wrist POV.

.. code-block:: bash

    ~/isaaclab-env/bin/python scripts/demo_poses.py --headless --video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the src-layout package importable without installing it. This must
# happen before the manus imports below, but those imports are deliberately
# deferred until after AppLauncher has started Isaac Sim: manus.robot and
# manus.scene pull in isaaclab extensions that only exist on a live app.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--video", action="store_true", help="record the wrist POV to runs/demo_wrist_pov.mp4")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The scene always carries the wrist camera, and Isaac Lab's Camera refuses to
# initialise unless the app renders sensors -- so require it rather than making
# the caller remember --enable_cameras.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio.v3 as iio
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from manus import specs
from manus.control import DEMO_SEQUENCE
from manus.scene import SoArmSceneCfg

PHYSICS_DT = 1.0 / 120.0
"""Simulation step, in seconds."""

STEPS_PER_SEGMENT = 120
"""Steps spent interpolating between two waypoints, i.e. 1 s per segment."""

VIDEO_STRIDE = 4
"""Capture every Nth step; 120 Hz / 4 = 30 captured frames per second of sim."""

VIDEO_FPS = 30
"""Playback rate, chosen as 1 / (PHYSICS_DT * VIDEO_STRIDE) so video is real-time."""

HOME_TOLERANCE = 0.15
"""Tolerated deviation from home at the end of the sequence, in radians."""

VIDEO_PATH = Path(__file__).resolve().parents[1] / "runs" / "demo_wrist_pov.mp4"
"""Where --video writes the wrist POV recording."""


def main() -> int:
    """Run the pose demo. Returns the process exit code."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))
    scene = InteractiveScene(SoArmSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    camera = scene["wrist_cam"]
    sim_dt = sim.get_physics_dt()
    device = robot.data.joint_pos.torch.device
    frames: list[np.ndarray] = []

    # One extra step so the loop lands exactly on the final waypoint boundary.
    for step in range(DEMO_SEQUENCE.total_steps(STEPS_PER_SEGMENT) + 1):
        targets = DEMO_SEQUENCE.at(step, STEPS_PER_SEGMENT)
        # Targets (radians) in the articulation's own joint order.
        target = torch.tensor([[targets[name] for name in robot.joint_names]], device=device)
        robot.set_joint_position_target_index(target=target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if args_cli.video and step % VIDEO_STRIDE == 0:
            frames.append(camera.data.output["rgb"].torch[0].to(torch.uint8).cpu().numpy())

        waypoint, offset = divmod(step, STEPS_PER_SEGMENT)
        if offset == 0:
            measured = robot.data.joint_pos.torch[0].tolist()
            reached = "  ".join(f"{n}={v:+.4f}" for n, v in zip(robot.joint_names, measured))
            print(f"{DEMO_SEQUENCE.waypoints[waypoint][0]:>16}: {reached}")

    if args_cli.video:
        VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(VIDEO_PATH, np.stack(frames), fps=VIDEO_FPS)
        print(f"wrote {len(frames)} wrist POV frames at {VIDEO_FPS} fps to {VIDEO_PATH}")

    # The sequence ends on the home waypoint, so the arm must be back at home.
    home = torch.tensor([[specs.HOME_POSE[name] for name in robot.joint_names]], device=device)
    error = (robot.data.joint_pos.torch - home).abs().max().item()
    assert error < HOME_TOLERANCE, f"ended {error:.4f} rad from home, expected < {HOME_TOLERANCE} rad"

    print(f"PASS: {len(DEMO_SEQUENCE.waypoints)} waypoints, back home within {error:.4f} rad")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
