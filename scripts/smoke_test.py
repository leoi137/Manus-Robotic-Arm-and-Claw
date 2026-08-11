"""Headless smoke test: load the SO-101 scene, hold home pose, assert it settles,
and check the wrist camera renders a real (non-black) frame.

.. code-block:: bash

    ~/isaaclab-env/bin/python scripts/smoke_test.py --headless
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The scene always carries the wrist camera, and Isaac Lab's Camera refuses to
# initialise unless the app renders sensors -- so require it rather than making
# the caller remember --enable_cameras.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from manus import specs
from manus.scene import WRIST_CAM_HEIGHT, WRIST_CAM_WIDTH, SoArmSceneCfg

NUM_STEPS = 200
"""Physics steps to hold the home pose for."""

MAX_SAG = 0.1
"""Tolerated steady-state deviation from home under gravity, in radians."""

RGB_SHAPE = (1, WRIST_CAM_HEIGHT, WRIST_CAM_WIDTH, 3)
"""Expected wrist camera output: (num_envs, H, W, RGB), uint8."""

MIN_RGB_STD = 5.0
"""Minimum per-frame intensity spread (0-255 levels) for a frame with content."""

FRAME_PATH = Path(__file__).resolve().parents[1] / "runs" / "wrist_cam_smoke.png"
"""Where the captured wrist POV frame is written for eyeballing."""


def main() -> int:
    """Run the smoke test. Returns the process exit code."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    scene = InteractiveScene(SoArmSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    # Home pose (radians) in the articulation's own joint order.
    home = torch.tensor(
        [[specs.HOME_POSE[name] for name in robot.joint_names]],
        device=robot.data.joint_pos.torch.device,
    )

    for _ in range(NUM_STEPS):
        robot.set_joint_position_target_index(target=home)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    joint_pos = robot.data.joint_pos.torch
    assert robot.joint_names == list(specs.JOINT_NAMES), (
        f"joint order mismatch: {robot.joint_names} != {list(specs.JOINT_NAMES)}"
    )
    assert torch.isfinite(joint_pos).all(), f"non-finite joint positions: {joint_pos}"
    sag = (joint_pos - home).abs().max().item()
    assert sag < MAX_SAG, f"drifted {sag:.4f} rad from home, expected < {MAX_SAG} rad"

    rgb = scene["wrist_cam"].data.output["rgb"].torch
    assert tuple(rgb.shape) == RGB_SHAPE, f"wrist cam gave {tuple(rgb.shape)}, expected {RGB_SHAPE}"
    frame = rgb[0].to(torch.uint8).cpu().numpy()
    assert frame.any(), "wrist cam frame is uniformly black"
    std = float(np.asarray(frame, dtype=np.float32).std())
    assert std > MIN_RGB_STD, f"wrist cam frame is near-flat: std {std:.2f} <= {MIN_RGB_STD}"

    FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(FRAME_PATH)

    print(f"PASS: {robot.num_joints} joints, {NUM_STEPS} steps at home, max sag {sag:.4f} rad")
    print(f"PASS: wrist cam {tuple(rgb.shape)} rgb, std {std:.2f}, saved {FRAME_PATH}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
