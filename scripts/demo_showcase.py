"""Third-person showcase: run the demo sequence while an exterior camera slowly
orbits the whole arm, with the wrist POV inset in the corner.

.. code-block:: bash

    ~/isaaclab-env/bin/python scripts/demo_showcase.py --headless
"""

from __future__ import annotations

import argparse
import math
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

# The scene always carries cameras, and Isaac Lab's Camera refuses to
# initialise unless the app renders sensors -- so require it rather than making
# the caller remember --enable_cameras.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

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

EXT_WIDTH, EXT_HEIGHT = 1280, 720
"""Exterior camera resolution."""

INSET_WIDTH, INSET_HEIGHT = 320, 240
"""Wrist POV inset size within the exterior frame."""

INSET_MARGIN = 16
"""Pixels between the inset and the frame edge."""

ORBIT_RADIUS = 1.0
"""Exterior camera distance from the orbit axis, in metres."""

ORBIT_HEIGHT = 0.45
"""Exterior camera eye height, in metres."""

ORBIT_START = math.radians(45.0)
"""Starting azimuth of the orbit (3/4 front view)."""

LOOK_AT = (0.0, 0.0, 0.12)
"""Point the exterior camera keeps in the centre of frame, in metres."""

VIDEO_PATH = Path(__file__).resolve().parents[1] / "runs" / "demo_showcase.mp4"
"""Where the composed showcase video is written."""


@configclass
class ShowcaseSceneCfg(SoArmSceneCfg):
    """The standard scene plus a free-flying exterior camera (posed per frame)."""

    exterior_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/exterior_cam",
        update_period=0.0,
        width=EXT_WIDTH,
        height=EXT_HEIGHT,
        data_types=["rgb"],
        # ~60 deg hFOV: wide enough to hold the whole arm at 1 m distance.
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
    )


def compose(exterior: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Paste the wrist POV (with a 2 px white border) into the bottom-right corner."""
    frame = exterior.copy()
    inset = np.asarray(
        Image.fromarray(wrist).resize((INSET_WIDTH, INSET_HEIGHT), Image.BILINEAR)
    )
    top = EXT_HEIGHT - INSET_MARGIN - INSET_HEIGHT
    left = EXT_WIDTH - INSET_MARGIN - INSET_WIDTH
    frame[top - 2 : top + INSET_HEIGHT + 2, left - 2 : left + INSET_WIDTH + 2] = 255
    frame[top : top + INSET_HEIGHT, left : left + INSET_WIDTH] = inset
    return frame


def main() -> int:
    """Run the showcase. Returns the process exit code."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))
    scene = InteractiveScene(ShowcaseSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    wrist_cam = scene["wrist_cam"]
    exterior_cam = scene["exterior_cam"]
    sim_dt = sim.get_physics_dt()
    device = robot.data.joint_pos.torch.device
    total_steps = DEMO_SEQUENCE.total_steps(STEPS_PER_SEGMENT)
    frames: list[np.ndarray] = []

    # One extra step so the loop lands exactly on the final waypoint boundary.
    for step in range(total_steps + 1):
        # One full, slow orbit over the whole sequence.
        azimuth = ORBIT_START + 2.0 * math.pi * step / total_steps
        eye = (
            LOOK_AT[0] + ORBIT_RADIUS * math.cos(azimuth),
            LOOK_AT[1] + ORBIT_RADIUS * math.sin(azimuth),
            ORBIT_HEIGHT,
        )
        exterior_cam.set_world_poses_from_view(
            torch.tensor([eye], device=device), torch.tensor([LOOK_AT], device=device)
        )

        targets = DEMO_SEQUENCE.at(step, STEPS_PER_SEGMENT)
        # Targets (radians) in the articulation's own joint order.
        target = torch.tensor([[targets[name] for name in robot.joint_names]], device=device)
        robot.set_joint_position_target_index(target=target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if step % VIDEO_STRIDE == 0:
            exterior = exterior_cam.data.output["rgb"].torch[0].to(torch.uint8).cpu().numpy()
            wrist = wrist_cam.data.output["rgb"].torch[0].to(torch.uint8).cpu().numpy()
            frames.append(compose(exterior, wrist))

    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(VIDEO_PATH, np.stack(frames), fps=VIDEO_FPS)
    print(f"PASS: wrote {len(frames)} showcase frames at {VIDEO_FPS} fps to {VIDEO_PATH}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
