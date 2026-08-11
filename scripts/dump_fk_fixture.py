"""Dump Isaac's link poses for a batch of random arm configurations, as the
ground truth ``tests/test_kinematics.py::test_fk_matches_isaac`` checks the
pure-numpy FK against.

Kinematics only: each configuration is written straight into the articulation
and ``sim.forward()`` refreshes the transforms without advancing physics, so
gravity, the PD controller and contacts never touch the recorded poses. This
script must never call ``sim.step()``.

Poses are stored relative to the root (``base_link``) pose, so the fixture is
independent of where the robot happens to be spawned, and joint angles are the
values read back from the simulator rather than the commanded ones, so any
clamping PhysX applies is part of the ground truth.

.. code-block:: bash

    ~/isaaclab-env/bin/python scripts/dump_fk_fixture.py --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the src-layout package importable without installing it. This must
# happen before the manus imports below, but manus.robot is deliberately
# deferred until after AppLauncher has started Isaac Sim: it pulls in isaaclab
# extensions that only exist on a live app.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--configs", type=int, default=100, help="how many random configurations to dump")
parser.add_argument("--seed", type=int, default=0, help="seed for the configuration sampler")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from manus import kinematics, specs
from manus.robot import SO101_CFG, SO101_USD_PATH

PHYSICS_DT = 1.0 / 120.0
"""Simulation step, in seconds. Never actually consumed -- nothing is stepped."""

DECIMALS = 9
"""Rounding applied to the recorded floats: 1 nm / 1e-9 of a rotation entry,
far below the 0.5 mm / 0.1 deg tolerances the fixture is compared at."""

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fk_fixture.json"
"""Where the fixture is written."""


def _root_relative(
    positions: np.ndarray, quats: np.ndarray, root_position: np.ndarray, root_quat: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Express world body poses in the root frame.

    Args:
        positions: Shape (num_bodies, 3) world positions, metres.
        quats: Shape (num_bodies, 4) world orientations, ``(x, y, z, w)``.
        root_position: Shape (3,) world position of the articulation root.
        root_quat: Shape (4,) world orientation of the root, ``(x, y, z, w)``.

    Returns:
        One ``(position, rotation)`` pair per body, in root coordinates.
    """
    root_rotation = kinematics.rotation_from_quat_xyzw(root_quat)
    return [
        (
            root_rotation.T @ (position - root_position),
            root_rotation.T @ kinematics.rotation_from_quat_xyzw(quat),
        )
        for position, quat in zip(positions, quats, strict=True)
    ]


def main() -> int:
    """Dump the fixture. Returns the process exit code."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))
    # The robot alone: no ground, no lights, nothing that could move it.
    robot = Articulation(SO101_CFG)
    sim.reset()

    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    assert joint_names == list(specs.JOINT_NAMES), (
        f"joint order mismatch: {joint_names} != {list(specs.JOINT_NAMES)}"
    )
    missing = [link for link in kinematics.CHAIN_LINKS if link not in body_names]
    assert not missing, f"chain links absent from the articulation: {missing} (have {body_names})"

    device = robot.data.joint_pos.torch.device
    lower = np.array([specs.JOINT_LIMITS[name][0] for name in joint_names])
    upper = np.array([specs.JOINT_LIMITS[name][1] for name in joint_names])
    # The gripper is off the FK chain, so its value carries no information;
    # collapsing its range to zero keeps the dump deterministic and readable.
    gripper = joint_names.index("gripper")
    lower[gripper] = upper[gripper] = 0.0

    rng = np.random.default_rng(args_cli.seed)
    configs = []
    for _ in range(args_cli.configs):
        position = torch.tensor(
            rng.uniform(lower, upper)[None, :], dtype=torch.float32, device=device
        )
        robot.write_joint_state_to_sim_index(
            position=position, velocity=torch.zeros_like(position), full_data=True
        )
        sim.forward()  # kinematics-only update; never sim.step() here
        robot.update(0.0)

        poses = _root_relative(
            robot.data.body_link_pos_w.torch[0].cpu().numpy(),
            robot.data.body_link_quat_w.torch[0].cpu().numpy(),
            robot.data.root_link_pos_w.torch[0].cpu().numpy(),
            robot.data.root_link_quat_w.torch[0].cpu().numpy(),
        )
        bodies = {
            name: {
                "pos": np.round(body_position, DECIMALS).tolist(),
                "rot": np.round(rotation, DECIMALS).tolist(),
            }
            for name, (body_position, rotation) in zip(body_names, poses, strict=True)
        }
        # The fixture frame is only base_link if the root really is base_link.
        base = bodies[kinematics.BASE_LINK]
        assert np.allclose(base["pos"], 0.0, atol=1e-6) and np.allclose(
            base["rot"], np.eye(3), atol=1e-6
        ), f"root is not {kinematics.BASE_LINK}: it sits at {base['pos']}"

        configs.append(
            {
                "joint_pos": np.round(
                    robot.data.joint_pos.torch[0].cpu().numpy(), DECIMALS
                ).tolist(),
                "bodies": bodies,
            }
        )

    # Isaac's pose buffers are cached views; if one ever failed to refresh, the
    # dump would be N copies of a single pose rather than an obvious error.
    if len(configs) > 1:
        tcp = np.array([config["bodies"][kinematics.TCP_LINK]["pos"] for config in configs])
        assert np.ptp(tcp, axis=0).max() > 1e-3, (
            "every configuration recorded the same TCP pose -- the pose buffers went stale"
        )

    fixture = {
        "generated_by": "scripts/dump_fk_fixture.py",
        # Repo-relative: the fixture is committed, and an absolute path would
        # bake the generating machine's home directory into tracked content.
        "usd_path": str(
            Path(SO101_USD_PATH).resolve().relative_to(Path(__file__).resolve().parents[1])
        ),
        "seed": args_cli.seed,
        "num_configs": len(configs),
        "pose_frame": kinematics.BASE_LINK,
        "source_quat_order": "xyzw",  # Isaac Lab 3.x body_link_quat_w convention
        "joint_names": joint_names,
        "body_names": body_names,
        "configs": configs,
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Machine-written and read only by pytest, so compact rather than indented.
    FIXTURE_PATH.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")

    # Spot-check against the numpy chain the fixture exists to validate. The
    # gate itself is test_fk_matches_isaac; this is only here so a bad dump is
    # obvious without leaving the terminal.
    chain = kinematics.KinematicChain()
    arm_columns = [joint_names.index(name) for name in kinematics.ARM_JOINT_NAMES]
    worst_position, worst_rotation = 0.0, 0.0
    for config in configs:
        predicted = chain.fk(np.asarray(config["joint_pos"])[arm_columns])
        for link in kinematics.CHAIN_LINKS:
            body = config["bodies"][link]
            worst_position = max(
                worst_position, float(np.linalg.norm(predicted[link][0] - np.asarray(body["pos"])))
            )
            worst_rotation = max(
                worst_rotation,
                kinematics.rotation_error_deg(predicted[link][1], np.asarray(body["rot"])),
            )

    print(f"PASS: {len(configs)} configs x {len(body_names)} bodies -> {FIXTURE_PATH}")
    print(f"PASS: worst FK deviation {worst_position * 1e3:.4f} mm, {worst_rotation:.4f} deg")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
