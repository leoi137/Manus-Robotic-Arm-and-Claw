"""GUI keyboard teleop: jog each SO-101 joint individually from the app window.

Needs the Isaac Sim GUI (do not pass --headless): key events are read off the
app window's keyboard.

.. code-block:: bash

    ~/isaaclab-env/bin/python scripts/teleop_keyboard.py
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
parser.add_argument(
    "--autoclose",
    type=float,
    default=None,
    help="exit after this many seconds of wall time; for automated testing",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The scene always carries the wrist camera, and Isaac Lab's Camera refuses to
# initialise unless the app renders sensors -- so require it rather than making
# the caller remember --enable_cameras.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import weakref

import carb
import omni
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from manus import control, specs
from manus.scene import SoArmSceneCfg

PHYSICS_DT = 1.0 / 120.0
"""Simulation step, in seconds."""

PRINT_EVERY = round(1.0 / PHYSICS_DT)
"""Steps between joint-state printouts, i.e. once per second of sim time."""


class JointJogKeyboard:
    """Turns app-window key presses into per-joint position targets (radians).

    Isaac Lab's own keyboard devices (``Se2Keyboard``/``Se3Keyboard``) only emit
    task-space commands, so this subscribes to ``carb.input`` directly, using the
    same weakref'd subscription pattern ``Se3Keyboard`` uses.
    """

    def __init__(self) -> None:
        self.targets: dict[str, float] = dict(specs.HOME_POSE)
        self.stop_requested = False
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        # Weakref in the callback so the live subscription cannot keep this
        # object alive past its owner.
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def close(self) -> None:
        """Release the keyboard subscription."""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
        self._subscription = None

    def _on_keyboard_event(self, event, *args) -> bool:
        """Jog on key-down; ESC asks the run loop to stop. Unmapped keys no-op."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "ESCAPE":
                self.stop_requested = True
            else:
                self.targets = control.apply_jog(self.targets, event.input.name)
        return True


def print_key_help() -> None:
    """Print the jog key table, joint by joint."""
    increase = {index: key for key, (index, direction) in control.KEYMAP.items() if direction > 0}
    decrease = {index: key for key, (index, direction) in control.KEYMAP.items() if direction < 0}
    print(f"\nSO-101 keyboard teleop -- each press jogs one joint by {control.JOG_STEP_RAD} rad")
    print(f"{'joint':<16} {'+':<3} {'-':<3} limits (rad)")
    for index, name in enumerate(specs.JOINT_NAMES):
        lower, upper = specs.JOINT_LIMITS[name]
        print(f"{name:<16} {increase[index].upper():<3} {decrease[index].upper():<3} [{lower:+.3f}, {upper:+.3f}]")
    print("ESC, or closing the window, exits.\n")


def main() -> int:
    """Run the teleop loop. Returns the process exit code."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))
    scene = InteractiveScene(SoArmSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    device = robot.data.joint_pos.torch.device

    keyboard = JointJogKeyboard()
    print_key_help()
    deadline = None if args_cli.autoclose is None else time.monotonic() + args_cli.autoclose

    step = 0
    try:
        while simulation_app.is_running() and not keyboard.stop_requested:
            if deadline is not None and time.monotonic() >= deadline:
                break
            # Targets (radians) in the articulation's own joint order.
            target = torch.tensor([[keyboard.targets[name] for name in robot.joint_names]], device=device)
            robot.set_joint_position_target_index(target=target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            if step % PRINT_EVERY == 0:
                measured = robot.data.joint_pos.torch[0].tolist()
                line = "  ".join(f"{n}={v:+.4f}" for n, v in zip(robot.joint_names, measured))
                print(f"\r{line}", end="", flush=True)
            step += 1
    finally:
        keyboard.close()
        print()

    print(f"exited cleanly after {step} steps")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
