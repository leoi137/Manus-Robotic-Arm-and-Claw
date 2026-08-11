"""Sim-free joint-space control helpers for the SO-ARM101 arm.

Pure Python (no Isaac Sim, no numpy) so it is unit-testable and reusable on
the real arm. All joint targets are absolute positions in radians.
"""

from collections.abc import Iterable

from manus.specs import HOME_POSE, JOINT_LIMITS, JOINT_NAMES


def clamp_targets(targets: dict[str, float]) -> dict[str, float]:
    """Clamp each named target (radians) into its JOINT_LIMITS range.

    Raises ValueError on an unknown joint name.
    """
    clamped = {}
    for name, value in targets.items():
        if name not in JOINT_LIMITS:
            raise ValueError(f"unknown joint {name!r}; expected one of {JOINT_NAMES}")
        lower, upper = JOINT_LIMITS[name]
        clamped[name] = min(max(value, lower), upper)
    return clamped


def pose(**joints: float) -> dict[str, float]:
    """Full six-joint pose (radians): HOME_POSE overridden by `joints`, clamped."""
    return clamp_targets({**HOME_POSE, **joints})


def _checked_steps(steps_per_segment: int) -> int:
    if steps_per_segment < 1:
        raise ValueError("steps_per_segment must be >= 1")
    return steps_per_segment


class PoseSequence:
    """Ordered, named waypoints with linear interpolation between them.

    Each waypoint is a ``(name, targets)`` pair whose targets (radians) cover
    every joint in JOINT_NAMES; they are clamped to JOINT_LIMITS on construction.
    """

    def __init__(self, waypoints: Iterable[tuple[str, dict[str, float]]]) -> None:
        self.waypoints: list[tuple[str, dict[str, float]]] = []
        for name, targets in waypoints:
            checked = clamp_targets(targets)
            if len(checked) != len(JOINT_NAMES):
                raise ValueError(
                    f"waypoint {name!r} must set all {len(JOINT_NAMES)} joints"
                )
            self.waypoints.append((name, checked))
        if not self.waypoints:
            raise ValueError("a PoseSequence needs at least one waypoint")

    def total_steps(self, steps_per_segment: int) -> int:
        """Steps needed to walk the whole sequence once."""
        return (len(self.waypoints) - 1) * _checked_steps(steps_per_segment)

    def at(self, step: int, steps_per_segment: int) -> dict[str, float]:
        """Targets (radians) at `step`, holding the final pose past the end."""
        _checked_steps(steps_per_segment)
        if step <= 0:
            return dict(self.waypoints[0][1])
        segment, offset = divmod(step, steps_per_segment)
        if segment >= len(self.waypoints) - 1:
            return dict(self.waypoints[-1][1])
        start = self.waypoints[segment][1]
        end = self.waypoints[segment + 1][1]
        fraction = offset / steps_per_segment
        return {
            name: start[name] + fraction * (end[name] - start[name]) for name in start
        }


# Gripper jaw targets (radians): open is 86% of the joint's upper limit.
GRIPPER_OPEN = 1.5
GRIPPER_CLOSED = 0.0

# Arm poses reused across the demo waypoints; signs verified by forward
# kinematics against the vendored URDF (reach extends the tool frame from
# 0.39 m to 0.46 m horizontal radius, and no waypoint dips below the ground).
_REACH = {"shoulder_lift": 0.9, "elbow_flex": -1.1, "wrist_flex": -0.3}
_WRIST = {**_REACH, "wrist_flex": 0.6, "wrist_roll": 1.2}

#: Scripted demo: reach out, articulate the wrist, work the gripper, go home.
DEMO_SEQUENCE = PoseSequence(
    [
        ("home", pose()),
        ("reach_forward", pose(**_REACH)),
        ("wrist_articulate", pose(**_WRIST)),
        ("gripper_open", pose(**_WRIST, gripper=GRIPPER_OPEN)),
        ("gripper_close", pose(**_WRIST, gripper=GRIPPER_CLOSED)),
        ("return_home", pose()),
    ]
)

# Teleop: keyboard key -> (index into JOINT_NAMES, direction).
KEYMAP: dict[str, tuple[int, float]] = {
    "q": (0, 1.0), "a": (0, -1.0),  # shoulder_pan
    "w": (1, 1.0), "s": (1, -1.0),  # shoulder_lift
    "e": (2, 1.0), "d": (2, -1.0),  # elbow_flex
    "r": (3, 1.0), "f": (3, -1.0),  # wrist_flex
    "t": (4, 1.0), "g": (4, -1.0),  # wrist_roll
    "y": (5, 1.0), "h": (5, -1.0),  # gripper
}

#: Radians added to a joint target per key press.
JOG_STEP_RAD = 0.05


def apply_jog(current: dict[str, float], key: str) -> dict[str, float]:
    """Jog one joint by JOG_STEP_RAD (radians) and clamp; unknown key: copy.

    Key lookup is case-insensitive, since input devices report e.g. "Q".
    """
    entry = KEYMAP.get(key.lower())
    if entry is None:
        return dict(current)
    index, direction = entry
    name = JOINT_NAMES[index]
    jogged = dict(current)
    jogged[name] = current.get(name, 0.0) + direction * JOG_STEP_RAD
    return clamp_targets(jogged)
