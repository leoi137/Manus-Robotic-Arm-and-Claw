"""Tests for the sim-free control helpers. All joint values are radians."""

import pytest

from manus.control import (
    DEMO_SEQUENCE,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    JOG_STEP_RAD,
    KEYMAP,
    PoseSequence,
    apply_jog,
    clamp_targets,
    pose,
)
from manus.specs import FEETECH_BUS_MAP, HOME_POSE, JOINT_LIMITS, JOINT_NAMES


@pytest.mark.parametrize("name", JOINT_NAMES)
def test_clamp_targets_clamps_both_limits(name):
    lower, upper = JOINT_LIMITS[name]
    assert clamp_targets({name: lower - 10.0}) == {name: lower}
    assert clamp_targets({name: upper + 10.0}) == {name: upper}


def test_clamp_targets_passes_through_in_range_value():
    assert clamp_targets({"elbow_flex": 0.5}) == {"elbow_flex": 0.5}


def test_clamp_targets_rejects_unknown_joint():
    with pytest.raises(ValueError, match="unknown joint"):
        clamp_targets({"elbow": 0.0})


def test_bus_map_matches_joint_order():
    # KEYMAP indexes JOINT_NAMES, so this order is the Feetech motor-ID contract.
    assert FEETECH_BUS_MAP == {
        1: "shoulder_pan",
        2: "shoulder_lift",
        3: "elbow_flex",
        4: "wrist_flex",
        5: "wrist_roll",
        6: "gripper",
    }
    assert tuple(FEETECH_BUS_MAP[i] for i in range(1, 7)) == JOINT_NAMES


# --- PoseSequence -----------------------------------------------------------

STEPS = 10
SEQUENCE = PoseSequence(
    [
        ("home", pose()),
        ("bend", pose(shoulder_lift=0.8, elbow_flex=-1.0)),
        ("twist", pose(shoulder_lift=0.8, elbow_flex=-1.0, wrist_roll=1.0)),
    ]
)


def test_sequence_rejects_incomplete_waypoint():
    with pytest.raises(ValueError, match="must set all"):
        PoseSequence([("partial", {"elbow_flex": 0.1})])


def test_sequence_total_steps():
    assert SEQUENCE.total_steps(STEPS) == 2 * STEPS


def test_sequence_starts_at_first_waypoint():
    assert SEQUENCE.at(0, STEPS) == SEQUENCE.waypoints[0][1]


def test_sequence_hits_intermediate_waypoint():
    assert SEQUENCE.at(STEPS, STEPS) == SEQUENCE.waypoints[1][1]


def test_sequence_ends_at_last_waypoint():
    last = SEQUENCE.waypoints[-1][1]
    assert SEQUENCE.at(SEQUENCE.total_steps(STEPS), STEPS) == last


def test_sequence_holds_final_pose_beyond_end():
    last = SEQUENCE.waypoints[-1][1]
    assert SEQUENCE.at(SEQUENCE.total_steps(STEPS) + 500, STEPS) == last


def test_sequence_interpolates_segment_midpoint():
    midpoint = SEQUENCE.at(STEPS // 2, STEPS)
    assert midpoint["shoulder_lift"] == pytest.approx(0.4)
    assert midpoint["elbow_flex"] == pytest.approx(-0.5)
    assert midpoint["wrist_roll"] == pytest.approx(0.0)


def test_sequence_interpolates_second_segment():
    midpoint = SEQUENCE.at(STEPS + STEPS // 2, STEPS)
    assert midpoint["wrist_roll"] == pytest.approx(0.5)
    assert midpoint["shoulder_lift"] == pytest.approx(0.8)


def test_sequence_rejects_bad_steps_per_segment():
    with pytest.raises(ValueError, match="steps_per_segment"):
        SEQUENCE.at(1, 0)


def test_demo_sequence_waypoints_within_limits():
    # Construction clamps, so also require clearance: a demo value authored
    # past a limit would be silently truncated onto it rather than caught.
    margin = 0.1
    for name, targets in DEMO_SEQUENCE.waypoints:
        assert set(targets) == set(JOINT_NAMES), name
        for joint, value in targets.items():
            lower, upper = JOINT_LIMITS[joint]
            assert lower + margin <= value <= upper - margin, (name, joint, value)


def test_demo_sequence_exercises_gripper_and_returns_home():
    labels = [name for name, _ in DEMO_SEQUENCE.waypoints]
    assert labels[0] == "home" and labels[-1] == "return_home"
    grips = {name: targets["gripper"] for name, targets in DEMO_SEQUENCE.waypoints}
    assert grips["gripper_open"] == GRIPPER_OPEN
    assert grips["gripper_close"] == GRIPPER_CLOSED
    assert DEMO_SEQUENCE.waypoints[-1][1] == HOME_POSE


# --- Teleop -----------------------------------------------------------------


def test_keymap_covers_every_joint_in_both_directions():
    assert len(KEYMAP) == 2 * len(JOINT_NAMES)  # a duplicate key would shrink this
    for index in range(len(JOINT_NAMES)):
        directions = {d for i, d in KEYMAP.values() if i == index}
        assert directions == {1.0, -1.0}


def test_apply_jog_steps_the_mapped_joint():
    jogged = apply_jog(dict(HOME_POSE), "e")
    assert jogged["elbow_flex"] == pytest.approx(JOG_STEP_RAD)
    assert apply_jog(dict(HOME_POSE), "d")["elbow_flex"] == pytest.approx(-JOG_STEP_RAD)


def test_apply_jog_accepts_uppercase_key():
    assert apply_jog(dict(HOME_POSE), "E")["elbow_flex"] == pytest.approx(JOG_STEP_RAD)


def test_apply_jog_ignores_unknown_key():
    current = dict(HOME_POSE)
    jogged = apply_jog(current, "z")
    assert jogged == current
    assert jogged is not current


@pytest.mark.parametrize("key,index,bound", [("q", 0, 1), ("a", 0, 0)])
def test_apply_jog_clamps_at_limits(key, index, bound):
    name = JOINT_NAMES[index]
    limit = JOINT_LIMITS[name][bound]
    at_limit = {**HOME_POSE, name: limit}
    assert apply_jog(at_limit, key)[name] == limit
