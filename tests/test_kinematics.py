"""Tests for the sim-free kinematics. Lengths metres, angles radians.

Four layers: the transcribed URDF constants are re-derived from the URDF
itself, the FK is pinned by hand-checkable golden geometry,
``test_fk_matches_isaac`` compares it against link poses dumped from the
simulator (skipped until ``scripts/dump_fk_fixture.py`` has been run), and the
IK is closed back on the FK by round-tripping the whole grasp region.
"""

import dataclasses
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from manus import kinematics, specs
from manus.kinematics import CHAIN_JOINTS, CHAIN_LINKS, TOOL_YAW_OFFSET, KinematicChain

CHAIN = KinematicChain()

ARM_LOWER = np.array([specs.JOINT_LIMITS[name][0] for name in kinematics.ARM_JOINT_NAMES])
ARM_UPPER = np.array([specs.JOINT_LIMITS[name][1] for name in kinematics.ARM_JOINT_NAMES])
"""Joint travel in ``q`` order, as arrays: what "in limits" means below."""

HOME_TCP = (0.3914, 0.0, 0.2265)
"""Golden TCP position at q = 0, from the panel-verified geometry."""

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fk_fixture.json"
"""Isaac ground truth written by scripts/dump_fk_fixture.py."""

MAX_POS_ERROR = 5e-4
"""Position agreement required against Isaac, in metres."""

MAX_ROT_ERROR_DEG = 0.1
"""Orientation agreement required against Isaac, in degrees."""


def _floats(text: str) -> tuple[float, ...]:
    """Parse a URDF whitespace-separated vector attribute."""
    return tuple(float(value) for value in text.split())


def _urdf_joints() -> dict[str, ET.Element]:
    """Typed ``<joint>`` elements of the vendored URDF, keyed by joint name.

    The file also carries untyped ``<joint name=...>`` stubs inside its
    ``<transmission>`` blocks; those are not kinematic, and dropping the
    type-less elements is what keeps them out.
    """
    root = ET.parse(kinematics.SO101_URDF_PATH).getroot()
    return {
        element.get("name"): element
        for element in root.iter("joint")
        if element.get("type") is not None
    }


# --- Transcription vs. URDF ---------------------------------------------------


@pytest.mark.parametrize("joint", CHAIN_JOINTS, ids=lambda joint: joint.name)
def test_transcribed_joint_matches_urdf(joint):
    element = _urdf_joints()[joint.name]
    origin = element.find("origin")
    assert _floats(origin.get("xyz")) == joint.xyz
    assert _floats(origin.get("rpy")) == joint.rpy
    assert element.find("parent").get("link") == joint.parent
    assert element.find("child").get("link") == joint.child
    axis = _floats(element.find("axis").get("xyz"))
    if joint.axis is None:
        assert element.get("type") == "fixed"
        assert axis == (0.0, 0.0, 0.0)
    else:
        assert element.get("type") == "revolute"
        assert axis == joint.axis


def test_every_urdf_revolute_joint_turns_about_local_z():
    # The whole FK rests on this: joint transform = origin then Rz(q).
    revolute = {
        name: element
        for name, element in _urdf_joints().items()
        if element.get("type") == "revolute"
    }
    assert set(revolute) == set(specs.JOINT_NAMES)
    for name, element in revolute.items():
        assert _floats(element.find("axis").get("xyz")) == kinematics.Z_AXIS, name


def test_joint_frame_rejects_a_non_z_axis():
    with pytest.raises(ValueError, match="assumes local"):
        kinematics.JointFrame(
            name="tilted",
            parent="a",
            child="b",
            xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, 0.0, 0.0),
            axis=(0.0, 1.0, 0.0),
        )


def test_chain_agrees_with_specs():
    assert kinematics.ARM_JOINT_NAMES == specs.JOINT_NAMES[: kinematics.NUM_ARM_JOINTS]
    assert CHAIN_LINKS == (*specs.LINK_CHAIN, specs.GRIPPER_FRAME_LINK)
    assert kinematics.TCP_LINK == specs.GRIPPER_FRAME_LINK


def test_chain_is_connected_base_outward():
    assert CHAIN_JOINTS[0].parent == kinematics.BASE_LINK
    for parent, child in zip(CHAIN_JOINTS, CHAIN_JOINTS[1:]):
        assert parent.child == child.parent


# --- Rotation helpers ---------------------------------------------------------


def test_rotation_from_identity_quat():
    assert kinematics.rotation_from_quat_xyzw([0.0, 0.0, 0.0, 1.0]) == pytest.approx(np.eye(3))


def test_rotation_from_quat_is_xyzw_not_wxyz():
    # 90 deg about +Z: reading this as (w, x, y, z) would give a Y rotation.
    half = np.sqrt(0.5)
    rotation = kinematics.rotation_from_quat_xyzw([0.0, 0.0, half, half])
    assert rotation @ [1.0, 0.0, 0.0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


def test_rotation_from_quat_rejects_zero_quat():
    with pytest.raises(ValueError, match="zero quaternion"):
        kinematics.rotation_from_quat_xyzw([0.0, 0.0, 0.0, 0.0])


def test_rotation_error_is_zero_for_equal_rotations():
    rotation = CHAIN.fk_tcp(np.zeros(kinematics.NUM_ARM_JOINTS))[1]
    assert kinematics.rotation_error_deg(rotation, rotation) == pytest.approx(0.0, abs=1e-9)


def test_rotation_error_stays_linear_near_zero():
    # The arccos-of-the-trace form would report ~1e-3 deg for this, swamping
    # the small deviations the Isaac fixture comparison has to resolve.
    tiny = np.deg2rad(1e-4)
    nudged = kinematics.rotation_from_quat_xyzw([0.0, 0.0, np.sin(tiny / 2), np.cos(tiny / 2)])
    assert kinematics.rotation_error_deg(np.eye(3), nudged) == pytest.approx(1e-4, rel=1e-6)


def test_rotation_error_measures_a_known_angle():
    half = np.sqrt(0.5)
    quarter_turn = kinematics.rotation_from_quat_xyzw([0.0, 0.0, half, half])
    assert kinematics.rotation_error_deg(np.eye(3), quarter_turn) == pytest.approx(90.0)


# --- Forward kinematics -------------------------------------------------------


def test_fk_reports_every_chain_link():
    poses = CHAIN.fk(np.zeros(kinematics.NUM_ARM_JOINTS))
    assert tuple(poses) == CHAIN_LINKS


def test_fk_base_link_is_the_identity_pose():
    position, rotation = CHAIN.fk(np.zeros(kinematics.NUM_ARM_JOINTS))[kinematics.BASE_LINK]
    assert position == pytest.approx(np.zeros(3))
    assert rotation == pytest.approx(np.eye(3))


def test_fk_returns_proper_rotations():
    q = np.array([0.4, -0.3, 0.7, -0.2, 1.1])
    for link, (position, rotation) in CHAIN.fk(q).items():
        assert position.shape == (3,), link
        assert rotation.shape == (3, 3), link
        assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12), link
        assert np.linalg.det(rotation) == pytest.approx(1.0), link


def test_fk_rejects_a_wrong_length_q():
    with pytest.raises(ValueError, match="shape"):
        CHAIN.fk(np.zeros(len(specs.JOINT_NAMES)))


def test_fk_tcp_at_home_matches_golden():
    position, _ = CHAIN.fk_tcp(np.zeros(kinematics.NUM_ARM_JOINTS))
    assert position == pytest.approx(HOME_TCP, abs=1e-3)


def test_fk_tcp_is_the_gripper_frame_link_pose():
    q = np.array([0.2, 0.5, -0.6, 0.3, -0.4])
    position, rotation = CHAIN.fk_tcp(q)
    expected_position, expected_rotation = CHAIN.fk(q)[kinematics.TCP_LINK]
    assert position == pytest.approx(expected_position)
    assert rotation == pytest.approx(expected_rotation)


def test_positive_shoulder_pan_swings_the_tcp_to_negative_y():
    # The pan axis points along world -Z, so the sign is worth pinning down.
    position, _ = CHAIN.fk_tcp(np.array([0.3, 0.0, 0.0, 0.0, 0.0]))
    assert position[1] < -0.1


# --- Tool-vertical geometry ---------------------------------------------------


def _vertical_configs(count: int = 200, seed: int = 0) -> list[np.ndarray]:
    """Seeded in-limit configurations whose pitch sum aims the tool down."""
    rng = np.random.default_rng(seed)
    configs: list[np.ndarray] = []
    while len(configs) < count:
        q = rng.uniform(ARM_LOWER, ARM_UPPER)
        q[3] = kinematics.wrist_flex_for_vertical(q[1], q[2])
        if ARM_LOWER[3] <= q[3] <= ARM_UPPER[3]:
            configs.append(q)
    return configs


VERTICAL_CONFIGS = _vertical_configs()


def _tilt_from_down_deg(q: np.ndarray) -> float:
    """Angle between the tool approach axis (TCP +Z) and world -Z, in degrees."""
    _, rotation = CHAIN.fk_tcp(q)
    return float(np.degrees(np.arccos(np.clip(-rotation[2, 2], -1.0, 1.0))))


def test_pitch_sum_reads_the_three_pitch_joints():
    assert kinematics.pitch_sum(np.array([9.0, 0.1, 0.2, 0.3, 9.0])) == pytest.approx(0.6)


def test_wrist_flex_for_vertical_completes_the_pitch_sum():
    wrist_flex = kinematics.wrist_flex_for_vertical(0.4, -0.9)
    q = np.array([0.0, 0.4, -0.9, wrist_flex, 0.0])
    assert kinematics.pitch_sum(q) == pytest.approx(kinematics.PITCH_SUM_VERTICAL)


def test_tool_approach_axis_is_plus_z_and_points_down_at_vertical():
    # Documents the convention the whole grasp stack leans on: the approach
    # direction is the TCP's +Z, and PITCH_SUM_VERTICAL aims it at world -Z.
    q = np.array([0.0, np.pi / 6, np.pi / 6, np.pi / 6, 0.0])
    assert kinematics.pitch_sum(q) == pytest.approx(kinematics.PITCH_SUM_VERTICAL)
    _, rotation = CHAIN.fk_tcp(q)
    assert rotation[:, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-4)


def test_pitch_sum_constraint_makes_the_tool_vertical():
    worst = max(VERTICAL_CONFIGS, key=_tilt_from_down_deg)
    assert _tilt_from_down_deg(worst) < 0.01, f"tool not vertical at q={worst}"


def test_tool_yaw_relation_holds_at_vertical_configs():
    def error_deg(q: np.ndarray) -> float:
        predicted = TOOL_YAW_OFFSET - q[0] + q[4]
        return abs(np.degrees((CHAIN.tool_yaw(q) - predicted + np.pi) % (2 * np.pi) - np.pi))

    worst = max(VERTICAL_CONFIGS, key=error_deg)
    assert error_deg(worst) < 0.1, f"yaw relation broken at q={worst}"


# --- Isaac ground truth -------------------------------------------------------


def test_fk_matches_isaac():
    """Every dumped Isaac link pose is reproduced by the numpy chain."""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"no fixture yet: run scripts/dump_fk_fixture.py --headless ({FIXTURE_PATH})")
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert fixture["pose_frame"] == kinematics.BASE_LINK
    configs = fixture["configs"]
    assert len(configs) == fixture["num_configs"]
    arm_columns = [fixture["joint_names"].index(name) for name in kinematics.ARM_JOINT_NAMES]

    worst_position = (0.0, "")
    worst_rotation = (0.0, "")
    for index, config in enumerate(configs):
        q = np.asarray(config["joint_pos"], dtype=float)[arm_columns]
        poses = CHAIN.fk(q)
        for link in CHAIN_LINKS:
            assert link in config["bodies"], f"config {index}: {link} missing from the fixture"
            body = config["bodies"][link]
            position, rotation = poses[link]
            label = f"config {index} ({link}) q={q}"
            offset = float(np.linalg.norm(position - np.asarray(body["pos"], dtype=float)))
            angle = kinematics.rotation_error_deg(rotation, np.asarray(body["rot"], dtype=float))
            worst_position = max(worst_position, (offset, label))
            worst_rotation = max(worst_rotation, (angle, label))

    assert worst_position[0] < MAX_POS_ERROR, f"{worst_position[0]:.6f} m at {worst_position[1]}"
    assert worst_rotation[0] < MAX_ROT_ERROR_DEG, f"{worst_rotation[0]:.4f} deg at {worst_rotation[1]}"


# --- The grasp region ---------------------------------------------------------


def _tcp_roll_swing_radius() -> float:
    """Radius the TCP sweeps about the wrist_roll axis, in metres.

    Traced with FK over a *full* turn of wrist_roll -- past the joint's own
    320 deg of travel, so the number is a property of the geometry rather than
    of the limits. It is the whole position/yaw coupling the IK has to resolve.
    """
    rolls = np.linspace(-np.pi, np.pi, 721)
    xy = np.array(
        [
            CHAIN.fk_tcp(np.array([0.0, 0.0, 0.0, kinematics.PITCH_SUM_VERTICAL, roll]))[0][:2]
            for roll in rolls
        ]
    )
    return float(np.ptp(xy, axis=0).max()) / 2.0


TCP_ROLL_SWING = _tcp_roll_swing_radius()
"""Measured once: the 7.9 mm the region's azimuth cap is derived from."""


def test_region_aliases_are_the_dataclass():
    # manus.randomize re-exports these; they must not drift into a second
    # definition of the workspace.
    region = kinematics.GRASP_REGION
    assert kinematics.PAN_AXIS_XY == region.pan_axis_xy
    assert kinematics.REGION_R == region.radius
    assert kinematics.REGION_AZ_DEG == region.azimuth_max_deg
    assert kinematics.BASE_KEEPOUT_X == region.keepout_x
    assert kinematics.BASE_KEEPOUT_ABS_Y == region.keepout_abs_y
    assert kinematics.in_grasp_region == region.in_annulus
    assert kinematics.in_base_keepout == region.in_keepout


def test_region_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        kinematics.GRASP_REGION.azimuth_max_deg = 180.0


def test_region_polar_is_measured_about_the_pan_axis():
    pan_x, pan_y = kinematics.PAN_AXIS_XY
    radius, azimuth = kinematics.GRASP_REGION.polar(pan_x + 0.2, pan_y)
    assert (radius, azimuth) == pytest.approx((0.2, 0.0))


def test_region_membership_needs_the_annulus_and_the_keepout():
    # The keep-out genuinely bites inside the annulus rather than merely
    # trimming a corner of it, which is why `contains` has to check both.
    pan_x, pan_y = kinematics.PAN_AXIS_XY
    angle = np.radians(60.0)
    x = pan_x + 0.112 * np.cos(angle)
    y = pan_y + 0.112 * np.sin(angle)
    assert kinematics.GRASP_REGION.in_annulus(x, y)
    assert kinematics.GRASP_REGION.in_keepout(x, y)
    assert not kinematics.GRASP_REGION.contains(x, y)


def test_tcp_hangs_the_panel_verified_7_9_mm_off_the_wrist_roll_axis():
    assert TCP_ROLL_SWING == pytest.approx(0.0079, abs=1e-5)


def test_region_azimuth_leaves_pan_travel_for_that_swing():
    """105 deg is not a round number -- it is what shoulder_pan has left over.

    Reaching an (x, y) at azimuth `a` costs up to ``asin(swing / r)`` of pan
    beyond `a`, worst at the inner radius, because the TCP has to be swung onto
    the point rather than the wrist. Step 4 measured the untrimmed 110 deg edge
    losing ~1.4% of the region to that, all of it beyond |azimuth| 107.3.
    """
    pan_travel_deg = np.degrees(specs.JOINT_LIMITS["shoulder_pan"][1])
    worst_extra_deg = np.degrees(np.arcsin(TCP_ROLL_SWING / kinematics.REGION_R[0]))
    headroom = pan_travel_deg - worst_extra_deg
    assert kinematics.REGION_AZ_DEG <= headroom, "the region asks for pan the arm has not got"
    # ... and the shave is a shave, not a retreat: within a degree of the edge.
    assert kinematics.REGION_AZ_DEG > headroom - 2.0, "more workspace given up than needed"


def test_tcp_to_pad_centre_lifts_the_tcp_clear_of_the_pads():
    """PLACEHOLDER-TUNED (Step 7 re-tunes it against grasps that actually hold).

    Bounded rather than pinned: the geometric first guess is the 7.3 mm from the
    gripper_frame origin down to the jaw tips, with the pads' centre about at tip
    level. Step 7 may move it a few mm; a sign flip or a centimetre is a bug.
    """
    assert 0.0 < kinematics.TCP_TO_PAD_CENTRE < 0.02


# --- Inverse kinematics -------------------------------------------------------

GRASP_HEIGHTS = (0.015, 0.030)
"""TCP heights the grasp itself happens at, metres (short and tall objects)."""

HOVER_HEIGHT = 0.045
"""TCP height of the pregrasp waypoint, metres: a grasp height plus 3 cm."""

TARGETS_PER_HEIGHT = 500
"""Round-trip samples per height. 1500 solves run in well under a second."""

MAX_IK_POS_ERROR = 1e-3
"""Position agreement required of a converged solve, in metres."""

MAX_IK_TILT_DEG = 0.5
"""Tool tilt off vertical tolerated in a converged solve, in degrees."""

MAX_IK_YAW_DEG = 1.0
"""Grasp-yaw error tolerated in a converged solve, in degrees."""

UNREACHABLE_TARGETS = (
    np.array([0.60, 0.00, 0.030]),  # past the arm's reach entirely
    np.array([0.00, 0.00, 0.500]),  # straight up, far above the vertical ceiling
    np.array([0.05, 0.00, 0.020]),  # inside the base, under the shoulder
)
"""Requests no in-limit vertical-tool pose answers; the solver must say so."""


def _region_targets(
    height: float, seed: int, count: int = TARGETS_PER_HEIGHT
) -> list[tuple[np.ndarray, float]]:
    """Seeded ``(target_pos, target_yaw)`` grasp requests over the whole region.

    Uniform by area: the radius is drawn as ``sqrt(U(r_min^2, r_max^2))``, the
    same draw ``manus.randomize`` places objects with, so the sample is not
    piled up against the crowded inner edge. Yaws span a half turn, which is
    every distinct parallel-jaw grasp (see :func:`grasp_yaw_error`).
    """
    region = kinematics.GRASP_REGION
    inner, outer = region.radius
    azimuth_max = np.radians(region.azimuth_max_deg)
    pan_x, pan_y = region.pan_axis_xy
    rng = np.random.default_rng(seed)
    targets: list[tuple[np.ndarray, float]] = []
    while len(targets) < count:
        radius = np.sqrt(rng.uniform(inner**2, outer**2))
        azimuth = rng.uniform(-azimuth_max, azimuth_max)
        x = pan_x + radius * np.cos(azimuth)
        y = pan_y + radius * np.sin(azimuth)
        if region.in_keepout(x, y):
            continue
        targets.append((np.array([x, y, height]), float(rng.uniform(-np.pi / 2, np.pi / 2))))
    return targets


REGION_TARGETS = tuple(
    request
    for index, height in enumerate((*GRASP_HEIGHTS, HOVER_HEIGHT))
    for request in _region_targets(height, seed=index)
)
"""The round-trip sample: every height, the whole region, arbitrary grasp yaw."""

SOLUTIONS = tuple(
    (target, yaw, *kinematics.ik_solve(target, yaw)) for target, yaw in REGION_TARGETS
)
"""``(target, yaw, q, converged)`` per request, solved once and shared."""


def test_ik_round_trips_the_grasp_region():
    """Every request answered, and every answer verified by running FK back."""
    misses = [(target, yaw) for target, yaw, _, converged in SOLUTIONS if not converged]
    rate = 1.0 - len(misses) / len(SOLUTIONS)
    assert rate >= 0.99, f"{rate:.2%} of {len(SOLUTIONS)} solved; e.g. {misses[:3]}"

    worst = {"pos": 0.0, "tilt": 0.0, "yaw": 0.0, "pitch": 0.0}
    for target, yaw, q, converged in SOLUTIONS:
        if not converged:
            continue
        position_error, tilt, yaw_error = kinematics.ik_errors(q, target, yaw)
        worst["pos"] = max(worst["pos"], position_error)
        worst["tilt"] = max(worst["tilt"], np.degrees(tilt))
        worst["yaw"] = max(worst["yaw"], np.degrees(yaw_error))
        worst["pitch"] = max(
            worst["pitch"],
            abs(np.degrees(kinematics.pitch_sum(q) - kinematics.PITCH_SUM_VERTICAL)),
        )
    assert worst["pos"] < MAX_IK_POS_ERROR, f"worst position error {worst['pos'] * 1e3:.4f} mm"
    assert worst["tilt"] < MAX_IK_TILT_DEG, f"worst tool tilt {worst['tilt']:.4f} deg"
    assert worst["yaw"] < MAX_IK_YAW_DEG, f"worst grasp yaw error {worst['yaw']:.4f} deg"
    # The solver drives the pitch sum and only *claims* that aims the tool down.
    # Asserting the tool-axis tilt above and its proxy here is what ties the two
    # together -- a chain edit that broke the equivalence would pass only one.
    assert worst["pitch"] < MAX_IK_TILT_DEG, f"worst pitch-sum error {worst['pitch']:.4f} deg"


def test_ik_solves_the_whole_boundary_of_the_region():
    """The ">=99% of the region" bar above cannot see this, and the edge is
    where the region actually breaks.

    Sampling the interior uniformly dilutes an infeasible edge into noise. The
    region fails *only* past its azimuth cap, and widening the cap to 107 or 108
    deg -- both genuinely unreachable out there -- still scores 100% on the bar
    above; even the untrimmed 110 deg version scores 98.6% and squeaks under a
    99% threshold, leaving the expert to find the dead corner one dropped cube at
    a time. So the four edges are swept deterministically and held to 100%: this
    test fails at 107, the rate test does not.
    """
    region = kinematics.GRASP_REGION
    pan_x, pan_y = region.pan_axis_xy
    radii = np.linspace(*region.radius, 12)
    azimuths = np.radians(np.linspace(-region.azimuth_max_deg, region.azimuth_max_deg, 12))
    edge = [(radius, azimuths[0]) for radius in radii]
    edge += [(radius, azimuths[-1]) for radius in radii]
    edge += [(region.radius[0], azimuth) for azimuth in azimuths]
    edge += [(region.radius[1], azimuth) for azimuth in azimuths]

    for radius, azimuth in edge:
        x = pan_x + radius * np.cos(azimuth)
        y = pan_y + radius * np.sin(azimuth)
        if region.in_keepout(x, y):
            continue
        for height in (*GRASP_HEIGHTS, HOVER_HEIGHT):
            target = np.array([x, y, height])
            for yaw in np.linspace(-np.pi / 2, np.pi / 2, 9):
                q, converged = kinematics.ik_solve(target, yaw)
                assert converged, (
                    f"region edge unreachable: r={radius:.4f} "
                    f"az={np.degrees(azimuth):+.2f} deg z={height} "
                    f"yaw={np.degrees(yaw):+.1f} deg -> q={np.round(q, 4)}"
                )


def test_ik_returns_joints_the_arm_can_actually_hold():
    # Misses included: an unreachable request comes back clamped, not overrun.
    for target, _, q, _ in SOLUTIONS:
        assert np.all(q >= ARM_LOWER) and np.all(q <= ARM_UPPER), f"{q} at {target}"


def test_ik_reports_an_unreachable_target_rather_than_faking_one():
    for target in UNREACHABLE_TARGETS:
        q, converged = kinematics.ik_solve(target, 0.0)
        assert not converged, f"claimed to reach {target} with q={q}"
        assert np.all(q >= ARM_LOWER) and np.all(q <= ARM_UPPER), f"{q} at {target}"


def test_ik_converges_within_fifteen_iterations():
    """Seed quality: the refinement must not be doing the seed's job for it."""
    quick = sum(
        kinematics.ik_solve(target, yaw, max_iters=15)[1] for target, yaw in REGION_TARGETS
    )
    rate = quick / len(REGION_TARGETS)
    assert rate >= 0.95, f"only {rate:.2%} converged within 15 iterations"


def test_the_analytic_seed_is_already_the_answer():
    """Stronger than the 15-iteration bar, and the reason it is met so easily.

    The decomposition is exact -- pan, then the planar 2R, then the wrist -- once
    the fixed-point passes have settled the TCP offset, so over the region the
    seed lands two orders of magnitude inside the position tolerance (worst
    24 um over 58k samples) and the refinement runs zero iterations. Held to
    0.1 mm here so a seed regression fires *here*, loudly, instead of quietly
    eating the solver's whole error budget.
    """
    worst = max(
        kinematics.ik_errors(kinematics.analytic_seed(target, yaw)[0], target, yaw)[0]
        for target, yaw in REGION_TARGETS
    )
    assert worst < 1e-4, f"seed off by {worst * 1e3:.4f} mm; the fixed point is not settling"


def test_grasp_yaw_moves_shoulder_pan_by_exactly_the_tcp_swing():
    """The coupling the seed's fixed point exists to resolve, measured.

    The TCP hangs :data:`TCP_ROLL_SWING` off the wrist_roll axis, so the pan that
    puts it on a given (x, y) depends on which way the jaws face -- and over a
    half turn of grasp yaw the swing sweeps a half turn, spreading pan by twice
    its angular size at that reach.
    """
    target = np.array([0.25, 0.0, 0.030])
    radius, _ = kinematics.GRASP_REGION.polar(target[0], target[1])
    yaws = np.linspace(-np.pi / 2, np.pi / 2, 37)
    pans = [kinematics.ik_solve(target, yaw)[0][0] for yaw in yaws]
    spread = max(pans) - min(pans)
    assert spread == pytest.approx(2.0 * np.arcsin(TCP_ROLL_SWING / radius), rel=0.01)


def test_the_pi_flipped_grasp_covers_the_wrist_roll_gap():
    """wrist_roll travels 320 deg, so 40 deg of tool yaws have no direct
    representative at a given pan. Parallel jaws make the pi-flip the same
    physical grasp, and that is what the solver answers with instead."""
    target = np.array([0.25, 0.0, 0.030])
    flipped = 0
    for yaw in np.linspace(-np.pi / 2, np.pi / 2, 181):
        q, converged = kinematics.ik_solve(target, yaw)
        assert converged, f"no solution at grasp yaw {np.degrees(yaw):.1f} deg"
        actual = CHAIN.tool_yaw(q)
        assert abs(kinematics.grasp_yaw_error(actual, yaw)) < np.radians(MAX_IK_YAW_DEG)
        flipped += abs(np.degrees((actual - yaw + np.pi) % (2 * np.pi) - np.pi)) > 90.0
    assert flipped, "wrist_roll reached every yaw head-on; the travel gap has moved"


def test_grasp_yaw_error_treats_a_half_turn_as_the_same_grasp():
    assert kinematics.grasp_yaw_error(0.3, 0.3) == pytest.approx(0.0)
    assert kinematics.grasp_yaw_error(0.3 + np.pi, 0.3) == pytest.approx(0.0)
    assert kinematics.grasp_yaw_error(0.4, 0.3) == pytest.approx(0.1)
    assert kinematics.grasp_yaw_error(0.2, 0.3) == pytest.approx(-0.1)


def test_ik_errors_reads_the_tool_axis_rather_than_the_solver_proxy():
    """`ik_errors` measures tilt off the tool's own +Z, and it still agrees with
    the pitch-sum residual the solver actually drives -- which is what licenses
    the solver to drive the proxy. Checked well off vertical, where the two
    would part company if the pitch joints ever stopped sharing an axis."""
    q = np.array([0.3, 0.2, -0.1, 0.4, -0.5])  # 61 deg from vertical
    target = CHAIN.fk_tcp(q)[0] + np.array([0.01, 0.0, 0.0])
    position_error, tilt, yaw_error = kinematics.ik_errors(q, target, 0.0)
    assert position_error == pytest.approx(0.01)
    assert np.degrees(tilt) == pytest.approx(_tilt_from_down_deg(q))
    assert yaw_error == pytest.approx(abs(kinematics.grasp_yaw_error(CHAIN.tool_yaw(q), 0.0)))
    proxy = abs(kinematics.pitch_sum(q) - kinematics.PITCH_SUM_VERTICAL)
    assert tilt == pytest.approx(proxy, abs=np.radians(0.01))


def test_ik_is_deterministic():
    target, yaw = REGION_TARGETS[0]
    hint = np.full(kinematics.NUM_ARM_JOINTS, 0.1)
    assert np.array_equal(*(kinematics.ik_solve(target, yaw)[0] for _ in range(2)))
    assert np.array_equal(*(kinematics.ik_solve(target, yaw, q_seed=hint)[0] for _ in range(2)))


def test_a_warm_start_is_never_worse_than_no_seed():
    """A hint may not cost the caller a solution.

    Regression test: deriving the yaw branch from the *hint's* pan instead of
    the target's lost ~4% of warm-started solves -- every one of them with
    wrist_roll pinned at its limit, having been sent after the branch the target
    could not reach.
    """
    rng = np.random.default_rng(11)
    for target, yaw, cold, converged in SOLUTIONS[::13]:
        if not converged:
            continue  # the round-trip test owns the miss rate; this one is about hints
        for sigma in (0.05, 0.5):
            hint = np.clip(
                cold + rng.normal(0.0, sigma, kinematics.NUM_ARM_JOINTS), ARM_LOWER, ARM_UPPER
            )
            q, warm_converged = kinematics.ik_solve(target, yaw, q_seed=hint)
            assert warm_converged, f"hint at sigma {sigma} lost the solution at {target}"
            assert kinematics.ik_errors(q, target, yaw)[0] < MAX_IK_POS_ERROR


def test_a_useless_warm_start_still_answers_the_same_grasp():
    # Seeded from the home pose, which is nowhere near any of these targets.
    # Joint space may differ by the wrist_roll half-turn the jaws cannot tell
    # apart, so the tool poses are compared rather than the q vectors.
    home = np.zeros(kinematics.NUM_ARM_JOINTS)
    for target, yaw, cold, converged in SOLUTIONS[::13]:
        if not converged:
            continue
        q, warm_converged = kinematics.ik_solve(target, yaw, q_seed=home)
        assert warm_converged, f"home hint lost the solution at {target}"
        assert CHAIN.fk_tcp(q)[0] == pytest.approx(CHAIN.fk_tcp(cold)[0], abs=2 * MAX_IK_POS_ERROR)
        assert abs(kinematics.grasp_yaw_error(CHAIN.tool_yaw(q), yaw)) < np.radians(MAX_IK_YAW_DEG)


def test_ik_rejects_malformed_arguments():
    with pytest.raises(ValueError, match="target_pos must have shape"):
        kinematics.ik_solve(np.zeros(2), 0.0)
    with pytest.raises(ValueError, match="q_seed must have shape"):
        kinematics.ik_solve(np.array([0.25, 0.0, 0.03]), 0.0, q_seed=np.zeros(6))
