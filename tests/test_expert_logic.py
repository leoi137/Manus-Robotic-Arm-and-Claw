"""Tests for the scripted grasp expert's sim-free logic. Radians and metres.

Four layers, none of which need Isaac: the *plan* is closed back on the FK
(waypoints put the TCP where they claim, the retraction really raises it), the
*yaw branch* is checked against the object symmetry it is derived from, the
*FSM* is driven by fake plants whose behaviour is known exactly (a perfect
servo, a wedged joint, a drooping servo, a jaw that hits an object), and the
*success predicate* is exercised on synthetic height traces.

The fake plants matter more than they look: the real gate the expert has to
pass is "advance on convergence, not on a step count", and the only way to test
that without a simulator is to hand it a plant that converges and one that
never does, and assert the two produce different exits.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from manus import expert as expert_mod
from manus import kinematics, objects, specs
from manus.control import GRIPPER_OPEN
from manus.expert import (
    ARM_STATES,
    CLOSE,
    DESCEND,
    DONE,
    HOLD,
    LIFT,
    PREGRASP,
    STATE_SEQUENCE,
    ExpertConfig,
    GraspSuccessMonitor,
    ScriptedGraspExpert,
    classify_outcome,
    grasp_yaw_candidates,
    joint_vector,
    plan_grasp,
    plan_lift,
    yaw_symmetry,
)
from manus.kinematics import GRASP_REGION, TCP_TO_PAD_CENTRE, KinematicChain
from manus.objects import OBJECTS, ObjectSpec
from manus.randomize import draw_episode

CHAIN = KinematicChain()
CUBE = OBJECTS["cube_3cm"]
CYLINDER = OBJECTS["cylinder_3cm"]
DOMINO = OBJECTS["domino_20x40"]  # the rectangular case: two branches, not four
PUCK = OBJECTS["puck_d40x10"]  # the widest grasp and the only raised one
BALL = OBJECTS["pingpong_40mm"]  # 2.7 g, round in every direction
DUPLO = OBJECTS["duplo_32x64"]  # the other rectangular case, 32 mm across

ARM = kinematics.NUM_ARM_JOINTS
HOME = np.zeros(ARM)

MIN_LIFT_RISE = 0.06
"""TCP rise the LIFT retraction has to deliver, metres (the plan's bar)."""

MOVING_JAW_DEEPEST_Z = 0.00806
"""Deepest the *moving* jaw reaches below the TCP over the closing sweep, metres.

Measured off the meshes by :func:`test_the_closing_jaw_reaches_deeper_than_the_tips`
below, which is also where the number comes from. It peaks at 0.14 rad, 1.8 mm
past the static fingertips: the finger swings down as it closes and back up as
it goes past. Only the short objects care, and only during CLOSE -- the arm is
frozen by then, so this is a clearance to keep, not a motion to plan.
"""


def region_samples(n_radius: int = 5, n_azimuth: int = 9) -> list[tuple[float, float]]:
    """A coarse grid of legal placements spanning the whole grasp region."""
    r_min, r_max = GRASP_REGION.radius
    points = []
    for radius in np.linspace(r_min, r_max, n_radius):
        for azimuth in np.radians(
            np.linspace(-GRASP_REGION.azimuth_max_deg, GRASP_REGION.azimuth_max_deg, n_azimuth)
        ):
            x = GRASP_REGION.pan_axis_xy[0] + radius * math.cos(azimuth)
            y = GRASP_REGION.pan_axis_xy[1] + radius * math.sin(azimuth)
            if not GRASP_REGION.in_keepout(x, y):
                points.append((float(x), float(y)))
    return points


# --- Fake plants ---------------------------------------------------------------


class FakeArm:
    """A first-order servo with a constant steady-state droop, plus a jaw.

    ``measured <- measured + follow * (command - droop - measured)``, so the
    arm settles at ``command - droop`` -- which is exactly how a P controller
    with no gravity feed-forward behaves, and is the thing the expert's droop
    integrator exists to cancel. The jaw follows the same law but stops at
    :attr:`jaw_stop`, standing in for an object between the pads.

    Attributes:
        q: Current six-joint measurement, radians.
    """

    def __init__(
        self,
        *,
        follow: float = 0.35,
        droop: float | np.ndarray = 0.0,
        jaw_stop: float | None = None,
        frozen: bool = False,
    ) -> None:
        self.q = np.zeros(len(specs.JOINT_NAMES))
        self.follow = follow
        self.droop = np.broadcast_to(np.asarray(droop, dtype=float), (ARM,)).copy()
        self.jaw_stop = jaw_stop
        self.frozen = frozen

    def apply(self, targets: dict[str, float]) -> np.ndarray:
        """Consume one command dict and return the resulting measurement."""
        command = np.array([targets[name] for name in specs.JOINT_NAMES])
        if not self.frozen:
            self.q[:ARM] += self.follow * (command[:ARM] - self.droop - self.q[:ARM])
            jaw = self.q[ARM] + self.follow * (command[ARM] - self.q[ARM])
            self.q[ARM] = jaw if self.jaw_stop is None else max(jaw, self.jaw_stop)
        return self.q.copy()


def run(
    expert: ScriptedGraspExpert, plant: FakeArm, max_steps: int = 4000
) -> list[tuple[str, np.ndarray]]:
    """Drive `expert` against `plant` until DONE; returns (state, measured) per step."""
    trace = []
    measured = plant.q.copy()
    for _ in range(max_steps):
        if expert.done:
            break
        state = expert.state
        targets = expert.step(measured)
        measured = plant.apply(targets)
        trace.append((state, measured.copy()))
    assert expert.done, f"expert never finished; stuck in {expert.state}"
    return trace


def fresh(placement=(0.20, 0.0, 0.0), **overrides) -> ScriptedGraspExpert:
    """A cube expert at `placement`, with `overrides` folded into the config."""
    return ScriptedGraspExpert(CUBE, placement, config=ExpertConfig(**overrides))


# --- Planning ------------------------------------------------------------------


def test_plan_puts_the_tcp_where_the_waypoints_claim():
    """Both IK waypoints are closed back on the FK, not merely trusted."""
    plan = plan_grasp(CUBE, (0.19, 0.05, 0.3))
    assert plan.ok, plan.reason
    for q, target in ((plan.q_pregrasp, plan.tcp_pregrasp), (plan.q_grasp, plan.tcp_grasp)):
        assert np.linalg.norm(CHAIN.fk_tcp(q)[0] - target) < 1e-3


def object_in_tool_frame(plan, object_xy, object_z=CUBE.spawn_z) -> np.ndarray:
    """Where the object sits in the grasp pose's own tool frame, metres.

    The whole point of :func:`tcp_target`, read back through the FK rather than
    through the formula that produced it.
    """
    position, rotation = CHAIN.fk_tcp(plan.q_grasp)
    return rotation.T @ (np.array([*object_xy, object_z]) - position)


def test_the_object_lands_between_the_jaws_not_under_the_tcp():
    """The grasp pose puts the object at the jaw centre: offset in x, deeper in z.

    Aiming the TCP itself at the object -- the obvious reading of "grasp at
    (x, y)" -- buries the static finger in the object's lid, because the jaws
    straddle the wrist_roll axis and the TCP hangs 7.9 mm off it.
    """
    for object_xy in ((0.19, 0.0), (0.14, 0.09), (0.06, -0.17)):
        for object_yaw in np.linspace(-math.pi, math.pi, 9):
            plan = plan_grasp(CUBE, (*object_xy, float(object_yaw)))
            assert plan.ok, plan.reason
            local = object_in_tool_frame(plan, object_xy)
            assert local[0] == pytest.approx(plan.lateral_offset, abs=1e-3)
            assert local[1] == pytest.approx(0.0, abs=1e-3)
            assert local[2] == pytest.approx(TCP_TO_PAD_CENTRE, abs=1e-3)


@pytest.mark.parametrize("spec", [DOMINO, DUPLO], ids=["domino_20x40", "duplo_32x64"])
def test_the_jaws_line_up_with_the_short_axis_of_a_rectangular_object(spec):
    """The object's local x -- the axis ``grasp_width_m`` measures -- lands on the tool's x.

    The convention :class:`~manus.objects.ObjectSpec` enforces, closed back
    through the FK rather than argued from the formula: the jaws close along
    the tool's own x (see :func:`~manus.expert.tcp_target`), so this is what
    stops the plan asking them to span the domino's 40 mm length. Only the
    rectangular objects can express it -- for a square or round one both axes
    are the grasp axis.
    """
    for object_yaw in np.linspace(-math.pi, math.pi, 9):
        plan = plan_grasp(spec, (0.19, 0.02, float(object_yaw)))
        assert plan.ok, plan.reason
        _, rotation = CHAIN.fk_tcp(plan.q_grasp)
        local_x = np.array([math.cos(object_yaw), math.sin(object_yaw), 0.0])
        assert abs((rotation.T @ local_x)[0]) == pytest.approx(1.0, abs=2e-3)


def test_the_lateral_offset_clears_the_static_jaw_by_the_clearance():
    from manus.expert import JAW_CLEARANCE, JAW_FIXED_FACE_X

    offset = expert_mod.pad_lateral_offset(CUBE)
    assert offset < 0.0  # away from the static jaw, which sits at +x
    assert offset + 0.5 * CUBE.grasp_width_m == pytest.approx(
        JAW_FIXED_FACE_X - JAW_CLEARANCE
    )


def test_grasp_height_is_the_object_height_plus_the_pad_offset():
    plan = plan_grasp(CUBE, (0.19, 0.0, 0.0))
    assert plan.tcp_grasp[2] == pytest.approx(CUBE.spawn_z + TCP_TO_PAD_CENTRE)
    assert plan.tcp_pregrasp[2] == pytest.approx(plan.tcp_grasp[2] + 0.03)
    assert plan.tcp_pregrasp[0] == pytest.approx(plan.tcp_grasp[0])
    assert plan.tcp_pregrasp[1] == pytest.approx(plan.tcp_grasp[1])


def test_the_puck_is_the_only_grasp_the_table_pushes_up():
    """Pinned: the 10 mm puck is gripped 2.3 mm above its own centre, nothing else is."""
    from manus.expert import grasp_height

    assert grasp_height(PUCK) == pytest.approx(0.0073)
    assert grasp_height(PUCK) - PUCK.spawn_z == pytest.approx(0.0023)
    assert [
        spec.name for spec in OBJECTS.values() if grasp_height(spec) > spec.spawn_z
    ] == ["puck_d40x10"]


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_only_an_object_too_short_to_centre_on_is_raised(spec):
    """grasp_height centres the pads on the object unless the tips would hit the table."""
    from manus.expert import JAW_TIP_Z, MIN_TIP_CLEARANCE, grasp_height

    raise_by = grasp_height(spec) - spec.spawn_z
    assert raise_by >= 0.0
    tall_enough = spec.spawn_z + TCP_TO_PAD_CENTRE - JAW_TIP_Z >= MIN_TIP_CLEARANCE
    assert (raise_by == 0.0) == tall_enough, f"{spec.name} raised by {raise_by * 1e3:.1f} mm"
    # The pads must still land on the object, not above it.
    assert grasp_height(spec) < spec.spawn_z + 0.5 * spec.extent_z


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_the_fingertips_stay_off_the_table_at_the_grasp_pose(spec):
    """Every object's grasp pose keeps both jaws clear of the ground.

    Two separate clearances, because the two fingers reach their lowest at
    different moments. The static jaw is the one that matters on the way down
    (the moving jaw is swung right out at :data:`GRIPPER_OPEN`), and its tip is
    :data:`~manus.expert.JAW_TIP_Z` below the TCP. The moving jaw only comes
    down during CLOSE, with the arm already frozen, and it reaches deeper than
    the static one -- 8.06 mm below the TCP at 0.14 rad, measured here off the
    meshes -- so it gets its own, smaller bar.
    """
    from manus.expert import JAW_TIP_Z, MIN_TIP_CLEARANCE, grasp_height

    tcp_z = grasp_height(spec) + TCP_TO_PAD_CENTRE
    tip_height = tcp_z - JAW_TIP_Z
    assert tip_height >= MIN_TIP_CLEARANCE - 1e-9, (
        f"{spec.name}: fingertips only {tip_height * 1e3:.1f} mm off the floor"
    )
    assert tcp_z - MOVING_JAW_DEEPEST_Z > 0.003, (
        f"{spec.name}: the closing jaw comes within 3 mm of the table"
    )
    # ... and enough of the object is between the pads to be held by them.
    engaged = min(spec.extent_z, spec.spawn_z + 0.5 * spec.extent_z - tip_height)
    assert engaged >= 0.004, f"{spec.name}: only {engaged * 1e3:.1f} mm of the object is gripped"


def test_the_waypoints_are_tool_vertical():
    """PREGRASP and DESCEND keep the approach axis down; LIFT deliberately does not."""
    plan = plan_grasp(CUBE, (0.17, -0.06, 1.0))
    for q in (plan.q_pregrasp, plan.q_grasp):
        assert kinematics.pitch_sum(q) == pytest.approx(kinematics.PITCH_SUM_VERTICAL, abs=1e-3)


def test_plan_is_deterministic_for_a_placement():
    a = plan_grasp(CUBE, (0.18, 0.03, 0.7))
    b = plan_grasp(CUBE, (0.18, 0.03, 0.7))
    assert np.array_equal(a.q_grasp, b.q_grasp) and a.grasp_yaw == b.grasp_yaw


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_plan_covers_the_whole_region_for_every_object(spec):
    """Every legal placement at four yaws plans -- no silent holes to fall into.

    The catalogue's whole feasibility claim, and the one that a new object is
    most likely to break: the tool has to stand a grasp-half-width off the
    object (40 mm for the puck is a 22 mm stand-off, against the cube's 17), and
    a rectangular object can only spend two of the four yaw branches getting
    out of the way.
    """
    for x, y in region_samples():
        for yaw in np.radians([0.0, 37.0, 90.0, 143.0]):
            plan = plan_grasp(spec, (x, y, float(yaw)))
            assert plan.ok, f"{spec.name} at ({x:.3f}, {y:.3f}, {yaw:.2f}): {plan.reason}"


def test_plan_outside_the_region_reports_itself_infeasible():
    """A placement the arm cannot reach comes back as a plan with ok False."""
    plan = plan_grasp(CUBE, (0.60, 0.0, 0.0))
    assert not plan.ok and plan.reason.startswith("ik_")
    assert plan.q_grasp.shape == (ARM,)  # still a usable, in-limit pose


def test_plan_waypoints_stay_inside_the_joint_limits():
    for x, y in region_samples(3, 5):
        plan = plan_grasp(CUBE, (x, y, 0.4))
        for q in (plan.q_pregrasp, plan.q_grasp, plan.q_lift):
            lower = np.array([specs.JOINT_LIMITS[n][0] for n in kinematics.ARM_JOINT_NAMES])
            upper = np.array([specs.JOINT_LIMITS[n][1] for n in kinematics.ARM_JOINT_NAMES])
            assert np.all(q >= lower - 1e-9) and np.all(q <= upper + 1e-9)


# --- Jaw geometry, re-derived from the meshes -------------------------------------
#
# The three constants in manus.expert that describe the hand were measured off
# the vendored STLs. These tests re-measure them from the same files (plus the
# URDF origins) so a re-exported model cannot silently invalidate the grasp: a
# 2 mm drift in the inner face is a 2 mm miss on a 30 mm object.

STL_DIR = kinematics.SO101_URDF_PATH.parent / "assets"
"""Where the URDF's ``<mesh filename="assets/...">`` paths resolve to."""


def read_stl(name: str) -> np.ndarray:
    """Vertices (N, 3) of a binary STL, in metres (the vendor's unit)."""
    data = (STL_DIR / name).read_bytes()
    count = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    records = np.frombuffer(data[84 : 84 + count * 50], dtype=np.uint8).reshape(count, 50)
    return records[:, 12:48].copy().view("<f4").reshape(-1, 3).astype(float)


def urdf_origin(joint_or_link: str, index: int = 0) -> tuple[tuple, tuple]:
    """``(xyz, rpy)`` of a joint's origin, or of a link's `index`-th visual origin."""
    tree = ET.parse(kinematics.SO101_URDF_PATH)
    for element in tree.getroot():
        if element.get("name") != joint_or_link:
            continue
        origin = (
            element.find("origin")
            if element.tag == "joint"
            else element.findall("visual")[index].find("origin")
        )
        return (
            tuple(float(value) for value in origin.get("xyz").split()),
            tuple(float(value) for value in origin.get("rpy").split()),
        )
    raise AssertionError(f"{joint_or_link!r} not in the URDF")


def homogeneous(xyz, rpy) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = kinematics._rotation_from_rpy(rpy)
    matrix[:3, 3] = xyz
    return matrix


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def jaw_clouds(gripper: float) -> tuple[np.ndarray, np.ndarray]:
    """(static, moving) jaw vertices in the TCP frame at jaw angle `gripper`.

    TCP frame: +z is the approach axis (out of the jaws), +x is the side the
    static jaw is on -- the frame :mod:`manus.expert`'s constants are written in.
    """
    into_tcp = np.linalg.inv(homogeneous(*urdf_origin("gripper_frame_joint")))
    # Static jaw: the printed follower is gripper_link's second visual.
    static = transform(
        into_tcp @ homogeneous(*urdf_origin("gripper_link", index=1)),
        read_stl("wrist_roll_follower_so101_v1.stl"),
    )
    spin = np.eye(4)
    spin[:2, :2] = [
        [math.cos(gripper), -math.sin(gripper)],
        [math.sin(gripper), math.cos(gripper)],
    ]
    moving = transform(
        into_tcp
        @ homogeneous(*urdf_origin("gripper"))
        @ spin
        @ homogeneous(*urdf_origin("moving_jaw_so101_v1_link")),
        read_stl("moving_jaw_so101_v1.stl"),
    )
    return static, moving


def width_probe(width: float, height: float = 0.030) -> ObjectSpec:
    """A square block `width` metres across: sweeps the jaw geometry off-catalogue."""
    return ObjectSpec(
        name=f"probe_{width * 1e3:.0f}mm",
        shape="cuboid",
        half_extents=(0.5 * width, 0.5 * width, 0.5 * height),
        mass_kg=0.05,
        grasp_width_m=width,
        spawn_z=0.5 * height,
        close_target_rad=0.0,
        yaw_symmetry="quarter",
    )


def engaged_band(spec=CUBE) -> tuple[float, float]:
    """TCP-frame z range over which the fingers overlap a resting object, metres."""
    from manus.expert import JAW_TIP_Z, grasp_height

    top_of_object = spec.spawn_z + 0.5 * spec.extent_z
    return ((grasp_height(spec) + TCP_TO_PAD_CENTRE) - top_of_object, JAW_TIP_Z)


def jaw_gap(gripper: float, spec=CUBE) -> float:
    """Narrowest opening between the jaws over the engaged band, metres."""
    static, moving = jaw_clouds(gripper)
    low, high = engaged_band(spec)
    gaps = []
    for centre in np.arange(low, high, 0.002):
        def slab(cloud: np.ndarray, centre: float = centre) -> np.ndarray:
            """Vertices in this 2 mm depth slice, within the object's own width."""
            return cloud[
                (cloud[:, 2] >= centre)
                & (cloud[:, 2] < centre + 0.002)
                & (np.abs(cloud[:, 1]) < 0.5 * spec.grasp_width_m)
            ]

        fixed_face = slab(static)
        moving_face = slab(moving)
        if not len(fixed_face):
            continue
        if not len(moving_face):
            continue  # jaw swung clear of this slice entirely
        gaps.append(fixed_face[:, 0].min() - moving_face[:, 0].max())
    return min(gaps) if gaps else math.inf


def test_the_closing_jaw_reaches_deeper_than_the_tips():
    """Where MOVING_JAW_DEEPEST_Z comes from, and that it is past the static tip."""
    from manus.expert import JAW_TIP_Z

    depths = {angle: jaw_clouds(angle)[1][:, 2].max() for angle in np.arange(-0.17, 0.5, 0.02)}
    deepest = max(depths.values())
    assert deepest == pytest.approx(MOVING_JAW_DEEPEST_Z, abs=1e-4)
    assert deepest > JAW_TIP_Z
    assert max(depths, key=depths.get) == pytest.approx(0.14, abs=0.02)


def test_the_fingertip_constant_matches_the_mesh():
    from manus.expert import JAW_TIP_Z

    static, _ = jaw_clouds(0.0)
    assert static[:, 2].max() == pytest.approx(JAW_TIP_Z, abs=5e-4)


def test_the_static_jaw_face_constant_matches_the_mesh():
    """JAW_FIXED_FACE_X is the innermost static-jaw material over the engaged band.

    Innermost, because that is the point an object being lowered past the finger
    has to clear -- the deeper steps only matter once it is already down.
    """
    from manus.expert import JAW_FIXED_FACE_X

    static, _ = jaw_clouds(0.0)
    low, high = engaged_band()
    band = static[(static[:, 2] >= low) & (static[:, 2] <= high) & (np.abs(static[:, 1]) < 0.015)]
    assert len(band) > 100
    assert band[:, 0].min() == pytest.approx(JAW_FIXED_FACE_X, abs=1e-3)


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_the_descent_corridor_is_clear_of_the_static_jaw(spec):
    """With the planned offset, no static-jaw material overlaps the object's footprint."""
    static, moving = jaw_clouds(GRIPPER_OPEN)
    low, high = engaged_band(spec)
    half_width = 0.5 * spec.grasp_width_m
    band = static[
        (static[:, 2] >= low) & (static[:, 2] <= high) & (np.abs(static[:, 1]) < half_width)
    ]
    object_far_face = expert_mod.pad_lateral_offset(spec) + half_width
    assert band[:, 0].min() - object_far_face == pytest.approx(expert_mod.JAW_CLEARANCE, abs=1e-4)
    assert object_far_face < band[:, 0].min()
    # ... and the moving jaw is swung entirely out of the band while open, so
    # the widest objects descend past it too.
    assert moving[:, 2].max() < low


def test_the_open_jaws_clear_the_object_and_the_closed_jaws_squeeze_it():
    """The two facts the whole CLOSE phase rests on, measured off the meshes."""
    assert jaw_gap(GRIPPER_OPEN) > CUBE.grasp_width_m + 0.02
    assert jaw_gap(0.0) < CUBE.grasp_width_m - 0.005


def contact_angle(spec) -> float:
    """Jaw angle at which the pads first touch `spec`, from the meshes, radians."""
    low, high = -0.174533, 1.2
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if jaw_gap(middle, spec) < spec.grasp_width_m else (low, middle)
    return 0.5 * (low + high)


def test_the_measured_stall_anchor_matches_the_meshes():
    """The one sim measurement the whole catalogue is extrapolated from.

    A held 30 mm cube stalls the fingers at 0.189 rad in sim (Step 20, with the
    jaws on SDF colliders). The meshes say 0.195. They have to agree, because
    :func:`~manus.objects.close_target_for_width` carries the sim number to
    every other width along the *mesh* slope -- if they disagreed, the two
    halves of the formula would be describing different hands.
    """
    assert contact_angle(CUBE) == pytest.approx(objects.MEASURED_STALL_30MM_RAD, abs=0.01)


def test_the_jaw_width_rate_matches_the_meshes():
    """JAW_WIDTH_PER_RAD, re-measured: contact angle is linear in object width."""
    widths = np.array([0.016, 0.020, 0.030, 0.040])
    angles = np.array([contact_angle(width_probe(width)) for width in widths])
    slope, intercept = np.polyfit(angles, widths, 1)
    assert slope == pytest.approx(objects.JAW_WIDTH_PER_RAD, abs=5e-4)
    residual = np.abs(np.polyval([slope, intercept], angles) - widths).max()
    assert residual < 2e-4, f"contact angle is not linear in width: {residual * 1e3:.2f} mm off"


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_the_close_target_squeezes_rather_than_merely_touching(spec):
    """The close target must sit well below the angle the object stops the jaws at.

    A target *above* the contact angle never touches the object; a target just
    below it touches without force. Both leave the object on the table.
    """
    from manus import specs

    assert jaw_gap(spec.close_target_rad, spec) < spec.grasp_width_m - 0.005
    # 20 mrad of slack, because the pads are not parallel: the contact angle
    # depends on how far along the finger the object sits, and the formula is
    # anchored on the cube's engagement band. The puck, gripped at the very
    # tips, contacts 13 mrad early and so squeezes 13 mrad less than nominal.
    squeeze = contact_angle(spec) - spec.close_target_rad
    assert squeeze == pytest.approx(objects.SQUEEZE_RAD, abs=0.02), spec.name
    # ... and that squeeze has to be worth something in torque: 2.2-2.5 N.m
    # of the servo's 3.35, across the catalogue.
    assert squeeze * specs.STS3215_KP > 0.5 * specs.STS3215_EFFORT_LIMIT


# --- The lift retraction --------------------------------------------------------


def test_lift_raises_the_tcp_by_at_least_six_centimetres_everywhere():
    """The plan's LIFT bar, checked by FK over the region at four grasp yaws."""
    worst = math.inf
    for x, y in region_samples():
        for yaw in np.radians([0.0, 45.0, 90.0, 135.0]):
            plan = plan_grasp(CUBE, (x, y, float(yaw)))
            assert plan.ok, plan.reason
            start = CHAIN.fk_tcp(plan.q_grasp)[0][2]
            end = CHAIN.fk_tcp(plan.q_lift)[0][2]
            worst = min(worst, end - start)
    assert worst >= MIN_LIFT_RISE, f"worst lift rise {worst * 1e3:.1f} mm"


def test_lift_is_a_pure_pitch_retraction():
    """shoulder_pan and wrist_roll are untouched: the grasp does not swing or spin."""
    plan = plan_grasp(CUBE, (0.16, 0.08, 0.2))
    assert plan.q_lift[0] == pytest.approx(plan.q_grasp[0])
    assert plan.q_lift[4] == pytest.approx(plan.q_grasp[4])
    assert not np.allclose(plan.q_lift[1:4], plan.q_grasp[1:4])


def test_lift_reports_the_rise_it_actually_achieved():
    q_lift, rise = plan_lift(plan_grasp(CUBE, (0.19, 0.0, 0.0)).q_grasp, rise=0.09)
    assert rise == pytest.approx(0.09, abs=0.01)
    assert rise >= 0.09


def test_lift_from_a_folded_pose_stops_rather_than_spinning():
    """A pose with nowhere up to go returns quickly with an honest, small rise."""
    upper = np.array([specs.JOINT_LIMITS[n][1] for n in kinematics.ARM_JOINT_NAMES])
    q_lift, rise = plan_lift(upper.copy(), rise=1.0, max_steps=50)
    assert rise < 1.0 and np.all(np.isfinite(q_lift))


# --- Grasp yaw ------------------------------------------------------------------


def test_symmetry_is_a_quarter_turn_a_half_turn_or_free():
    assert yaw_symmetry(CUBE) == pytest.approx(math.pi / 2)
    assert yaw_symmetry(DOMINO) == pytest.approx(math.pi)
    assert yaw_symmetry(CYLINDER) == 0.0
    assert yaw_symmetry(BALL) == 0.0


def test_a_rectangular_object_is_offered_two_branches_not_four():
    """The two quarter turns that would ask the jaws to span the long side are gone."""
    candidates = grasp_yaw_candidates(DOMINO, 0.3, 1.0)
    assert len(candidates) == 2
    assert abs(expert_mod._wrap(candidates[0] - candidates[1])) == pytest.approx(math.pi)
    for candidate in candidates:
        halves = (candidate - 0.3) / math.pi
        assert halves == pytest.approx(round(halves), abs=1e-9)


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_every_candidate_presents_the_grasp_width_to_the_jaws(spec):
    """Each candidate lines the jaws up with the object's own grasp axis.

    Written in the world frame the jaws actually close in: the grasp direction
    is ``(cos yaw, sin yaw)`` (see :func:`~manus.expert.tcp_target`), and the
    object's extent along it must come out as ``grasp_width_m``.
    """
    object_yaw = 0.4
    for candidate in grasp_yaw_candidates(spec, object_yaw, 1.3):
        # Extent of the (possibly rectangular) footprint along the grasp axis.
        angle = candidate - object_yaw
        if spec.shape == "cuboid":
            half_x, half_y, _ = spec.half_extents
            extent = 2 * (abs(half_x * math.cos(angle)) + abs(half_y * math.sin(angle)))
        else:
            extent = 2 * spec.radius
        assert extent == pytest.approx(spec.grasp_width_m, abs=1e-9), math.degrees(angle)


@pytest.mark.parametrize("object_yaw_deg", range(-180, 181, 15))
def test_chosen_yaw_is_the_branch_nearest_the_current_tool_yaw(object_yaw_deg):
    """The preferred branch is within a 45 deg half-period of where the tool is."""
    tool_yaw = CHAIN.tool_yaw(HOME)
    candidates = grasp_yaw_candidates(CUBE, math.radians(object_yaw_deg), tool_yaw)
    distances = [abs(expert_mod._wrap(yaw - tool_yaw)) for yaw in candidates]
    assert distances[0] == pytest.approx(min(distances))
    assert distances[0] <= math.pi / 4 + 1e-9
    assert distances == sorted(distances)  # fallbacks in increasing travel order


@pytest.mark.parametrize("object_yaw_deg", [0, 17, 45, 91, -130])
def test_every_candidate_is_a_face_aligned_grasp(object_yaw_deg):
    """Candidates only ever differ from the object yaw by whole quarter turns."""
    object_yaw = math.radians(object_yaw_deg)
    for candidate in grasp_yaw_candidates(CUBE, object_yaw, 1.3):
        quarters = (candidate - object_yaw) / (math.pi / 2)
        assert quarters == pytest.approx(round(quarters), abs=1e-9)


def test_all_four_quarter_turns_are_offered():
    """A pi flip is a different grasp here -- it swaps which side the fixed jaw is on."""
    candidates = grasp_yaw_candidates(CUBE, 0.3, 1.0)
    assert len(candidates) == 4
    quarters = sorted(round((yaw - 0.3) / (math.pi / 2)) % 4 for yaw in candidates)
    assert quarters == [0, 1, 2, 3]


def test_the_pi_flip_moves_the_tool_to_the_other_side_of_the_object():
    """Why the flip cannot be treated as free: it is a 2 x 16 mm move of the TCP."""
    plans = [
        plan_grasp(CUBE, (0.19, 0.0, 0.0), np.array([0.0, 0.0, 0.0, 0.0, roll]))
        for roll in (0.0, math.pi)
    ]
    separation = float(np.linalg.norm(plans[0].tcp_grasp[:2] - plans[1].tcp_grasp[:2]))
    assert separation == pytest.approx(2 * abs(plans[0].lateral_offset), abs=1e-6)


def test_a_cylinder_is_grasped_at_the_current_tool_yaw():
    candidates = grasp_yaw_candidates(CYLINDER, 2.1, 0.4)
    assert candidates[0] == pytest.approx(0.4)


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_the_planned_wrist_roll_keeps_clear_of_its_limits(spec):
    """wrist_roll is the joint the yaw branch spends, and the one with least travel.

    The bars below are what this test's own 3x5 grid at 9 yaws sees (worst
    case 7.6 deg, on the duplo). Swept finer -- 5x9 placements at 13 yaws, too
    slow to run every time -- the worst cases are 11.8 deg for the cube, 13-17
    for the other free and quarter-turn objects, and **1.9 deg (domino) /
    1.0 deg (duplo)** for the rectangular ones: exactly the price of having two
    yaw branches instead of four, since the pair is pinned 180 deg apart and
    cannot be nudged off a stop. Still positive everywhere, so no grasp is
    planned into a joint the articulation would clamp; a rectangular object
    simply has no roll to spare, which is worth knowing before blaming the
    physics for a slightly yawed grasp.
    """
    lower, upper = specs.JOINT_LIMITS["wrist_roll"]
    margin = math.inf
    for x, y in region_samples(3, 5):
        for object_yaw in np.linspace(-math.pi, math.pi, 9):
            plan = plan_grasp(spec, (x, y, float(object_yaw)))
            assert plan.ok, plan.reason
            roll = plan.q_grasp[4]
            margin = min(margin, roll - lower, upper - roll)
    floor = 5.0 if spec.yaw_symmetry == "half" else 10.0
    assert margin > math.radians(floor), (
        f"{spec.name}: wrist_roll within {math.degrees(margin):.2f} deg of a limit"
    )


# --- FSM sequencing -------------------------------------------------------------


def test_a_converging_plant_walks_every_state_in_order():
    expert = fresh()
    trace = run(expert, FakeArm())
    assert [state for state, _ in trace][:1] == [PREGRASP]
    visited = [report.state for report in expert.reports]
    assert visited == [PREGRASP, DESCEND, CLOSE, LIFT, HOLD]
    assert expert.state == DONE and expert.done
    assert expert.timeouts == []
    assert [report.exit for report in expert.reports] == [
        "converged",
        "converged",
        "stalled",
        "converged",
        "elapsed",
    ]


def test_states_advance_on_convergence_not_on_the_budget():
    """Each arm state ends well inside its budget and inside the tolerance."""
    config = ExpertConfig()
    expert = fresh()
    run(expert, FakeArm())
    for report in expert.reports:
        assert report.steps < config.state_budget
        if report.state in ARM_STATES:
            assert report.joint_error < config.converge_tol


def test_a_wedged_arm_times_out_of_every_arm_state_and_says_so():
    """A plant that never moves burns exactly the budget in each arm state.

    CLOSE is the exception and rightly so: a jaw that cannot move *is* stalled,
    which is the same observation as a jaw squeezing an object, so CLOSE ends on
    its stall rule rather than on the budget.
    """
    config = ExpertConfig(state_budget=40, hold_steps=5, close_ramp=10)
    expert = ScriptedGraspExpert(CUBE, (0.20, 0.0, 0.0), config=config)
    run(expert, FakeArm(frozen=True), max_steps=1000)
    assert expert.timeouts == [PREGRASP, DESCEND, LIFT]
    for report in expert.reports:
        if report.state in ARM_STATES:
            assert report.exit == "timeout" and report.steps == config.state_budget
    close = [report for report in expert.reports if report.state == CLOSE][0]
    assert close.exit == "stalled"


def test_the_hold_runs_its_configured_length():
    expert = fresh(hold_steps=17)
    run(expert, FakeArm())
    hold = [report for report in expert.reports if report.state == HOLD][0]
    assert hold.steps == 17 and hold.exit == "elapsed"


def test_step_after_done_repeats_the_final_command():
    expert = fresh()
    plant = FakeArm()
    run(expert, plant)
    last = expert.step(plant.q)
    again = expert.step(plant.q)
    assert last == again and expert.state == DONE


def test_step_before_reset_is_an_error():
    expert = ScriptedGraspExpert(CUBE)
    with pytest.raises(RuntimeError):
        expert.step(np.zeros(6))
    with pytest.raises(RuntimeError):
        _ = expert.plan


def test_reset_rewinds_the_fsm_and_can_re_place_the_object():
    expert = fresh()
    run(expert, FakeArm())
    plan = expert.reset((0.14, -0.05, 0.9))
    assert expert.state == PREGRASP and not expert.done
    assert expert.reports == [] and expert.timeouts == []
    assert object_in_tool_frame(plan, (0.14, -0.05))[0] == pytest.approx(
        plan.lateral_offset, abs=1e-3
    )
    assert np.allclose(expert.bias, 0.0)


def test_reset_without_a_placement_reuses_the_plan():
    expert = fresh()
    first = expert.plan.q_grasp.copy()
    run(expert, FakeArm())
    expert.reset()
    assert np.array_equal(expert.plan.q_grasp, first)


def test_reset_needs_a_placement_the_first_time():
    with pytest.raises(ValueError):
        ScriptedGraspExpert(CUBE).reset()


# --- Commands -------------------------------------------------------------------


def test_every_command_is_a_full_in_limit_six_joint_dict():
    expert = fresh()
    plant = FakeArm()
    measured = plant.q.copy()
    while not expert.done:
        targets = expert.step(measured)
        assert set(targets) == set(specs.JOINT_NAMES)
        for name, value in targets.items():
            lower, upper = specs.JOINT_LIMITS[name]
            assert lower - 1e-12 <= value <= upper + 1e-12
        measured = plant.apply(targets)


def test_the_jaws_stay_open_until_close_and_shut_afterwards():
    expert = fresh()
    plant = FakeArm(jaw_stop=0.6)
    measured = plant.q.copy()
    seen: dict[str, list[float]] = {}
    while not expert.done:
        targets = expert.step(measured)
        # A command belongs to the state the FSM is in *after* the call: step()
        # settles the transition first, then issues the new state's command.
        seen.setdefault(expert.state, []).append(targets["gripper"])
        measured = plant.apply(targets)
    assert seen[PREGRASP][-1] == pytest.approx(GRIPPER_OPEN)
    assert all(value == pytest.approx(GRIPPER_OPEN) for value in seen[DESCEND])
    assert seen[CLOSE][-1] == pytest.approx(CUBE.close_target_rad)
    assert all(v == pytest.approx(CUBE.close_target_rad) for v in seen[LIFT] + seen[HOLD])


def test_the_arm_is_frozen_through_close():
    """CLOSE must not move the arm: the jaws close on the pose DESCEND reached."""
    expert = fresh()
    plant = FakeArm(jaw_stop=0.5)
    measured = plant.q.copy()
    commands = []
    while not expert.done:
        targets = expert.step(measured)
        if expert.state == CLOSE:  # the state the command belongs to
            commands.append([targets[n] for n in kinematics.ARM_JOINT_NAMES])
        measured = plant.apply(targets)
    assert len(commands) > 1
    assert np.allclose(np.ptp(np.array(commands), axis=0), 0.0)


def test_the_ramp_walks_the_arm_in_rather_than_stepping_it():
    """The first PREGRASP command is a fraction of the way, not the whole way."""
    expert = fresh(pregrasp_ramp=45)
    first = expert.step(np.zeros(6))
    travel = np.array([first[n] for n in kinematics.ARM_JOINT_NAMES])
    waypoint = expert.plan.q_pregrasp
    assert np.linalg.norm(travel) < 0.2 * np.linalg.norm(waypoint)


# --- CLOSE ----------------------------------------------------------------------


def test_close_ends_when_the_jaws_stall_on_the_object():
    expert = fresh()
    run(expert, FakeArm(jaw_stop=0.55))
    close = [report for report in expert.reports if report.state == CLOSE][0]
    assert close.exit == "stalled"
    assert close.gripper == pytest.approx(0.55, abs=1e-6)


def test_close_cannot_stall_before_the_window_has_filled():
    """A jaw that is still travelling does not count as stalled, however slowly."""
    config = ExpertConfig(gripper_stall_window=15, close_ramp=1, state_budget=200)
    expert = ScriptedGraspExpert(CUBE, (0.20, 0.0, 0.0), config=config)
    run(expert, FakeArm(follow=0.02))  # jaw creeps: < 0.002 rad only much later
    close = [report for report in expert.reports if report.state == CLOSE][0]
    assert close.steps > config.gripper_stall_window


def test_close_times_out_if_the_jaws_never_settle():
    """A jaw driven by an oscillating plant burns the budget instead of hanging."""

    class Chatter(FakeArm):
        def apply(self, targets):
            measured = super().apply(targets)
            self.q[len(specs.JOINT_NAMES) - 1] += 0.05 * (-1) ** len(str(id(measured)))
            self.q[len(specs.JOINT_NAMES) - 1] = float(
                np.clip(self.q[len(specs.JOINT_NAMES) - 1], 0.0, GRIPPER_OPEN)
            )
            return self.q.copy()

    config = ExpertConfig(state_budget=60, hold_steps=5)
    expert = ScriptedGraspExpert(CUBE, (0.20, 0.0, 0.0), config=config)
    plant = Chatter()
    measured = plant.q.copy()
    for _ in range(2000):
        if expert.done:
            break
        targets = expert.step(measured)
        measured = plant.apply(targets)
    close = [report for report in expert.reports if report.state == CLOSE]
    assert close and close[0].steps <= config.state_budget


# --- Droop compensation ----------------------------------------------------------


def test_the_droop_integrator_drives_the_measured_pose_onto_the_waypoint():
    droop = np.array([0.01, 0.05, -0.03, 0.02, 0.0])
    expert = fresh()
    run(expert, FakeArm(droop=droop))
    assert expert.timeouts == []
    for report in expert.reports:
        if report.state in (PREGRASP, DESCEND):
            assert report.joint_error < ExpertConfig().converge_tol
            assert report.tcp_error is not None and report.tcp_error < 3e-3


def test_the_bias_measures_the_droop_it_cancels():
    """commanded - measured at convergence is the droop, which is the report."""
    droop = np.array([0.0, 0.06, -0.04, 0.0, 0.0])
    expert = fresh()
    run(expert, FakeArm(droop=droop))
    descend = [report for report in expert.reports if report.state == DESCEND][0]
    assert np.allclose(np.array(descend.bias), droop, atol=0.02)


def test_without_the_integrator_a_drooping_arm_never_converges():
    """The control that shows the integrator is doing the work, not the tolerance."""
    droop = np.array([0.0, 0.06, -0.04, 0.0, 0.0])
    config = ExpertConfig(droop_gain=0.0, state_budget=60, hold_steps=5)
    expert = ScriptedGraspExpert(CUBE, (0.20, 0.0, 0.0), config=config)
    run(expert, FakeArm(droop=droop), max_steps=1000)
    assert PREGRASP in expert.timeouts and DESCEND in expert.timeouts


def test_the_bias_is_capped():
    config = ExpertConfig(droop_limit=0.05, state_budget=80, hold_steps=5)
    expert = ScriptedGraspExpert(CUBE, (0.20, 0.0, 0.0), config=config)
    run(expert, FakeArm(droop=np.full(ARM, 0.15)), max_steps=1000)
    assert np.all(np.abs(expert.bias) <= 0.05 + 1e-12)


def test_the_integrator_does_not_wind_up_while_the_arm_is_still_travelling():
    expert = fresh(droop_engage=0.05)
    plant = FakeArm(follow=0.05)  # slow: large errors persist for many steps
    measured = plant.q.copy()
    for _ in range(30):
        measured = plant.apply(expert.step(measured))
    assert np.allclose(expert.bias, 0.0)


# --- Measurement plumbing ---------------------------------------------------------


def test_joint_vector_accepts_a_mapping_or_a_sequence():
    values = [0.1, -0.2, 0.3, -0.4, 0.5, 0.6]
    mapping = dict(zip(specs.JOINT_NAMES, values, strict=True))
    assert np.allclose(joint_vector(mapping), values)
    assert np.allclose(joint_vector(np.array(values)), values)


def test_joint_vector_rejects_malformed_measurements():
    with pytest.raises(KeyError):
        joint_vector({"shoulder_pan": 0.0})
    with pytest.raises(ValueError):
        joint_vector([0.0] * 5)
    with pytest.raises(ValueError):
        joint_vector([0.0, 0.0, 0.0, 0.0, 0.0, float("nan")])


def test_the_fsm_gives_the_same_answer_for_both_measurement_forms():
    values = np.zeros(6)
    a = fresh().step(values)
    b = fresh().step(dict(zip(specs.JOINT_NAMES, values, strict=True)))
    assert a == b


def test_an_episode_draw_can_be_handed_in_directly():
    draw = draw_episode("expert_demo", 3)
    expert = ScriptedGraspExpert(CUBE, draw)
    local = object_in_tool_frame(expert.plan, (draw.object_x, draw.object_y))
    assert local[0] == pytest.approx(expert.plan.lateral_offset, abs=1e-3)
    assert local[2] == pytest.approx(TCP_TO_PAD_CENTRE, abs=1e-3)


def test_a_malformed_placement_is_rejected():
    with pytest.raises(ValueError):
        ScriptedGraspExpert(CUBE, (0.2, 0.0))


def test_state_sequence_is_walked_by_index():
    """The FSM advances by position in STATE_SEQUENCE; pin the order it assumes."""
    assert STATE_SEQUENCE == (PREGRASP, DESCEND, CLOSE, LIFT, HOLD, DONE)


def test_telemetry_is_json_ready():
    import json

    expert = fresh()
    run(expert, FakeArm())
    payload = json.dumps(expert.telemetry())
    assert '"states"' in payload and '"plan_ok": true' in payload


# --- Success predicate -------------------------------------------------------------


def test_success_needs_the_height_the_sustain_and_the_jaws():
    monitor = GraspSuccessMonitor(0.015, lift=0.05, sustain=30, gripper_max=1.0)
    for _ in range(29):
        assert not monitor.update(0.10, 0.5)
    assert monitor.update(0.10, 0.5)
    assert monitor.success and monitor.best_streak >= 30


def test_a_dropped_object_restarts_the_count():
    monitor = GraspSuccessMonitor(0.015, sustain=10)
    for _ in range(9):
        monitor.update(0.10, 0.5)
    monitor.update(0.02, 0.5)  # dropped
    for _ in range(9):
        monitor.update(0.10, 0.5)
    assert not monitor.success and monitor.streak == 9


def test_an_open_gripper_never_succeeds_however_high_the_object():
    monitor = GraspSuccessMonitor(0.015, sustain=5, gripper_max=1.0)
    for _ in range(50):
        monitor.update(0.30, GRIPPER_OPEN)
    assert not monitor.success and monitor.peak_z == pytest.approx(0.30)


def test_the_bar_is_the_spawn_height_plus_the_lift():
    monitor = GraspSuccessMonitor(0.03, lift=0.05)
    assert monitor.threshold_z == pytest.approx(0.08)
    for _ in range(40):
        monitor.update(0.0799, 0.4)
    assert not monitor.success


def test_success_latches():
    monitor = GraspSuccessMonitor(0.015, sustain=2)
    monitor.update(0.10, 0.4)
    monitor.update(0.10, 0.4)
    assert monitor.success
    monitor.update(0.0, GRIPPER_OPEN)
    assert monitor.success and monitor.to_dict()["success"] is True


# --- Failure taxonomy ---------------------------------------------------------------


def finished(placement=(0.20, 0.0, 0.0), **kwargs) -> ScriptedGraspExpert:
    """An expert that has run a whole attempt against a converging plant."""
    expert = fresh(placement, **kwargs)
    run(expert, FakeArm(jaw_stop=0.5))
    return expert


def fed(monitor: GraspSuccessMonitor, heights, gripper=0.5) -> GraspSuccessMonitor:
    """Feed a height trace into a monitor."""
    for height in heights:
        monitor.update(height, gripper)
    return monitor


def test_a_met_predicate_is_a_success():
    monitor = fed(GraspSuccessMonitor(0.015), [0.10] * 40)
    assert classify_outcome(finished(), monitor) == "success"


def test_an_unreachable_placement_is_named_as_such():
    expert = fresh((0.60, 0.0, 0.0))
    run(expert, FakeArm(jaw_stop=0.5), max_steps=6000)
    monitor = fed(GraspSuccessMonitor(0.015), [0.015] * 40)
    assert not expert.plan.ok
    assert classify_outcome(expert, monitor) == "ik_infeasible"


def test_over_the_bar_but_not_held_is_a_slip():
    monitor = fed(GraspSuccessMonitor(0.015), [0.10] * 10 + [0.015] * 40)
    assert classify_outcome(finished(), monitor) == "slipped"


def test_lifted_but_short_of_the_bar_is_a_short_lift():
    monitor = fed(GraspSuccessMonitor(0.015), [0.05] * 40)
    assert classify_outcome(finished(), monitor) == "short_lift"


def test_an_untouched_object_after_a_clean_run_is_a_missed_grasp():
    monitor = fed(GraspSuccessMonitor(0.015), [0.015] * 40)
    expert = finished()
    assert expert.timeouts == []
    assert classify_outcome(expert, monitor) == "no_grasp"


def test_an_untouched_object_after_a_wedged_run_is_a_timeout():
    expert = fresh(state_budget=30, hold_steps=5)
    run(expert, FakeArm(frozen=True), max_steps=1000)
    monitor = fed(GraspSuccessMonitor(0.015), [0.015] * 40)
    assert classify_outcome(expert, monitor) == "timeout"
