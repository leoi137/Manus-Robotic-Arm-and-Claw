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

import dataclasses
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
from manus.randomize import draw_episode, placement_region

CHAIN = KinematicChain()
CUBE = OBJECTS["cube_3cm"]
CYLINDER = OBJECTS["cylinder_3cm"]  # the catalogue's one SIDE grasp
DIE = OBJECTS["die_16mm"]  # the narrowest grasp, and the tightest converge_tol
DOMINO = OBJECTS["domino_20x40"]  # the rectangular case: two branches, not four
PUCK = OBJECTS["puck_d40x10"]  # the widest grasp and the only raised one
THICK_PUCK = OBJECTS["puck_d40x20"]  # the same disc with a rim the pads can centre on
BALL = OBJECTS["pingpong_40mm"]  # 2.7 g, round in every direction
DUPLO = OBJECTS["duplo_32x64"]  # the other rectangular case, 32 mm across

TOP_CYLINDER = dataclasses.replace(CYLINDER, grasp_mode="top")
"""The cylinder as it was grasped before Step 23: straight down, at 36 mm.

Kept as a fixture rather than deleted, because several tests below are the
*record* of why the top-down cylinder failed -- and the top-down arithmetic that
produced those numbers has to go on producing them, or "we changed the mode, not
the maths" is an untested claim.
"""

TOP_OBJECTS = [spec for spec in OBJECTS.values() if spec.grasp_mode == "top"]
SIDE_OBJECTS = [spec for spec in OBJECTS.values() if spec.grasp_mode == "side"]
TOP_IDS = [spec.name for spec in TOP_OBJECTS]
SIDE_IDS = [spec.name for spec in SIDE_OBJECTS]
"""The catalogue split by grasp mode: the top-down bars do not apply to a side
grasp and vice versa, so the parametrised geometry tests run over one or the
other rather than over everything."""

ARM = kinematics.NUM_ARM_JOINTS
HOME = np.zeros(ARM)

MIN_LIFT_RISE = 0.06
"""TCP rise the LIFT retraction has to deliver, metres (the plan's bar)."""

MOVING_JAW_DEEPEST_Z = expert_mod.MOVING_JAW_DEEPEST_Z
"""Deepest the *moving* jaw reaches below the TCP over the closing sweep, metres.

The constant :mod:`manus.expert` carries, re-measured off the meshes by
:func:`test_the_closing_jaw_reaches_deeper_than_the_tips` below. It peaks at
0.14 rad, 1.8 mm past the static fingertips: the finger swings down as it closes
and back up as it goes past. Only the short objects care, and only during CLOSE
-- the arm is frozen by then, so this is a clearance to keep, not a motion to
plan.
"""


def region_samples(
    n_radius: int = 5, n_azimuth: int = 9, spec=None
) -> list[tuple[float, float]]:
    """A coarse grid of legal placements spanning `spec`'s whole grasp region.

    ``spec=None`` means the top-down region, which is what every caller written
    before side grasps existed wants.
    """
    region = placement_region(spec)
    r_min, r_max = region.radius
    points = []
    for radius in np.linspace(r_min, r_max, n_radius):
        for azimuth in np.radians(
            np.linspace(-region.azimuth_max_deg, region.azimuth_max_deg, n_azimuth)
        ):
            x = region.pan_axis_xy[0] + radius * math.cos(azimuth)
            y = region.pan_axis_xy[1] + radius * math.sin(azimuth)
            if not region.in_keepout(x, y):
                points.append((float(x), float(y)))
    return points


def a_placement(spec, yaw: float = 0.0) -> tuple[float, float, float]:
    """One legal ``(x, y, yaw)`` in the middle of `spec`'s own region.

    The stand-in for the ``(0.20, 0.0, 0.0)`` literal that the FSM and predicate
    fixtures used when every object shared one region.
    """
    region = placement_region(spec)
    return (region.pan_axis_xy[0] + sum(region.radius) / 2.0, region.pan_axis_xy[1], yaw)


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


# --- The hover ------------------------------------------------------------------


def hover_clearance(spec, config=ExpertConfig()) -> float:
    """Gap between the lowest jaw material at PREGRASP and the top of `spec`, metres."""
    from manus.expert import jaw_depth, pregrasp_height

    return pregrasp_height(spec, config) - jaw_depth(config.gripper_open) - spec.top_z


@pytest.mark.parametrize("spec", TOP_OBJECTS, ids=TOP_IDS)
def test_the_hover_clears_the_top_of_every_object(spec):
    """The Step 21 fix, by FK: no jaw is inside the object at the pregrasp waypoint.

    Read off the *solved* waypoint rather than the requested height, so an IK
    solve that quietly landed low would fail this too.

    Top-down objects only: a side grasp's PREGRASP is beside the object at the
    grasp's own height, not above it, so there is no jaw over the lid to clear
    -- see :func:`test_the_side_pregrasp_stands_off_along_the_approach`.
    """
    from manus.expert import jaw_depth, pregrasp_height

    config = ExpertConfig()
    for x, y in region_samples(3, 5):
        plan = plan_grasp(spec, (x, y, 0.4), config=config)
        assert plan.ok, plan.reason
        tcp_z = float(CHAIN.fk_tcp(plan.q_pregrasp)[0][2])
        lowest = tcp_z - jaw_depth(config.gripper_open)
        assert lowest - spec.top_z >= config.hover_margin - 1e-3, (
            f"{spec.name}: jaws {(lowest - spec.top_z) * 1e3:.2f} mm over its top"
        )
        assert tcp_z == pytest.approx(pregrasp_height(spec, config), abs=1e-3)


def test_every_hover_height_is_pinned():
    """The absolute PREGRASP height of every catalogue object, to 0.1 mm.

    The cube's 49.0 mm is the number the 200-attempt Step 8 gate was run at and
    may not move; the cylinder's 74.3 mm is the Step 21 hover fix and may not
    move either -- in particular the Step 22 grasp-height change must not move
    it, because the two bars in :func:`~manus.expert.pregrasp_height` are a max
    and the taller one is still the object-clearing bar.
    """
    from manus.expert import pregrasp_height

    assert {
        name: round(pregrasp_height(spec) * 1e3, 1) for name, spec in OBJECTS.items()
    } == {
        "cube_3cm": 49.0,
        # The side grasp waits at the grasp's own height and stands off
        # radially instead, so its "hover" is the cup height exactly.
        "cylinder_3cm": 40.0,
        "die_16mm": 42.0,
        "domino_20x40": 41.5,
        "puck_d40x10": 41.3,
        # The thick puck's grasp was raised 4.1 mm (tip_clearance_m), and its
        # hover rides up with it: the stand-off bar (grasp + 4 + 30) is the
        # binding one for it, not the object-clearing bar (34.3 mm).
        "puck_d40x20": 48.1,
        "pingpong_40mm": 54.3,
        "duplo_32x64": 46.0,
    }
    # ... and the top-down arithmetic that produced the cylinder's old 74.3 mm
    # is untouched: only the mode changed.
    assert round(pregrasp_height(TOP_CYLINDER) * 1e3, 1) == 74.3


def test_the_hover_only_rises_for_an_object_tall_enough_to_need_it():
    """Pinned: the cylinder is raised 4.3 mm, the ball 0.3, and nothing else moves.

    The cube's hover in particular is untouched -- it is what the 200-attempt
    gate was run at.

    The cylinder's raise is *above its own stand-off*, and the stand-off moved
    at Step 22 (its grasp height went 30 -> 36 mm, see
    :data:`~manus.expert.JAW_PARALLEL_REACH`), so this number shrank from 10.3
    to 4.3 while the hover itself stayed at 74.3 mm -- which is what
    :func:`test_every_hover_height_is_pinned` is for.
    """
    from manus.expert import grasp_height, pregrasp_height

    config = ExpertConfig()
    raised = {
        spec.name: round((pregrasp_height(spec, config)
                          - (grasp_height(spec) + TCP_TO_PAD_CENTRE + config.hover_height)) * 1e3, 1)
        for spec in [*TOP_OBJECTS, TOP_CYLINDER]
    }
    assert raised == {
        "cube_3cm": 0.0,
        "cylinder_3cm": 4.3,
        "die_16mm": 0.0,
        "domino_20x40": 0.0,
        "puck_d40x10": 0.0,
        "puck_d40x20": 0.0,
        "pingpong_40mm": 0.3,
        "duplo_32x64": 0.0,
    }


def test_the_old_fixed_hover_really_did_bury_the_fingers_in_the_cylinder():
    """The failure this fix is for, reproduced from the geometry that caused it.

    The preview run knocked the 60 mm cylinder over on arrival at PREGRASP; the
    fixed 30 mm stand-off put the fingertips 2.3 mm below its top, because that
    stand-off is measured from the grasp pose at the object's *mid*-height.
    """
    from manus.expert import jaw_depth

    config = ExpertConfig()
    # The mid-height grasp the fixed stand-off was measured from, spelled out
    # rather than read from grasp_height(), which no longer returns it for this
    # object (Step 22 raised it; see JAW_PARALLEL_REACH).
    old_tcp_z = CYLINDER.spawn_z + TCP_TO_PAD_CENTRE + config.hover_height
    old_gap = old_tcp_z - jaw_depth(config.gripper_open) - CYLINDER.top_z
    assert old_gap == pytest.approx(-0.0023, abs=1e-4)
    assert hover_clearance(TOP_CYLINDER) == pytest.approx(0.008, abs=1e-4)


def test_the_descent_starts_at_the_hover_and_goes_straight_down():
    """DESCEND's start is the raised hover, and it is directly above the grasp."""
    from manus.expert import pregrasp_height

    for spec in (CUBE, BALL, THICK_PUCK):
        plan = plan_grasp(spec, (0.19, 0.02, 0.3))
        assert plan.tcp_pregrasp[2] == pytest.approx(pregrasp_height(spec))
        assert plan.tcp_pregrasp[:2] == pytest.approx(plan.tcp_grasp[:2])
        # DESCEND's waypoint is the grasp pose and its entry pose is the hover:
        # the descent is the whole gap, however tall the object made it.
        assert np.array_equal(plan.waypoint(DESCEND), plan.q_grasp)
        drop = CHAIN.fk_tcp(plan.q_pregrasp)[0][2] - CHAIN.fk_tcp(plan.q_grasp)[0][2]
        assert drop == pytest.approx(pregrasp_height(spec) - plan.tcp_grasp[2], abs=1e-3)


def test_a_taller_hover_margin_lifts_the_hover_and_a_zero_one_does_not():
    """The margin is a real knob, not a constant baked into the formula."""
    from manus.expert import pregrasp_height

    generous = pregrasp_height(BALL, ExpertConfig(hover_margin=0.02))
    assert generous == pytest.approx(pregrasp_height(BALL) + 0.012)
    assert pregrasp_height(CUBE, ExpertConfig(hover_margin=0.0)) == pytest.approx(
        pregrasp_height(CUBE)
    )


def test_the_two_pucks_are_the_only_grasps_the_tip_clearance_bar_sets():
    """Pinned: both pucks are gripped above their own centre, and nothing else is.

    They get there for opposite reasons, which is why the bar is a *default*
    with a per-object override. The thin puck is simply too short to centre the
    pads on, so :data:`~manus.expert.MIN_TIP_CLEARANCE` pushes it up 2.3 mm. The
    thick one could be centred and is deliberately not: its own
    ``tip_clearance_m`` raises it 4.1 mm, to the height at which the closing
    finger's deepest sweep stops reaching under its centre of mass (see
    :func:`test_the_thick_pucks_grasp_is_raised_out_of_the_levering_band`).
    """
    from manus.expert import JAW_TIP_Z, grasp_height, tip_clearance

    assert grasp_height(PUCK) == pytest.approx(0.0073)
    assert grasp_height(PUCK) - PUCK.spawn_z == pytest.approx(0.0023)
    assert grasp_height(THICK_PUCK) - THICK_PUCK.spawn_z == pytest.approx(0.0041, abs=1e-4)
    pushed_up = [
        spec.name
        for spec in OBJECTS.values()
        if spec.grasp_mode == "top"
        and grasp_height(spec) == pytest.approx(
            JAW_TIP_Z - TCP_TO_PAD_CENTRE + tip_clearance(spec)
        )
    ]
    assert pushed_up == ["puck_d40x10", "puck_d40x20"]


def test_every_grasp_height_is_pinned():
    """The grasp height of every catalogue object, to 0.1 mm.

    The cube's 15.0 mm is its own mid-height and is what the gate was run at.
    The three objects that are *not* their own mid-height are the ones the hand's
    own geometry moved: the thin puck, raised 2.3 mm so the fingertips clear the
    table; the thick puck, raised 4.1 mm so the closing finger stops reaching
    under its centre of mass; and the cylinder, taken from the side at its cup
    height rather than its waist.
    """
    from manus.expert import grasp_height

    assert {
        name: round(grasp_height(spec) * 1e3, 1) for name, spec in OBJECTS.items()
    } == {
        "cube_3cm": 15.0,
        # Side grasp: two thirds of the way up, the way a hand takes a cup,
        # which is 12.2 mm of table clearance under the hand's housing.
        "cylinder_3cm": 40.0,
        "die_16mm": 8.0,
        "domino_20x40": 7.5,
        "puck_d40x10": 7.3,
        "puck_d40x20": 14.1,
        "pingpong_40mm": 20.0,
        "duplo_32x64": 12.0,
    }
    # The cylinder's 36.0 mm was the top-down answer and still is: the object
    # moved mode, the two bars in grasp_height did not move.
    assert round(grasp_height(TOP_CYLINDER) * 1e3, 1) == 36.0


@pytest.mark.parametrize("spec", [*TOP_OBJECTS, TOP_CYLINDER], ids=[*TOP_IDS, "cylinder_top"])
def test_only_an_object_the_hand_cannot_centre_on_is_raised(spec):
    """grasp_height centres the pads on the object unless one of its two bars binds.

    Too short and the fingertips would hit the table; too tall and the closing
    jaw would lean into the object's upper body before the pads reached it.
    Everything in between is grasped at its own mid-height -- unless the
    catalogue has overridden its tip clearance, which is a deliberate,
    per-object raise rather than one of the rule's own two bars (the thick
    puck's, see :data:`~manus.expert.MIN_TIP_CLEARANCE`).
    """
    from manus.expert import (
        JAW_PARALLEL_REACH,
        JAW_TIP_Z,
        grasp_height,
        tip_clearance,
    )

    raise_by = grasp_height(spec) - spec.spawn_z
    assert raise_by >= 0.0
    tall_enough = spec.spawn_z + TCP_TO_PAD_CENTRE - JAW_TIP_Z >= tip_clearance(spec)
    short_enough = spec.top_z - (spec.spawn_z + TCP_TO_PAD_CENTRE) <= JAW_PARALLEL_REACH
    assert (raise_by == 0.0) == (tall_enough and short_enough), (
        f"{spec.name} raised by {raise_by * 1e3:.1f} mm"
    )
    # The pads must still land on the object, not above it.
    assert grasp_height(spec) < spec.spawn_z + 0.5 * spec.extent_z


@pytest.mark.parametrize("spec", TOP_OBJECTS, ids=TOP_IDS)
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
    for x, y in region_samples(spec=spec):
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


def test_the_moving_jaw_stops_leading_where_the_constant_says_it_does():
    """MOVING_JAW_CLEAR_RAD, re-measured: past it the static tips are the lowest point."""
    from manus.expert import JAW_TIP_Z, MOVING_JAW_CLEAR_RAD

    below = jaw_clouds(MOVING_JAW_CLEAR_RAD - 0.01)[1][:, 2].max()
    above = jaw_clouds(MOVING_JAW_CLEAR_RAD + 0.01)[1][:, 2].max()
    assert below > JAW_TIP_Z > above


def test_jaw_depth_at_the_open_angle_is_the_static_fingertip():
    """The number the hover is built on: with the jaws open, the tips are the low point.

    Not an assumption -- at :data:`~manus.control.GRIPPER_OPEN` the moving
    finger has swung tens of millimetres *behind* the TCP, so the hover only has
    to clear the static tips.
    """
    from manus.expert import JAW_TIP_Z, jaw_depth

    static, moving = jaw_clouds(GRIPPER_OPEN)
    assert max(static[:, 2].max(), moving[:, 2].max()) == pytest.approx(JAW_TIP_Z, abs=5e-4)
    assert moving[:, 2].max() < -0.05
    assert jaw_depth(GRIPPER_OPEN) == pytest.approx(JAW_TIP_Z)
    # Below the crossing the answer is the conservative one, never an underestimate.
    for angle in np.arange(-0.17, expert_mod.MOVING_JAW_CLEAR_RAD, 0.02):
        measured = max(cloud[:, 2].max() for cloud in jaw_clouds(float(angle)))
        assert jaw_depth(float(angle)) >= measured - 1e-4


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


@pytest.mark.parametrize("spec", TOP_OBJECTS, ids=TOP_IDS)
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


def first_rim_contact(spec, clearance: float) -> tuple[float, float, float]:
    """Where the closing jaw first meets a short object's rim, off the meshes.

    Sweeps the moving jaw shut and returns, for the first angle at which its
    material reaches the object's far face inside the rim's own height band,
    ``(angle, contact height above the table, pad purchase on the rim)`` in
    radians and metres.
    """
    tcp_z = clearance + expert_mod.JAW_TIP_Z
    top_face, table = tcp_z - spec.extent_z, tcp_z
    half_width = 0.5 * spec.grasp_width_m
    far_face = expert_mod.pad_lateral_offset(spec) - half_width
    for angle in np.arange(1.2, -0.175, -0.002):
        _, moving = jaw_clouds(float(angle))
        rim = moving[
            (moving[:, 2] >= top_face)
            & (moving[:, 2] <= table)
            & (np.abs(moving[:, 1]) <= half_width)
        ]
        if len(rim) and rim[:, 0].max() >= far_face:
            lead = rim[rim[:, 0] >= far_face - 0.0005]
            return float(angle), tcp_z - float(np.median(lead[:, 2])), spec.extent_z - clearance
    raise AssertionError(f"the jaw never reached {spec.name}'s rim")


_CLOUD_CACHE: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def cached_clouds(angle: float) -> tuple[np.ndarray, np.ndarray]:
    """:func:`jaw_clouds`, memoised on a 0.1 mrad grid (the sweeps below reuse angles)."""
    key = round(float(angle), 4)
    if key not in _CLOUD_CACHE:
        _CLOUD_CACHE[key] = jaw_clouds(key)
    return _CLOUD_CACHE[key]


def _penetration(spec, moving: np.ndarray, lateral: float) -> np.ndarray:
    """How far each moving-jaw vertex is inside `spec`'s side wall, metres."""
    if spec.shape == "cylinder":
        return spec.radius - np.hypot(moving[:, 0] - lateral, moving[:, 1])
    half = 0.5 * spec.grasp_width_m
    return np.minimum(half - np.abs(moving[:, 0] - lateral), half - np.abs(moving[:, 1]))


def closing_contact(spec, tcp_z: float, band: tuple[float, float]) -> tuple[float, float]:
    """First jaw angle whose moving-jaw material enters `spec`, sweeping shut.

    Args:
        spec: Object, standing on the table at :func:`pad_lateral_offset`.
        tcp_z: TCP height above the table, metres.
        band: ``(low, high)`` world-height window to look for contact in --
            the whole object, or just the slice the pads occupy.

    Returns:
        ``(angle, contact height above the table)`` in radians and metres.
    """
    lateral = expert_mod.pad_lateral_offset(spec)
    for angle in np.arange(0.9, -0.175, -0.002):
        _, moving = cached_clouds(float(angle))
        heights = tcp_z - moving[:, 2]
        inside = np.where(
            (heights >= band[0]) & (heights <= band[1]),
            _penetration(spec, moving, lateral),
            -1.0,
        )
        deepest = int(inside.argmax())
        if inside[deepest] > 0.0:
            return float(angle), float(heights[deepest])
    raise AssertionError(f"the closing jaw never reached {spec.name}")


def jaw_lead(spec, tcp_z: float) -> tuple[float, float]:
    """How early the closing jaw catches `spec`, and where, off the meshes.

    Returns ``(lead, height)``: how much further the jaws would still have had
    to close for the *pads* to reach the object, at the moment anything on the
    moving finger first touches it (metres of jaw gap), and the height above the
    table that first touch happens at. Zero lead is a hand that meets the object
    with its pads; a positive lead is a one-sided push somewhere else on the
    finger, and :func:`jaw_lead`'s whole point is that it grows with how far the
    object stands above the TCP.
    """
    tips = tcp_z - expert_mod.JAW_TIP_Z
    at_pads, _ = closing_contact(spec, tcp_z, (tips, tips + 0.004))
    leading, height = closing_contact(spec, tcp_z, (0.0, spec.top_z))
    return (leading - at_pads) * objects.JAW_WIDTH_PER_RAD, height


def test_the_closing_jaw_leans_in_and_that_is_where_the_reach_constant_comes_from():
    """JAW_PARALLEL_REACH, re-measured: the lead grows with height above the TCP.

    A column of the reference width is stood at the planned lateral offset and
    made taller and taller. While its top is inside the constant the closing jaw
    still meets it within a fifth of a millimetre of where it meets the 30 mm
    cube (which grasps); past the constant the finger's upper lobe takes over
    and the lead more than doubles. That step is the whole justification for
    :func:`~manus.expert.grasp_height` refusing to leave a tall object standing
    above it.
    """
    import dataclasses

    from manus.expert import JAW_PARALLEL_REACH

    tcp_z = 0.020
    leads = {}
    for reach in (0.008, 0.011, 0.016, JAW_PARALLEL_REACH, 0.024, 0.028):
        height = tcp_z + reach
        column = dataclasses.replace(
            width_probe(objects.REFERENCE_WIDTH_M, height),
            half_extents=(0.015, 0.015, 0.5 * height),
            spawn_z=0.5 * height,
        )
        leads[round(reach * 1e3)] = round(jaw_lead(column, tcp_z)[0] * 1e3, 2)
    # Flat while the column stays inside the reach, then a step.
    inside = [leads[8], leads[11], leads[16], leads[round(JAW_PARALLEL_REACH * 1e3)]]
    assert max(inside) - min(inside) < 0.3, leads
    assert leads[24] > max(inside) + 0.2, leads
    assert leads[28] > 2 * max(inside), leads


def test_the_cylinder_was_toppled_by_the_lean_and_the_raise_is_what_answers_it():
    """The Step 21 cylinder failure, and the Step 22 grasp height, both measured.

    At its old mid-height grasp the 60 mm cylinder stood 26 mm above the TCP --
    6 mm past :data:`~manus.expert.JAW_PARALLEL_REACH` -- so the closing finger
    reached it 2 mm before the pads did, 26 mm above its centre of mass. That is
    a topple push on an object that tips at ``atan(15/30)`` = 26.6 deg, and the
    preview shows it toppling (``runs/object_previews/cylinder_3cm_demo.json``:
    ``not_in_hand``, the object riding the shut jaws). The raised grasp puts the
    first contact back near the pads, with less lead than the cube's own.

    Measured against :data:`TOP_CYLINDER`, because the cylinder is now grasped
    from the side and Step 23's claim is that this whole problem is the
    *top-down* geometry's -- which means the top-down numbers have to still be
    exactly these. What the side grasp does with them is
    :func:`test_the_side_grasp_has_no_lean_to_answer`.
    """
    from manus.expert import grasp_height

    CYLINDER = TOP_CYLINDER  # noqa: N806 -- the object this failure belonged to
    old_tcp = CYLINDER.spawn_z + TCP_TO_PAD_CENTRE
    new_tcp = grasp_height(CYLINDER) + TCP_TO_PAD_CENTRE
    assert new_tcp - old_tcp == pytest.approx(0.006, abs=5e-4)

    old_lead, old_height = jaw_lead(CYLINDER, old_tcp)
    new_lead, new_height = jaw_lead(CYLINDER, new_tcp)
    cube_lead, cube_height = jaw_lead(CUBE, grasp_height(CUBE) + TCP_TO_PAD_CENTRE)

    assert old_lead == pytest.approx(0.0020, abs=3e-4)
    assert old_height - CYLINDER.spawn_z == pytest.approx(0.026, abs=1e-3)
    assert new_lead < cube_lead < old_lead
    assert new_height < old_height - 0.006
    # ... and the cube's own numbers, which are what "acceptable" means here.
    assert cube_lead == pytest.approx(0.0010, abs=3e-4)
    assert cube_height - CUBE.spawn_z == pytest.approx(0.006, abs=1e-3)


def test_the_die_is_met_by_the_pads_so_its_failure_is_not_the_cylinders():
    """The reconciliation: the 16 mm die's approach geometry is the catalogue's best.

    Round 1 read the die's ``no_grasp`` as the pads closing above it. The meshes
    say the opposite -- the die stands 4 mm above its TCP, a quarter of
    :data:`~manus.expert.JAW_PARALLEL_REACH`, so the closing finger reaches it
    within 0.2 mm of where the pads do, the smallest lead of any object in the
    catalogue. Whatever loses the die happens after contact, not on the way to
    it.
    """
    from manus.expert import JAW_PARALLEL_REACH, grasp_height

    die_tcp = grasp_height(DIE) + TCP_TO_PAD_CENTRE
    assert DIE.top_z - die_tcp == pytest.approx(0.004, abs=1e-4)
    assert DIE.top_z - die_tcp < 0.25 * JAW_PARALLEL_REACH
    lead, height = jaw_lead(DIE, die_tcp)
    assert lead < 0.0003, f"the die is caught {lead * 1e3:.2f} mm before its pads"
    assert lead < jaw_lead(CUBE, grasp_height(CUBE) + TCP_TO_PAD_CENTRE)[0]
    assert height - DIE.spawn_z == pytest.approx(0.0064, abs=1e-3)
    # And the pads really are centred on it: the round-1 claim that survives.
    assert grasp_height(DIE) == DIE.spawn_z
    assert die_tcp - expert_mod.JAW_TIP_Z - expert_mod.MIN_TIP_CLEARANCE == pytest.approx(
        0.0007, abs=1e-4
    )


def test_the_tip_clearance_override_cannot_lower_a_grasp_only_raise_it():
    """Why the die cannot be given "the pads' lower band" through tip_clearance_m.

    :func:`~manus.expert.grasp_height` is a max of three bars, so the clearance
    override can only ever push a grasp *up*. Round 2 considered dropping the
    die's clearance to 3 mm to buy 2 mm of engagement; it buys nothing, because
    the die's own mid-height is already 0.7 mm above the 5 mm bar and 2.7 mm
    above a 3 mm one.
    """
    import dataclasses

    from manus.expert import grasp_height

    for clearance in (0.001, 0.003, 0.005):
        lowered = dataclasses.replace(DIE, tip_clearance_m=clearance)
        assert grasp_height(lowered) == DIE.spawn_z


def test_the_tip_clearance_override_moves_a_short_objects_grasp():
    """The knob the rented box sweeps the puck's 3-7 mm band with, end to end.

    A spec-level override rather than a module constant, so ``demo_expert.py
    --tip-clearance`` changes the plan and nothing else -- and so it cannot
    reach an object it was not aimed at.
    """
    import dataclasses

    from manus.expert import MIN_TIP_CLEARANCE, grasp_height, pregrasp_height, tip_clearance

    assert tip_clearance(PUCK) == MIN_TIP_CLEARANCE
    for clearance in (0.003, 0.005, 0.007):
        swept = dataclasses.replace(PUCK, tip_clearance_m=clearance)
        tcp_z = grasp_height(swept) + TCP_TO_PAD_CENTRE
        assert tcp_z - expert_mod.JAW_TIP_Z == pytest.approx(clearance)
        plan = plan_grasp(swept, (0.19, 0.0, 0.0))
        assert plan.ok and plan.tcp_grasp[2] == pytest.approx(tcp_z)
        assert plan.tcp_pregrasp[2] == pytest.approx(pregrasp_height(swept))
    # A taller object cannot feel it: its pads centre on the object instead.
    assert grasp_height(dataclasses.replace(CUBE, tip_clearance_m=0.007)) == CUBE.spawn_z


def test_the_puck_cannot_be_fixed_by_grasp_height_alone():
    """The measurement behind ``puck_d40x10.experimental``.

    The preview run's puck was punted off the table by the closing jaw. The
    reason is in the sweep, not in the hover: the moving finger arrives tilted
    and still descending (its tip drops ~16 mm per radian around the puck's
    0.33 rad contact angle), so it touches the top edge of the 10 mm rim first
    and drags it inward and down. Raising or lowering the grasp does not move
    that contact -- over the whole feasible 3-7 mm tip-clearance band it stays
    within 1.4 mm of the puck's top face -- it only trades away the pads'
    purchase on the rim, from 7 mm down to 3 mm. Both ends of the band are bad,
    which is the definition of not fixable by height.
    """
    for clearance_mm in (3.0, 4.0, 5.0, 6.0, 7.0):
        angle, height, purchase = first_rim_contact(PUCK, clearance_mm * 1e-3)
        assert PUCK.top_z - height < 0.0015, (
            f"{clearance_mm} mm: first contact {height * 1e3:.2f} mm up a 10 mm rim"
        )
        assert purchase == pytest.approx(PUCK.extent_z - clearance_mm * 1e-3)
        # ... and the finger is still on its way down when it gets there.
        before = jaw_clouds(angle + 0.02)[1][:, 2].max()
        after = jaw_clouds(angle - 0.02)[1][:, 2].max()
        assert after > before, f"{clearance_mm} mm: the jaw is rising at contact"
    assert first_rim_contact(PUCK, 0.007)[2] <= 0.003  # the "pads on 3 mm of rim" end
    assert PUCK.experimental


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
    # Against the object's *own* declared squeeze, so the die's deliberately
    # lighter one is checked as what it is rather than as a discrepancy.
    squeeze = contact_angle(spec) - spec.close_target_rad
    assert squeeze == pytest.approx(spec.squeeze_rad, abs=0.02), spec.name
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


@pytest.mark.parametrize("spec", TOP_OBJECTS, ids=TOP_IDS)
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


def test_the_convergence_bar_is_scaled_by_the_object_and_pinned_per_object():
    """Pinned: the cube's bar is CONVERGE_TOL exactly, and narrow objects get less.

    The cube is the width the number was tuned at, so ``min(1, w / 0.030)`` is
    exactly 1 for it and its behaviour is bit-identical -- which matters,
    because 0.02 rad is what the 200-attempt Step 8 gate ran under. Only the two
    objects narrower than the reference are tightened.
    """
    from manus.expert import CONVERGE_TOL, SIDE_CONVERGE_TOL, converge_tol

    assert converge_tol(CUBE) == CONVERGE_TOL
    assert {
        name: round(converge_tol(spec), 6) for name, spec in OBJECTS.items()
    } == {
        "cube_3cm": 0.02,
        # The side grasp is off the width scale entirely: its bar comes out of
        # the hand's table clearance instead (SIDE_CONVERGE_TOL).
        "cylinder_3cm": SIDE_CONVERGE_TOL,
        "die_16mm": round(0.02 * 16 / 30, 6),
        "domino_20x40": round(0.02 * 20 / 30, 6),
        "puck_d40x10": 0.02,
        "puck_d40x20": 0.02,
        "pingpong_40mm": 0.02,
        "duplo_32x64": 0.02,
    }
    # Never loosened past the tuned bar, however wide the object.
    assert max(converge_tol(spec) for spec in OBJECTS.values()) == CONVERGE_TOL
    # It rides the config, so --converge-tol still sweeps the whole catalogue.
    assert converge_tol(DIE, ExpertConfig(converge_tol=0.04)) == pytest.approx(
        2 * converge_tol(DIE)
    )
    assert converge_tol(None) == CONVERGE_TOL


def test_a_narrow_object_holds_descend_until_it_is_inside_its_own_tighter_bar():
    """The scaled bar is the one the FSM actually exits on, not just a number.

    Against the same drooping plant, the die's DESCEND runs longer than the
    cube's and lands inside the die's tolerance -- which the cube's exit would
    not have satisfied. That extra time is the point: it is the droop
    integrator's, and it is what shrinks the residual CLOSE then freezes.
    """
    from manus.expert import converge_tol

    def descend(spec):
        expert = ScriptedGraspExpert(spec, (0.20, 0.0, 0.0))
        run(expert, FakeArm(droop=0.05))
        return [report for report in expert.reports if report.state == DESCEND][0]

    cube, die = descend(CUBE), descend(DIE)
    assert die.joint_error < converge_tol(DIE) < cube.joint_error < converge_tol(CUBE)
    assert die.steps > cube.steps


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


def close_commands(spec, config) -> list[float]:
    """Jaw commands issued during CLOSE, against a plant that stops at contact."""
    expert = ScriptedGraspExpert(spec, (0.20, 0.0, 0.0), config=config)
    plant = FakeArm(jaw_stop=spec.contact_angle_rad)
    measured = plant.q.copy()
    commands = []
    while not expert.done:
        targets = expert.step(measured)
        if expert.state == CLOSE:
            commands.append(targets["gripper"])
        measured = plant.apply(targets)
    return commands


@pytest.mark.parametrize(
    "spec,expected", [(CUBE, 60), (OBJECTS["die_16mm"], 150)], ids=["cube_60g", "die_5g"]
)
def test_close_ramps_over_the_length_the_objects_mass_asks_for(spec, expected):
    """The 5 g die's jaws come in over 150 steps, the 60 g cube's over 60.

    Behavioural, not just arithmetic: the ramp is what the FSM interpolates the
    jaw command over, and CLOSE cannot end before it has finished, so a
    per-object ramp is visible in both the step size and the state length.
    """
    config = ExpertConfig()
    assert config.ramp_steps(CLOSE, spec) == expected
    commands = close_commands(spec, config)
    assert len(commands) >= expected
    stride = (spec.close_target_rad - GRIPPER_OPEN) / expected
    assert commands[1] - commands[0] == pytest.approx(stride, abs=1e-4)


def test_an_explicit_close_ramp_overrides_the_mass_rule():
    """What ``demo_expert.py --close-ramp`` is for: one ramp on whatever is grasped."""
    config = ExpertConfig(close_ramp=25)
    assert config.ramp_steps(CLOSE, OBJECTS["die_16mm"]) == 25
    assert config.ramp_steps(CLOSE, CUBE) == 25
    assert ExpertConfig().ramp_steps(CLOSE, None) == objects.CLOSE_RAMP_REFERENCE_STEPS
    assert len(close_commands(CUBE, config)) < len(close_commands(CUBE, ExpertConfig()))


def test_the_telemetry_reports_the_ramp_that_was_used():
    expert = ScriptedGraspExpert(OBJECTS["die_16mm"], (0.20, 0.0, 0.0))
    assert expert.telemetry()["close_ramp"] == 150


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


# --- What the filmed previews actually measured -------------------------------------
#
# runs/object_previews/<name>_demo.json is the Step 21 catalogue run: one
# attempt per object, at one shared placement, with per-state convergence
# reports. It records |measured - waypoint| per joint at each state's exit and
# the droop bias in force there -- and the bias is the *signed* error a step
# earlier (see ScriptedGraspExpert._update_bias), which is enough to reconstruct
# the pose CLOSE froze the arm at and put it back through the FK. These tests
# are the reconciliation of the round-1 reading of that run.

PREVIEW_DIR = kinematics.SO101_URDF_PATH.parents[3] / "runs" / "object_previews"
"""Where the Step 21 per-object previews live (repo root / runs/object_previews)."""

PREVIEW_GRASP_HEIGHT_M = {"cube_3cm": 0.015, "die_16mm": 0.008, "cylinder_3cm": 0.030}
"""Grasp height each preview was flown at, metres: every object's own mid-height.

Spelled out rather than read from :func:`~manus.expert.grasp_height`, because
the cylinder's was first raised to 36 mm and then, at Step 23, moved to a side
grasp; these files record the run before either. :func:`preview_spec` does the
same for the mode.
"""


def preview_spec(name: str) -> ObjectSpec:
    """The catalogue spec as the preview run had it: every preview was top-down."""
    return dataclasses.replace(OBJECTS[name], grasp_mode="top")


def preview(name: str) -> dict:
    """The one attempt in ``runs/object_previews/<name>_demo.json``."""
    import json

    path = PREVIEW_DIR / f"{name}_demo.json"
    if not path.is_file():
        pytest.skip(f"{path} is not checked out")
    return json.loads(path.read_text(encoding="utf-8"))[0]


def descend_exit_pose(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(waypoint, measured)`` arm poses at the preview's DESCEND exit, radians.

    The waypoint is re-solved from the recorded draw and grasp yaw (and checked
    against the recorded lift rise, which is a function of it). The measured
    pose is the waypoint minus the recorded per-joint error, signed by the
    direction the droop integrator was pulling -- ``bias(DESCEND) -
    bias(PREGRASP)`` is ``droop_gain`` times the signed error on DESCEND's one
    post-ramp step.
    """
    from manus.expert import plan_lift, tcp_target

    spec = preview_spec(record["telemetry"]["object"])
    draw, states = record["draw"], {s["state"]: s for s in record["telemetry"]["states"]}
    yaw = record["telemetry"]["grasp_yaw"]
    target = tcp_target(
        (draw["object_x"], draw["object_y"]),
        PREVIEW_GRASP_HEIGHT_M[spec.name],
        yaw,
        spec,
    )
    waypoint, converged = kinematics.ik_solve(target, yaw)
    assert converged
    assert plan_lift(waypoint, ExpertConfig().lift_rise)[1] == pytest.approx(
        record["telemetry"]["lift_rise"]
    )
    signs = np.sign(np.array(states["DESCEND"]["bias"]) - np.array(states["PREGRASP"]["bias"]))
    error = signs * np.array(states["DESCEND"]["joint_errors"])
    return waypoint, waypoint - error


@pytest.mark.parametrize("name", list(PREVIEW_GRASP_HEIGHT_M))
def test_the_previews_descend_residual_is_lateral_not_vertical(name):
    """The round-1 reconciliation: the 4-6 mm of DESCEND error is not a high hand.

    Round 1 read the die's ``no_grasp`` as the pads sitting 4-6 mm high on a
    16 mm object, on the strength of the TCP error the previews report at the
    end of DESCEND. Put the recorded pose back through the FK and the error is
    real but mostly horizontal: 5.3-5.9 mm total, of which 4.9-5.4 mm is
    horizontal and only 2.2-2.5 mm vertical -- and that part points *down*, so
    the hand is low at the grasp, not high. The reconstruction is not free to be
    wrong about this: the signed
    pose it builds reproduces the tcp_error the run recorded to a micron, which
    32 sign combinations could not all do.
    """
    record = preview(name)
    reported = {s["state"]: s for s in record["telemetry"]["states"]}["DESCEND"]["tcp_error"]
    waypoint, measured = descend_exit_pose(record)
    offset = CHAIN.fk_tcp(measured)[0] - CHAIN.fk_tcp(waypoint)[0]

    assert float(np.linalg.norm(offset)) == pytest.approx(reported, abs=1e-6)
    assert 0.0053 <= reported <= 0.0060, f"{name}: {reported * 1e3:.2f} mm"
    assert -0.0026 < offset[2] < -0.0021, f"{name}: {offset[2] * 1e3:+.2f} mm vertical"
    # Horizontal error is more than twice the vertical one on every preview.
    assert float(np.linalg.norm(offset[:2])) > 2.0 * abs(offset[2])


def test_the_previews_low_hand_still_clears_the_table_and_still_lost_the_die():
    """What the measured 2.4 mm of sag costs each object, and why it is not the die's story.

    It is spent out of the fingertip clearance, so it bites hardest on the
    shortest grasp: the die keeps 3.3 mm of its nominal 5.7 mm, the cube 10.2 of
    12.7. Both still clear the table -- and the die's engaged pad band gets
    *longer*, not shorter, which is the opposite of the round-1 reading and the
    reason "grasp the die lower" cannot be the fix. It already was lower.
    """
    from manus.expert import JAW_TIP_Z

    flown = {}
    for name in PREVIEW_GRASP_HEIGHT_M:
        waypoint, measured = descend_exit_pose(preview(name))
        tcp_z = float(CHAIN.fk_tcp(measured)[0][2])
        flown[name] = tcp_z - JAW_TIP_Z
        assert tcp_z - JAW_TIP_Z > 0.0, f"{name}: the fingertips reached the table"
    assert flown["die_16mm"] == pytest.approx(0.0033, abs=3e-4)
    assert flown["cube_3cm"] == pytest.approx(0.0102, abs=3e-4)
    # The die was flown 2.4 mm below its plan and still came out no_grasp.
    assert preview("die_16mm")["outcome"] == "no_grasp"
    assert flown["die_16mm"] < 0.008 - JAW_TIP_Z + TCP_TO_PAD_CENTRE


def test_the_previews_residual_is_a_bigger_share_of_a_narrow_grasp():
    """Why the convergence bar is scaled by the object: the same error, three objects.

    The residual is a property of the arm and the ramp, not of what is being
    grasped -- all three previews land within 0.6 mm of each other -- so what
    changes across the catalogue is only what fraction of the object it is.
    """
    from manus.expert import converge_tol

    share = {}
    for name in PREVIEW_GRASP_HEIGHT_M:
        record = preview(name)
        reported = {s["state"]: s for s in record["telemetry"]["states"]}["DESCEND"]["tcp_error"]
        share[name] = reported / OBJECTS[name].grasp_width_m
    assert share["cube_3cm"] == pytest.approx(0.20, abs=0.01)
    assert share["cylinder_3cm"] == pytest.approx(0.18, abs=0.01)
    assert share["die_16mm"] == pytest.approx(0.37, abs=0.01)
    # ... which is what the scaled tolerance is holding roughly constant.
    assert converge_tol(OBJECTS["die_16mm"]) / converge_tol(OBJECTS["cube_3cm"]) == pytest.approx(
        OBJECTS["die_16mm"].grasp_width_m / OBJECTS["cube_3cm"].grasp_width_m
    )


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
#
# The predicate takes what the driver measures -- the object's position and the
# six joints -- so the fixtures below build both from a real plan: the arm at
# its lift pose, and the object wherever the caller says relative to the pads.

RECORDED_STALL_RAD = {
    "cube_3cm": 0.18908,
    "cylinder_3cm": -0.17453,  # rode the jaw to the hard stop: JAWS EMPTY
    "die_16mm": -0.02707,
    "domino_20x40": 0.05329,
    "duplo_32x64": 0.21495,
    "pingpong_40mm": 0.23276,
    "puck_d40x10": 0.18839,  # sat on the commanded target: JAWS EMPTY
    # Not filmed -- the respec is Step 23 and the preview run is Step 21. The
    # 40 mm disc's own contact angle stands in, which is what a held one has to
    # stall at, and it is the same disc so it is the same angle as the thin
    # puck's *nominal* one rather than the empty-jaw reading recorded above.
    "puck_d40x20": round(THICK_PUCK.contact_angle_rad, 5),
}
"""HOLD-phase jaw angle of each filmed preview, from ``runs/object_previews/*_demo.json``.

Every one of these attempts was scored a success by the old height-only
predicate. Two of them were not grasps at all -- the object was riding the
robot -- which is what the stall clause is for.
"""


def lift_pose(spec, stall: float, placement=None):
    """``(joints, object_pos)`` for `spec` held in the pads at the lift pose.

    The object is placed where a seated one sits -- at the pad centre in the
    tool's own frame -- so the in-hand clause sees a real grasp geometry and the
    stall clause sees `stall`.
    """
    plan = plan_grasp(spec, a_placement(spec) if placement is None else placement)
    assert plan.ok, plan.reason
    position, rotation = CHAIN.fk_tcp(plan.q_lift)
    seated = position + rotation @ np.array(
        [expert_mod.pad_lateral_offset(spec), 0.0, TCP_TO_PAD_CENTRE]
    )
    return np.concatenate([plan.q_lift, [stall]]), seated


def hold(monitor, joints, object_pos, steps=40):
    """Feed one steady state into `monitor` for `steps` steps."""
    for _ in range(steps):
        monitor.update(object_pos, joints)
    return monitor


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_the_predicate_reproduces_what_the_previews_actually_showed(spec):
    """The three recorded cases, and the four alongside them, on one truth table.

    Every filmed preview passed the height-only predicate. Fed the same jaw
    angle with the object seated in the pads, the hardened one keeps the five
    real grasps and rejects the two that were riding the robot -- the cylinder,
    whose jaws ran to their -0.1745 rad hard stop, and the puck, whose jaws
    stopped exactly on the 0.1876 rad target they were commanded to.
    """
    stall = RECORDED_STALL_RAD[spec.name]
    joints, seated = lift_pose(spec, stall)
    monitor = hold(GraspSuccessMonitor(spec), joints, seated)
    empty_handed = spec.name in {"cylinder_3cm", "puck_d40x10"}
    assert monitor.success is not empty_handed, (
        f"{spec.name}: stall {stall:.5f} against band {monitor.stall_band}"
    )
    # Either way the object was up: the old predicate said yes to all seven.
    assert monitor.height_only


def test_the_ball_is_the_tightest_pass_and_the_puck_the_nearest_miss():
    """Pinned margins, because STALL_SLACK_RAD is a judgement call between them.

    The ping-pong ball's pads sink into a tangent point and it stalls 94 mrad
    below its contact angle -- 6 mrad inside the band. The puck misses by 38.
    Anyone widening the slack should know it is 6 mrad from being unable to
    separate the two.
    """
    for name, expected in (("pingpong_40mm", 0.0062), ("puck_d40x10", -0.0382)):
        monitor = GraspSuccessMonitor(OBJECTS[name])
        margin = RECORDED_STALL_RAD[name] - monitor.stall_band[0]
        assert margin == pytest.approx(expected, abs=5e-4)


def test_an_object_riding_the_arm_is_not_in_the_hand():
    """The geometric half: high, jaws plausibly shut, but 25 cm from the pads."""
    joints, seated = lift_pose(CUBE, RECORDED_STALL_RAD["cube_3cm"])
    monitor = hold(GraspSuccessMonitor(CUBE), joints, seated + np.array([0.25, 0.0, 0.0]))
    assert not monitor.success and monitor.height_only
    assert monitor.tcp_distance > monitor.in_hand_radius


def test_the_in_hand_bar_is_generous_to_a_real_grasp():
    """A seated object sits ~17 mm from the TCP; the bar is 60."""
    joints, seated = lift_pose(CUBE, RECORDED_STALL_RAD["cube_3cm"])
    monitor = hold(GraspSuccessMonitor(CUBE), joints, seated)
    assert monitor.tcp_distance == pytest.approx(0.0175, abs=2e-3)
    assert monitor.success


def test_jaws_that_reach_the_commanded_target_are_empty_however_light_the_squeeze():
    """The die's band would degenerate without the target clause, so it is checked there.

    Its squeeze is only as big as the stall slack, so "within a slack of
    contact" alone would accept the empty closure that lands on the target.
    """
    die = OBJECTS["die_16mm"]
    assert die.squeeze_rad <= expert_mod.STALL_SLACK_RAD  # the degenerate case
    joints, seated = lift_pose(die, die.close_target_rad)
    assert not hold(GraspSuccessMonitor(die), joints, seated).success
    assert GraspSuccessMonitor(die).stall_band[0] == pytest.approx(
        die.close_target_rad + expert_mod.STALL_TARGET_MARGIN_RAD
    )


def test_jaws_stalled_far_above_contact_are_jammed_on_something_else():
    joints, seated = lift_pose(CUBE, 0.9)
    monitor = hold(GraspSuccessMonitor(CUBE, gripper_max=1.2), joints, seated)
    assert not monitor.success


def test_success_needs_the_height_the_sustain_and_the_jaws():
    joints, seated = lift_pose(CUBE, RECORDED_STALL_RAD["cube_3cm"])
    monitor = GraspSuccessMonitor(CUBE, sustain=30)
    for _ in range(29):
        assert not monitor.update(seated, joints)
    assert monitor.update(seated, joints)
    assert monitor.success and monitor.best_streak >= 30


def test_a_dropped_object_restarts_the_count():
    joints, seated = lift_pose(CUBE, RECORDED_STALL_RAD["cube_3cm"])
    dropped = np.array([seated[0], seated[1], CUBE.spawn_z])
    monitor = GraspSuccessMonitor(CUBE, sustain=10)
    hold(monitor, joints, seated, steps=9)
    monitor.update(dropped, joints)
    hold(monitor, joints, seated, steps=9)
    assert not monitor.success and monitor.streak == 9


def test_an_open_gripper_never_succeeds_however_high_the_object():
    joints, seated = lift_pose(CUBE, GRIPPER_OPEN)
    monitor = hold(GraspSuccessMonitor(CUBE, sustain=5), joints, seated + np.array([0, 0, 0.1]))
    assert not monitor.success
    assert monitor.peak_z == pytest.approx(seated[2] + 0.1)


def test_the_bar_is_the_spawn_height_plus_the_lift():
    monitor = GraspSuccessMonitor(CYLINDER, lift=0.05)
    assert monitor.threshold_z == pytest.approx(0.08)
    joints, seated = lift_pose(CYLINDER, CYLINDER.contact_angle_rad)
    just_short = np.array([seated[0], seated[1], 0.0799])
    assert not hold(monitor, joints, just_short).success


def test_success_latches():
    joints, seated = lift_pose(CUBE, RECORDED_STALL_RAD["cube_3cm"])
    monitor = GraspSuccessMonitor(CUBE, sustain=2)
    hold(monitor, joints, seated, steps=2)
    assert monitor.success
    dropped = np.array([seated[0], seated[1], 0.0])
    monitor.update(dropped, np.concatenate([joints[:5], [GRIPPER_OPEN]]))
    assert monitor.success and monitor.to_dict()["success"] is True


def test_the_monitor_summary_is_json_ready_and_carries_the_evidence():
    import json

    joints, seated = lift_pose(CUBE, RECORDED_STALL_RAD["cube_3cm"])
    payload = json.loads(json.dumps(hold(GraspSuccessMonitor(CUBE), joints, seated).to_dict()))
    assert payload["success"] is True and payload["height_only"] is True
    assert payload["tcp_distance"] < payload["in_hand_radius"]
    assert payload["stall_band"][0] < payload["gripper"] < payload["stall_band"][1]


def test_a_measurement_that_is_not_a_position_is_rejected():
    joints, _ = lift_pose(CUBE, 0.2)
    with pytest.raises(ValueError, match="object centre"):
        GraspSuccessMonitor(CUBE).update(0.10, joints)


# --- Failure taxonomy ---------------------------------------------------------------


def finished(placement=(0.20, 0.0, 0.0), **kwargs) -> ScriptedGraspExpert:
    """An expert that has run a whole attempt against a converging plant."""
    expert = fresh(placement, **kwargs)
    run(expert, FakeArm(jaw_stop=0.5))
    return expert


def fed(heights, stall=None, spec=CUBE) -> GraspSuccessMonitor:
    """Feed a height trace into a monitor, with the object seated in the pads.

    Only the height varies: the arm and the object's lateral position are a
    real grasp's, so an attempt fails on the clause the trace is testing rather
    than on an artefact of the fixture.
    """
    joints, seated = lift_pose(spec, RECORDED_STALL_RAD[spec.name] if stall is None else stall)
    monitor = GraspSuccessMonitor(spec)
    for height in heights:
        monitor.update(np.array([seated[0], seated[1], height]), joints)
    return monitor


def test_a_met_predicate_is_a_success():
    assert classify_outcome(finished(), fed([0.10] * 40)) == "success"


def test_an_unreachable_placement_is_named_as_such():
    expert = fresh((0.60, 0.0, 0.0))
    run(expert, FakeArm(jaw_stop=0.5), max_steps=6000)
    assert not expert.plan.ok
    assert classify_outcome(expert, fed([0.015] * 40)) == "ik_infeasible"


def test_an_object_that_stayed_up_without_being_held_gets_its_own_name():
    """The preview run's two false successes, as the gate report will see them.

    Height met, sustain met, jaws shut on nothing: not a slip (nothing was ever
    held to slip) and not a short lift (it went all the way up).
    """
    monitor = fed([0.10] * 40, stall=OBJECTS["cylinder_3cm"].close_target_rad)
    assert not monitor.success and monitor.height_only
    assert classify_outcome(finished(), monitor) == "not_in_hand"


def test_over_the_bar_but_not_held_is_a_slip():
    assert classify_outcome(finished(), fed([0.10] * 10 + [0.015] * 40)) == "slipped"


def test_lifted_but_short_of_the_bar_is_a_short_lift():
    assert classify_outcome(finished(), fed([0.05] * 40)) == "short_lift"


def test_an_untouched_object_after_a_clean_run_is_a_missed_grasp():
    expert = finished()
    assert expert.timeouts == []
    assert classify_outcome(expert, fed([0.015] * 40)) == "no_grasp"


def test_an_untouched_object_after_a_wedged_run_is_a_timeout():
    expert = fresh(state_budget=30, hold_steps=5)
    run(expert, FakeArm(frozen=True), max_steps=1000)
    assert classify_outcome(expert, fed([0.015] * 40)) == "timeout"


# --- The driver's output names ---------------------------------------------------


def test_the_demo_driver_names_its_artefacts_after_the_object():
    """``demo_expert.py`` cannot overwrite one object's preview with another's.

    A source check rather than an import, because the script parses its
    arguments and starts Isaac Sim at module scope, so there is nothing to
    import on the CPU-only side. What it pins is the thing that actually broke:
    with a fixed ``demo.json`` and an ``expert_demo_0000.mp4``, filming the
    seven-object catalogue into one ``--out-dir`` left one summary and one video
    -- the last object's. Both basenames now carry ``--object``, and ``--label``
    still overrides the video outright, which is what the committed
    ``runs/object_previews/<name>.mp4`` names were made with.
    """
    source = (PREVIEW_DIR.parent.parent / "scripts" / "demo_expert.py").read_text(
        encoding="utf-8"
    )
    assert 'f"{args_cli.object}_{attempt:04d}"' in source
    assert 'f"{args_cli.object}_demo.json"' in source
    assert 'f"{args_cli.object}_tuning.json"' in source
    assert 'f"{args_cli.object}_{slot}"' in source
    assert 'f"{args_cli.label or name}.mp4"' in source
    # ... and nothing writes the old fixed names any more.
    assert '"demo.json"' not in source
    assert '"tuning.json"' not in source


# --- The side grasp ---------------------------------------------------------------
#
# The cup grasp, re-derived here end to end: where the tool stands, which way up
# the hand is, what the closing jaw meets, and what the table clearance costs.
# None of it is read back from the constants it is checking -- the mesh clouds
# and the solved FK poses are the inputs.

SIDE_ROLL = expert_mod.SIDE_GRASP_ROLL


def side_plan(spec=CYLINDER, radius: float | None = None, azimuth_deg: float = 0.0, yaw=0.9):
    """``(plan, object_xyz)`` for a side grasp at a polar spot in the side region."""
    from manus.expert import grasp_height

    region = placement_region(spec)
    radius = float(np.mean(region.radius)) if radius is None else radius
    azimuth = math.radians(azimuth_deg)
    x = region.pan_axis_xy[0] + radius * math.cos(azimuth)
    y = region.pan_axis_xy[1] + radius * math.sin(azimuth)
    plan = plan_grasp(spec, (x, y, yaw))
    assert plan.ok, plan.reason
    return plan, np.array([x, y, grasp_height(spec)])


def test_the_catalogue_has_exactly_one_side_grasp_and_it_is_the_cylinder():
    assert [spec.name for spec in SIDE_OBJECTS] == ["cylinder_3cm"]
    assert expert_mod.is_side_grasp(CYLINDER)
    assert not expert_mod.is_side_grasp(CUBE)
    assert expert_mod.state_sequence(CYLINDER) == expert_mod.SIDE_STATE_SEQUENCE
    assert expert_mod.state_sequence(CUBE) == STATE_SEQUENCE
    assert expert_mod.SIDE_STATE_SEQUENCE == (
        PREGRASP,
        expert_mod.ADVANCE,
        CLOSE,
        LIFT,
        HOLD,
        DONE,
    )


def test_a_side_grasp_cannot_be_declared_on_an_object_with_a_grasp_axis():
    """The catalogue refuses what the arm cannot do: pick the approach azimuth."""
    with pytest.raises(ValueError, match="cannot choose its approach azimuth"):
        dataclasses.replace(CUBE, grasp_mode="side")
    with pytest.raises(ValueError, match="unknown grasp_mode"):
        dataclasses.replace(CUBE, grasp_mode="diagonal")


def test_the_side_grasp_puts_the_object_between_the_pads():
    """The whole point of the plan, closed back through the FK over the region.

    The object ends up at exactly the same place in the tool's own frame as a
    top-down grasp does -- ``(pad_lateral_offset, 0, TCP_TO_PAD_CENTRE)`` -- and
    that is the claim, because it is what makes the jaws' own geometry (the
    close target, the clearance, the contact angle) carry over unchanged.
    """
    for radius in np.linspace(*placement_region(CYLINDER).radius, 4):
        for azimuth_deg in (-105.0, -40.0, 0.0, 40.0, 105.0):
            plan, object_xyz = side_plan(radius=float(radius), azimuth_deg=azimuth_deg)
            position, rotation = CHAIN.fk_tcp(plan.q_grasp)
            local = rotation.T @ (object_xyz - position)
            assert local[0] == pytest.approx(plan.lateral_offset, abs=1e-4)
            assert local[1] == pytest.approx(0.0, abs=1e-4)
            assert local[2] == pytest.approx(TCP_TO_PAD_CENTRE, abs=1e-4)


def test_the_side_grasp_is_level_and_the_hand_lies_flat():
    """Pitch sum zero, jaws level, tool +Y pointing world up."""
    plan, _ = side_plan()
    assert plan.grasp_mode == "side"
    for q in (plan.q_pregrasp, plan.q_grasp):
        assert kinematics.pitch_sum(q) == pytest.approx(
            kinematics.PITCH_SUM_HORIZONTAL, abs=1e-3
        )
        rotation = CHAIN.fk_tcp(q)[1]
        assert abs(rotation[2, 2]) < 1e-3, "the approach axis is not level"
        assert abs(rotation[2, 0]) < 1e-3, "the jaws are not closing level"
        # SIDE_GRASP_ROLL is 0, which is the branch with +Y pointing up -- the
        # one that keeps the wrist camera above the table.
        assert rotation[:, 1] == pytest.approx([0.0, 0.0, 1.0], abs=1e-3)
    assert plan.grasp_yaw == SIDE_ROLL == pytest.approx(0.0)


def test_the_side_approach_is_radial_and_the_plan_says_which_way():
    """The azimuth is reported, not requested -- and it is close to the object's own.

    The arm has no freedom left to aim the approach, so it comes out along the
    reach direction: within a couple of degrees of the object's own azimuth,
    the difference being the 17 mm the tool stands off to one side.
    """
    for azimuth_deg in (-105.0, -40.0, 0.0, 40.0, 105.0):
        plan, object_xyz = side_plan(azimuth_deg=azimuth_deg)
        _, object_azimuth = placement_region(CYLINDER).polar(*object_xyz[:2])
        delivered = CHAIN.approach_azimuth(plan.q_grasp)
        assert plan.approach_azimuth == pytest.approx(delivered, abs=1e-9)
        offset_deg = abs(math.degrees(_wrap_pi(delivered - object_azimuth)))
        assert 0.5 < offset_deg < 3.0, f"approach is {offset_deg:.2f} deg off the radius"


def _wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def test_the_azimuth_fixed_point_is_what_puts_the_object_between_the_pads():
    """One pass is millimetres out; three are sub-micron. The contraction, measured.

    Standing the tool where the *object's* azimuth says, without closing the
    loop on the azimuth the arm actually delivers, misses by the pad offset
    times the azimuth error: 0.44 mm, which is 1.5% of the object it is trying
    to centre between the pads. Each pass takes ~99% of what is left, so the
    third one is already down on the URDF's own 0.15 um noise floor.
    """
    from manus.expert import side_tcp_target

    plan, object_xyz = side_plan()
    region = placement_region(CYLINDER)
    misses = []
    azimuth = region.polar(*object_xyz[:2])[1]
    for _ in range(4):
        target = side_tcp_target(tuple(object_xyz[:2]), object_xyz[2], azimuth, CYLINDER)
        q, converged = kinematics.ik_solve(
            target, SIDE_ROLL, family=kinematics.TOOL_HORIZONTAL
        )
        assert converged
        position, rotation = CHAIN.fk_tcp(q)
        local = rotation.T @ (object_xyz - position)
        misses.append(
            float(np.linalg.norm(local - [plan.lateral_offset, 0.0, TCP_TO_PAD_CENTRE]))
        )
        azimuth = CHAIN.approach_azimuth(q)
    assert misses[0] == pytest.approx(4.0e-4, abs=1e-4), "the naive aim moved"
    assert misses[1] < 0.02 * misses[0]
    assert misses[expert_mod.SIDE_AZIMUTH_PASSES] < 1e-6
    # ... and the plan itself lands where the last pass does.
    position, rotation = CHAIN.fk_tcp(plan.q_grasp)
    local = rotation.T @ (object_xyz - position)
    assert float(
        np.linalg.norm(local - [plan.lateral_offset, 0.0, TCP_TO_PAD_CENTRE])
    ) < 1e-6


def test_the_side_pregrasp_stands_off_along_the_approach():
    """PREGRASP is 40 mm straight back, at the same height -- no hover at all.

    And that 40 mm buys a real gap: the open fingertips reach 6.3 mm past the
    TCP and the cylinder's near face is 11 mm short of it, so the hand waits
    22.7 mm clear of the object rather than over it.
    """
    plan, object_xyz = side_plan()
    _, rotation = CHAIN.fk_tcp(plan.q_grasp)
    approach = rotation[:, 2]
    offset = plan.tcp_grasp - plan.tcp_pregrasp
    assert float(offset @ approach) == pytest.approx(ExpertConfig().side_retract, abs=1e-9)
    assert float(np.linalg.norm(offset)) == pytest.approx(ExpertConfig().side_retract, abs=1e-9)
    # Level to within the 1e-5 rad the URDF's own asymmetry leaves in the
    # approach axis, i.e. microns over a 40 mm stand-off.
    assert plan.tcp_pregrasp[2] == pytest.approx(plan.tcp_grasp[2], abs=1e-5)
    assert plan.tcp_pregrasp[2] == pytest.approx(expert_mod.grasp_height(CYLINDER), abs=1e-5)

    tips = CHAIN.fk_tcp(plan.q_pregrasp)[0] + rotation[:, 2] * expert_mod.JAW_TIP_Z
    near_face = object_xyz - approach * CYLINDER.radius
    assert float((near_face - tips) @ approach) == pytest.approx(0.0227, abs=5e-4)
    # ... and the stand-off really is radial: it moves the tool *in*, not down.
    assert float(np.linalg.norm(plan.tcp_pregrasp[:2])) < float(
        np.linalg.norm(plan.tcp_grasp[:2])
    )


def side_world_cloud(plan, gripper: float) -> np.ndarray:
    """Both jaw meshes in world coordinates at a solved side-grasp pose, metres."""
    position, rotation = CHAIN.fk_tcp(plan.q_grasp)
    return np.vstack(jaw_clouds(gripper)) @ rotation.T + position


def test_side_jaw_depth_is_the_housings_half_width_not_the_fingertips():
    """SIDE_JAW_DEPTH, re-derived off the meshes, and *which part* sets it.

    Turned on its side the hand's table clearance stops being about how far the
    fingers reach past the TCP and starts being about how wide the hand is. The
    fingers are slim -- everything forward of 50 mm behind the TCP stays inside
    :data:`~manus.expert.SIDE_PAD_HALF_REACH` of the tool axis -- but the
    wrist_roll follower's housing behind them is not, and it is asymmetric:
    24.2 mm one way, 27.8 mm the other. At
    :data:`~manus.expert.SIDE_GRASP_ROLL` = 0 the tool's -Y is world down, so
    the 27.8 mm side is the one that hangs over the table.
    """
    from manus.expert import SIDE_JAW_DEPTH, SIDE_JAW_DEPTH_SHALLOW, SIDE_PAD_HALF_REACH

    sweep = [float(angle) for angle in np.arange(-0.174, 1.55, 0.02)]
    clouds = [np.vstack(jaw_clouds(angle)) for angle in sweep]
    assert -min(cloud[:, 1].min() for cloud in clouds) == pytest.approx(
        SIDE_JAW_DEPTH, abs=1e-4
    )
    assert max(cloud[:, 1].max() for cloud in clouds) == pytest.approx(
        SIDE_JAW_DEPTH_SHALLOW, abs=1e-4
    )
    assert SIDE_JAW_DEPTH - SIDE_JAW_DEPTH_SHALLOW == pytest.approx(0.0036, abs=2e-4)
    # The fingers alone would allow a grasp half as low again.
    fingers = max(
        float(np.abs(cloud[cloud[:, 2] >= -0.05][:, 1]).max()) for cloud in clouds
    )
    assert fingers == pytest.approx(SIDE_PAD_HALF_REACH, abs=5e-4)
    assert fingers < 0.45 * SIDE_JAW_DEPTH


def solved_side_pose(object_xyz, roll, spec=CYLINDER):
    """The ``(position, rotation)`` a side grasp of `spec` is solved to at `roll`."""
    from manus.expert import side_tcp_target

    azimuth = placement_region(spec).polar(*object_xyz[:2])[1]
    for _ in range(expert_mod.SIDE_AZIMUTH_PASSES + 1):
        target = side_tcp_target(tuple(object_xyz[:2]), object_xyz[2], azimuth, spec, roll)
        q, converged = kinematics.ik_solve(target, roll, family=kinematics.TOOL_HORIZONTAL)
        assert converged
        azimuth = CHAIN.approach_azimuth(q)
    return q


def test_the_side_roll_branch_is_the_one_that_keeps_the_camera_out_of_the_table():
    """Why SIDE_GRASP_ROLL is 0 and not pi, and what the choice costs.

    Both level rolls are the same grasp with the fingers swapped, so the arm is
    indifferent -- but the wrist camera is bolted to the gripper link 55 mm off
    the tool axis and rides the roll with everything else. Measured through the
    *solved* pose of each branch, at the grasp height each one would be planned
    at:

    * ``pi`` hangs the camera 55 mm **below** the tool axis. Even at the raised
      cup height that is 15 mm under the ground plane, looking 33 deg up, with
      the image upside down -- which is exactly what the filmed cylinder
      recorded.
    * ``0`` stands it 55 mm above the axis, 95 mm over the table, looking 33 deg
      down at the object, upright.

    The price is 3.6 mm of table clearance, because the two rolls put opposite
    sides of the asymmetric housing down -- and the cup height pays it back
    3.4x over (12.2 mm of clearance against the 5.8 mm the old pi-at-mid-height
    grasp had).
    """
    from manus.expert import SIDE_JAW_DEPTH, SIDE_JAW_DEPTH_SHALLOW, grasp_height

    _, object_xyz = side_plan()
    lowest, camera = {}, {}
    for roll in (0.0, math.pi):
        q = solved_side_pose(object_xyz, roll)
        position, rotation = CHAIN.fk_tcp(q)
        cloud = np.vstack(jaw_clouds(GRIPPER_OPEN)) @ rotation.T + position
        lowest[roll] = float(cloud[:, 2].min())
        camera[roll] = CHAIN.wrist_camera_pose(q)
    # The camera decides it, and it is not close.
    assert camera[0.0][0][2] == pytest.approx(0.0952, abs=1e-3)
    assert camera[math.pi][0][2] == pytest.approx(-0.0152, abs=1e-3)
    assert camera[math.pi][0][2] < 0.0, "the discarded branch buries the camera"
    assert camera[0.0][1][2, 1] == pytest.approx(0.842, abs=1e-3)  # image up, world z
    assert camera[math.pi][1][2, 1] == pytest.approx(-0.842, abs=1e-3)
    # The clearance it costs, and what the cup height buys back.
    assert lowest[0.0] == pytest.approx(0.0122, abs=5e-4)
    assert lowest[math.pi] == pytest.approx(0.0158, abs=5e-4)
    assert lowest[math.pi] - lowest[0.0] == pytest.approx(
        SIDE_JAW_DEPTH - SIDE_JAW_DEPTH_SHALLOW, abs=2e-4
    )
    assert SIDE_ROLL == 0.0
    assert grasp_height(CYLINDER) - lowest[0.0] == pytest.approx(SIDE_JAW_DEPTH, abs=5e-4)
    # ... and the branch the previews were filmed with, at the height they were
    # filmed at, is the 5.8 mm the failure was measured against.
    old = solved_side_pose(np.array([*object_xyz[:2], CYLINDER.spawn_z]), math.pi)
    position, rotation = CHAIN.fk_tcp(old)
    old_cloud = np.vstack(jaw_clouds(GRIPPER_OPEN)) @ rotation.T + position
    assert float(old_cloud[:, 2].min()) == pytest.approx(0.0058, abs=5e-4)


def test_the_side_grasp_keeps_the_hand_off_the_table_through_the_whole_close():
    """Every jaw angle, at the solved pose, by FK -- not just the open hand.

    The moving finger swings during CLOSE, so the clearance is checked over the
    sweep rather than at one angle. It does not move: the lowest thing on the
    hand is the static housing, which does not swing.
    """
    from manus.expert import MIN_TIP_CLEARANCE

    plan, _ = side_plan()
    for angle in (GRIPPER_OPEN, 0.5, 0.3, CYLINDER.contact_angle_rad, CYLINDER.close_target_rad):
        lowest = float(side_world_cloud(plan, float(angle))[:, 2].min())
        assert lowest >= MIN_TIP_CLEARANCE, f"jaws at {angle:.3f}: {lowest * 1e3:.2f} mm"
        assert lowest == pytest.approx(0.0122, abs=5e-4)


def test_the_side_grasp_height_is_the_cup_grasp_between_the_table_and_the_rim():
    """Two thirds of the way up, and both bars that bracket it, measured.

    The floor is the hand: its housing hangs :data:`~manus.expert.SIDE_JAW_DEPTH`
    below the tool axis and has to clear the table by the tip clearance, so
    nothing can be side-grasped below 32.8 mm. The ceiling is the object: the
    fingers reach :data:`~manus.expert.SIDE_PAD_HALF_REACH` above the axis, so a
    grasp within that of the top hangs half the pad band off the rim. The
    cylinder's cup height sits between them with 7.2 and 8.3 mm to spare.
    """
    from manus.expert import (
        SIDE_GRASP_HEIGHT_FRACTION,
        SIDE_JAW_DEPTH,
        SIDE_PAD_HALF_REACH,
        grasp_height,
        side_table_clearance,
        tip_clearance,
    )

    assert grasp_height(CYLINDER) == pytest.approx(SIDE_GRASP_HEIGHT_FRACTION * 0.060)
    assert grasp_height(CYLINDER) == pytest.approx(0.040)
    assert grasp_height(CYLINDER) > CYLINDER.spawn_z, "a cup is taken above its waist"

    floor = SIDE_JAW_DEPTH + tip_clearance(CYLINDER)
    ceiling = CYLINDER.top_z - SIDE_PAD_HALF_REACH
    assert floor == pytest.approx(0.0328, abs=1e-4)
    assert ceiling == pytest.approx(0.0483, abs=1e-4)
    assert grasp_height(CYLINDER) - floor == pytest.approx(0.0072, abs=1e-4)
    assert ceiling - grasp_height(CYLINDER) == pytest.approx(0.0083, abs=1e-4)
    # The clearance that buys, against the 5.8 mm the filmed failure had.
    assert side_table_clearance(CYLINDER) == pytest.approx(0.0122, abs=1e-4)
    assert side_table_clearance(CYLINDER) > 2 * 0.0058

    # A shorter cylinder is pushed up off its own cup height by the table...
    short = dataclasses.replace(
        CYLINDER, name="short", height=0.045, spawn_z=0.0225, grasp_mode="side"
    )
    assert grasp_height(short) == pytest.approx(floor)
    # ... and a much taller one is pulled down off it by its own rim only when
    # the fraction would take the pads off the top, which two thirds never does.
    tall = dataclasses.replace(
        CYLINDER, name="tall", height=0.100, spawn_z=0.050, grasp_mode="side"
    )
    assert grasp_height(tall) == pytest.approx(SIDE_GRASP_HEIGHT_FRACTION * 0.100)
    assert grasp_height(tall) < tall.top_z - SIDE_PAD_HALF_REACH


def _side_penetration(spec, cloud: np.ndarray, lateral: float) -> np.ndarray:
    """How far each vertex is inside a side-grasped cylinder, metres, tool frame.

    The object's axis is now along the tool's y (it is standing on the table and
    the tool is lying flat), so its circular section lives in the tool's (x, z)
    plane rather than its (x, y) one -- the same rotation the whole grasp is.

    The object is *not* centred on the tool axis any more, either: the grasp is
    taken at :func:`~manus.expert.grasp_height`, so in the tool's own frame the
    body runs from ``-grasp_height`` (the table, since +y is world up at
    :data:`~manus.expert.SIDE_GRASP_ROLL`) up to ``top_z - grasp_height``.
    """
    height = expert_mod.grasp_height(spec)
    inside_height = (cloud[:, 1] >= -height) & (cloud[:, 1] <= spec.top_z - height)
    radial = spec.radius - np.hypot(cloud[:, 0] - lateral, cloud[:, 2] - TCP_TO_PAD_CENTRE)
    return np.where(inside_height, radial, -1.0)


def _side_closing_contact(spec, band=None) -> tuple[float, np.ndarray]:
    """First jaw angle whose moving-jaw material enters a side-grasped `spec`."""
    lateral = expert_mod.pad_lateral_offset(spec)
    for angle in np.arange(0.9, -0.175, -0.002):
        _, moving = cached_clouds(float(angle))
        penetration = _side_penetration(spec, moving, lateral)
        if band is not None:
            penetration = np.where(
                (moving[:, 2] >= band[0]) & (moving[:, 2] <= band[1]), penetration, -1.0
            )
        deepest = int(penetration.argmax())
        if penetration[deepest] > 0.0:
            return float(angle), moving[deepest]
    raise AssertionError(f"the closing jaw never reached {spec.name}")


def test_the_side_grasp_has_no_lean_to_answer():
    """The lean that toppled the top-down cylinder does not exist side-on.

    :data:`~manus.expert.JAW_PARALLEL_REACH` bounds how far an object may stand
    *past the TCP along the tool's -z*, because that is where the moving
    finger's face leans in. Rotate the hand onto its side and the object's
    height stops pointing that way -- only its own radius does, 11 mm of it,
    half the shelf. Measured off the meshes, the closing finger reaches the
    cylinder at exactly the same angle at the pads as anywhere else: zero lead,
    against 2.0 mm at the old mid-height top-down grasp and 0.7 mm at the raised
    one, and 4.3 mm below the tool axis rather than 26 mm above the object's
    centre of mass.
    """
    from manus.expert import JAW_PARALLEL_REACH, JAW_TIP_Z, side_body_behind_tcp

    assert side_body_behind_tcp(CYLINDER) == pytest.approx(0.011)
    assert side_body_behind_tcp(CYLINDER) < 0.6 * JAW_PARALLEL_REACH

    anywhere, point = _side_closing_contact(CYLINDER)
    at_pads, _ = _side_closing_contact(CYLINDER, (-0.002, JAW_TIP_Z))
    lead = (anywhere - at_pads) * objects.JAW_WIDTH_PER_RAD
    assert lead == pytest.approx(0.0, abs=1e-4), f"the side jaw leads by {lead * 1e3:.2f} mm"
    # +y is world *down* at SIDE_GRASP_ROLL, so a positive y is below the axis.
    assert point[1] * 1e3 == pytest.approx(4.3, abs=1.0)
    # ... and the top-down cylinder's lead is still what it was.
    assert jaw_lead(TOP_CYLINDER, expert_mod.grasp_height(TOP_CYLINDER) + TCP_TO_PAD_CENTRE)[
        0
    ] > 2 * lead + 3e-4


def test_the_contact_angle_is_width_based_and_does_not_care_which_way_the_hand_points():
    """Why the close target carries over to a side grasp unchanged.

    The jaws span the same 30 mm and meet it over the same slice of the
    approach axis: a 30 mm cylinder gripped across its diameter presents its
    surface from 11 mm behind the TCP to 6.3 mm in front of it -- which is,
    to the millimetre, the band a 30 mm cube presents to a top-down grasp. So
    the mesh-measured contact angle is the same number, and
    :func:`~manus.objects.contact_angle_for_width` -- which only ever saw the
    width -- is right for both.
    """
    from manus.expert import JAW_TIP_Z, grasp_height

    top_band = engaged_band(CUBE)
    side_low = TCP_TO_PAD_CENTRE - CYLINDER.radius
    assert side_low == pytest.approx(top_band[0], abs=1e-4)
    assert JAW_TIP_Z == pytest.approx(top_band[1])

    assert contact_angle(CYLINDER) == pytest.approx(contact_angle(CUBE), abs=1e-3)
    assert CYLINDER.close_target_rad == CUBE.close_target_rad
    assert CYLINDER.contact_angle_rad == CUBE.contact_angle_rad
    # The pads really do shut on it: the same squeeze bar the catalogue holds.
    assert grasp_height(CYLINDER) > 0.0
    assert jaw_gap(CYLINDER.close_target_rad, CYLINDER) < CYLINDER.grasp_width_m - 0.005


def test_the_side_lift_carries_the_object_past_the_success_bar():
    """LIFT is unchanged code and still clears the predicate's 5 cm, everywhere."""
    from manus.expert import SUCCESS_LIFT_M

    worst = math.inf
    for radius in np.linspace(*placement_region(CYLINDER).radius, 4):
        for azimuth_deg in (-105.0, 0.0, 105.0):
            plan, _ = side_plan(radius=float(radius), azimuth_deg=azimuth_deg)
            worst = min(worst, plan.lift_rise)
    assert worst >= MIN_LIFT_RISE
    assert worst > SUCCESS_LIFT_M + 0.02, f"only {worst * 1e3:.0f} mm of lift"


def test_the_side_plan_is_deterministic_and_needs_no_current_pose():
    """No yaw branch means no dependence on where the arm happens to be."""
    placement = a_placement(CYLINDER, 0.4)
    first = plan_grasp(CYLINDER, placement)
    second = plan_grasp(CYLINDER, placement, np.array([1.0, -0.3, 0.2, 0.5, 2.0]))
    assert np.array_equal(first.q_grasp, second.q_grasp)
    assert first.grasp_yaw == second.grasp_yaw
    # ... and the object's own yaw is irrelevant too: it is round.
    spun = plan_grasp(CYLINDER, (placement[0], placement[1], 2.7))
    assert np.array_equal(first.q_grasp, spun.q_grasp)


def test_a_top_down_placement_is_out_of_reach_for_a_side_grasp_and_says_so():
    """The two regions are disjoint, and a plan in the wrong one fails honestly."""
    plan = plan_grasp(CYLINDER, (0.20, 0.0, 0.0))
    assert not plan.ok and plan.reason.startswith("ik_")
    assert plan.q_grasp.shape == (ARM,)
    lower = np.array([specs.JOINT_LIMITS[n][0] for n in kinematics.ARM_JOINT_NAMES])
    upper = np.array([specs.JOINT_LIMITS[n][1] for n in kinematics.ARM_JOINT_NAMES])
    assert np.all(plan.q_grasp >= lower - 1e-9) and np.all(plan.q_grasp <= upper + 1e-9)


def test_a_side_grasp_walks_advance_where_a_top_down_one_descends():
    """The FSM's own sequence, driven against a converging plant."""
    from manus.expert import ADVANCE

    expert = ScriptedGraspExpert(CYLINDER, a_placement(CYLINDER))
    states = [state for state, _ in run(expert, FakeArm())]
    assert [report.state for report in expert.reports] == list(
        expert_mod.SIDE_STATE_SEQUENCE[:-1]
    )
    assert ADVANCE in states and DESCEND not in states
    advances = [report for report in expert.reports if report.state == ADVANCE]
    assert advances and all(report.exit == "converged" for report in advances)
    assert expert.sequence == expert_mod.SIDE_STATE_SEQUENCE


def test_advance_freezes_the_arm_for_close_the_way_descend_does():
    """CLOSE holds whatever the approach reached -- either approach."""
    from manus.expert import ADVANCE

    expert = ScriptedGraspExpert(CYLINDER, a_placement(CYLINDER))
    plant = FakeArm(droop=0.03)
    commands = {}
    measured = plant.q.copy()
    for _ in range(4000):
        if expert.done:
            break
        targets = expert.step(measured)
        # After the call, because step() decides the transition first and then
        # issues the command for whichever state it ended up in.
        commands.setdefault(expert.state, []).append(
            [targets[n] for n in kinematics.ARM_JOINT_NAMES]
        )
        measured = plant.apply(targets)
    assert np.allclose(commands[CLOSE][0], commands[ADVANCE][-1])
    assert np.allclose(commands[CLOSE][0], commands[CLOSE][-1])


def test_the_jaws_are_held_open_all_the_way_through_advance():
    from manus.expert import ADVANCE

    expert = ScriptedGraspExpert(CYLINDER, a_placement(CYLINDER))
    plant = FakeArm()
    measured = plant.q.copy()
    seen = []
    for _ in range(4000):
        if expert.done:
            break
        targets = expert.step(measured)
        if expert.state == ADVANCE:
            seen.append(targets["gripper"])
        measured = plant.apply(targets)
    assert seen and all(value == pytest.approx(GRIPPER_OPEN) for value in seen)


def test_the_advance_ramp_is_the_descend_ramp():
    """One approach knob, not two -- the two moves are the same size."""
    from manus.expert import ADVANCE

    config = ExpertConfig(descend_ramp=17)
    assert config.ramp_steps(ADVANCE, CYLINDER) == config.ramp_steps(DESCEND, CUBE) == 17


def test_the_side_telemetry_says_which_grasp_it_was():
    import json

    expert = ScriptedGraspExpert(CYLINDER, a_placement(CYLINDER))
    run(expert, FakeArm())
    telemetry = expert.telemetry()
    assert telemetry["grasp_mode"] == "side"
    assert telemetry["approach_azimuth"] == pytest.approx(expert.plan.approach_azimuth)
    assert json.dumps(telemetry)
    top = ScriptedGraspExpert(CUBE, (0.20, 0.0, 0.0))
    run(top, FakeArm())
    assert top.telemetry()["grasp_mode"] == "top"
    assert top.telemetry()["approach_azimuth"] is None


# --- The thick puck ----------------------------------------------------------------


def test_the_thick_puck_is_the_thin_ones_disc_with_a_rim_the_pads_can_hold():
    """The respec, measured: 8.2 mm of rim between the pads instead of 5.0.

    Both pucks end up gripped above their own centre, but only one of them has
    the rim to spare. The thin one is raised until its fingertips clear the
    table and that leaves the pads on the top half of a 10 mm rim; the thick one
    is raised further still, to get the closing finger out from under its centre
    of mass, and *still* keeps 8.2 mm of rim -- which is the bar the 20 mm
    respec was chosen for.
    """
    from manus.expert import JAW_TIP_Z, MIN_TIP_CLEARANCE, grasp_height

    assert THICK_PUCK.grasp_width_m == PUCK.grasp_width_m == 0.040
    assert THICK_PUCK.close_target_rad == PUCK.close_target_rad
    assert THICK_PUCK.extent_z == 2 * PUCK.extent_z

    purchase = {}
    for spec in (PUCK, THICK_PUCK):
        clearance = grasp_height(spec) + TCP_TO_PAD_CENTRE - JAW_TIP_Z
        purchase[spec.name] = spec.extent_z - clearance
        assert clearance >= MIN_TIP_CLEARANCE - 1e-9
    assert purchase["puck_d40x10"] == pytest.approx(0.0050, abs=1e-4)
    assert purchase["puck_d40x20"] == pytest.approx(0.0082, abs=1e-4)
    assert purchase["puck_d40x20"] >= 0.008
    assert grasp_height(THICK_PUCK) > THICK_PUCK.spawn_z
    assert grasp_height(PUCK) > PUCK.spawn_z


def test_the_thick_pucks_grasp_is_raised_out_of_the_levering_band():
    """Why the thick puck's grasp is 4.1 mm above its centre, and what it costs.

    Centred, the filmed attempt lost it: the puck climbed the closing finger and
    rode it 50 mm up with the arm frozen. The finger reaches its deepest 8.06 mm
    below the TCP (:data:`~manus.expert.MOVING_JAW_DEEPEST_Z`), which at the
    centred grasp is 4.1 mm **under** the puck's centre of mass -- a horizontal
    squeeze applied below the centre of a 40 mm disc, which is a lever, not a
    grip. Raising the grasp until that deepest sweep lands *at* the centre of
    mass is what ``tip_clearance_m`` buys, and the cost is measured here too:
    the first touch on the rim creeps from 2.9 mm below its top edge to 1.2, and
    the pads' purchase from 12.3 mm to 8.2.

    The thin puck cannot have both -- clearing its 5 mm centre of mass would
    leave the pads 3.2 mm of rim, well under the 8 mm bar -- which is the same
    reason it is experimental and its thick respec exists.
    """
    from manus.expert import (
        JAW_TIP_Z,
        MOVING_JAW_DEEPEST_Z,
        grasp_height,
        tip_clearance,
    )

    def deepest_above_table(spec, height):
        return height + TCP_TO_PAD_CENTRE - MOVING_JAW_DEEPEST_Z

    # The bar, at the height the catalogue actually asks for.
    assert deepest_above_table(THICK_PUCK, grasp_height(THICK_PUCK)) == pytest.approx(
        THICK_PUCK.spawn_z, abs=2e-4
    )
    assert deepest_above_table(THICK_PUCK, THICK_PUCK.spawn_z) == pytest.approx(
        THICK_PUCK.spawn_z - 0.0041, abs=2e-4
    )
    assert tip_clearance(THICK_PUCK) == pytest.approx(0.01176)

    # What it costs on the rim, both ways, off the meshes.
    centred = first_rim_contact(THICK_PUCK, THICK_PUCK.spawn_z + TCP_TO_PAD_CENTRE - JAW_TIP_Z)
    raised = first_rim_contact(
        THICK_PUCK, grasp_height(THICK_PUCK) + TCP_TO_PAD_CENTRE - JAW_TIP_Z
    )
    assert THICK_PUCK.top_z - centred[1] == pytest.approx(0.0029, abs=5e-4)
    assert THICK_PUCK.top_z - raised[1] == pytest.approx(0.0012, abs=5e-4)
    assert centred[2] == pytest.approx(0.0123, abs=1e-4)
    assert raised[2] == pytest.approx(0.0082, abs=1e-4)

    # The thin puck's two bars cannot both be met, which is what "experimental"
    # is recording: 3.2 mm of rim left if it clears its own centre of mass.
    thin_height = PUCK.spawn_z + MOVING_JAW_DEEPEST_Z - TCP_TO_PAD_CENTRE
    thin_purchase = PUCK.extent_z - (thin_height + TCP_TO_PAD_CENTRE - JAW_TIP_Z)
    assert thin_purchase == pytest.approx(0.0032, abs=2e-4)
    assert thin_purchase < 0.008
    assert not THICK_PUCK.experimental and PUCK.experimental


def test_the_thick_puck_joins_the_default_sweep_and_the_thin_one_does_not():
    from manus.objects import DEFAULT_OBJECTS

    assert "puck_d40x20" in DEFAULT_OBJECTS
    assert "puck_d40x10" not in DEFAULT_OBJECTS
    assert "cylinder_3cm" in DEFAULT_OBJECTS


# --- The creeping CLOSE ---------------------------------------------------------------


def close_trace(spec, config=None) -> list[float]:
    """The jaw targets CLOSE commands for `spec`, step by step."""
    from manus.expert import close_command, close_steps

    config = ExpertConfig() if config is None else config
    return [
        close_command(GRIPPER_OPEN, spec.close_target_rad, step, spec, config)
        for step in range(1, close_steps(spec, config) + 1)
    ]


def test_the_tuned_close_ramp_is_untouched_for_everything_that_already_grasps():
    """Bit-identical: no object the ramp already closes gets a different command.

    The creep is opt-in per object, so the five catalogue objects the 200-attempt
    gate and every committed dataset were produced with issue exactly the linear
    ramp they always did -- checked against the arithmetic it replaced, not
    against a recording of itself.
    """
    from manus.expert import close_command, close_steps

    config = ExpertConfig()
    for spec in OBJECTS.values():
        if spec.close_creep:
            continue
        span = config.ramp_steps(CLOSE, spec)
        assert close_steps(spec, config) == span
        for step in range(1, span + 5):
            expected = GRIPPER_OPEN + min(1.0, step / span) * (
                spec.close_target_rad - GRIPPER_OPEN
            )
            assert close_command(
                GRIPPER_OPEN, spec.close_target_rad, step, spec, config
            ) == pytest.approx(expected)
    assert [spec.name for spec in OBJECTS.values() if spec.close_creep] == [
        "cylinder_3cm",
        "puck_d40x20",
    ]


@pytest.mark.parametrize("spec", [CYLINDER, THICK_PUCK], ids=lambda spec: spec.name)
def test_a_creeping_close_is_fast_to_the_contact_band_and_slow_through_it(spec):
    """The two rates, and that the slow one covers everything that can touch.

    The hand-over is :data:`~manus.expert.CLOSE_CREEP_LEAD_RAD` above the
    object's own contact angle -- 2.2 mm of jaw gap, which is wider than the
    :data:`~manus.expert.JAW_CLEARANCE` the object has to be shoved across plus
    the arm's convergence residual -- so the jaw is already creeping before it
    can reach the object however the attempt landed.
    """
    from manus.expert import CLOSE_CREEP_LEAD_RAD, CLOSE_CREEP_RATE_RAD, close_steps

    trace = close_trace(spec)
    hand_over = spec.contact_angle_rad + CLOSE_CREEP_LEAD_RAD
    fast = [after - before for before, after in zip(trace, trace[1:]) if before > hand_over]
    slow = [after - before for before, after in zip(trace, trace[1:]) if after < hand_over]
    assert max(fast) < 0.0, "the fast leg has to close, not open"
    assert min(slow) == pytest.approx(-CLOSE_CREEP_RATE_RAD, abs=1e-9)
    assert abs(np.mean(fast)) > 10 * CLOSE_CREEP_RATE_RAD
    # It arrives, exactly, at the object's own close target...
    assert trace[-1] == pytest.approx(spec.close_target_rad)
    # ... and everything from first possible touch to the target is crept.
    crept = [value for value in trace if value <= hand_over]
    assert len(crept) * CLOSE_CREEP_RATE_RAD >= hand_over - spec.close_target_rad
    assert close_steps(spec) == len(trace)
    assert close_steps(spec) < ExpertConfig().state_budget, "CLOSE cannot finish its creep"


def test_the_creep_is_sized_by_the_energy_the_servo_stores_in_the_shove():
    """Why creeping is the fix, in joules -- the number both failures came out of.

    The object stands :data:`~manus.expert.JAW_CLEARANCE` off the static pad, so
    the closing jaw shoves it across that gap alone. The jaw is a position
    servo: every millimetre of blocked travel is stiffness, and by virtual work
    through :data:`~manus.objects.JAW_WIDTH_PER_RAD` the spring it stores by the
    time the object reaches the static pad is ``0.5 * kp * gap^2 / rate^2``.

    That energy is not small compared with the objects it is handed to: it is
    2.4x the work needed to tip the 60 mm cylinder onto its base edge, and on
    the 30 g puck it is two thirds of a metre per second. Creeping cuts it by
    the square of the speed ratio.
    """
    from manus.expert import CLOSE_CREEP_RATE_RAD, JAW_CLEARANCE

    rate = objects.JAW_WIDTH_PER_RAD
    stored = lambda gap: 0.5 * specs.STS3215_KP * gap**2 / rate**2  # noqa: E731

    shove = stored(JAW_CLEARANCE)
    assert shove == pytest.approx(0.0067, abs=2e-4)
    # The cylinder's topple barrier: lifting its centre of mass onto the edge
    # it would pivot about.
    barrier = CYLINDER.mass_kg * 9.81 * (
        math.hypot(CYLINDER.radius, 0.5 * CYLINDER.extent_z) - 0.5 * CYLINDER.extent_z
    )
    assert barrier == pytest.approx(0.0028, abs=2e-4)
    assert shove > 2 * barrier, "the shove would not have toppled it after all"
    # ... and on the puck it is a launch speed, which is what the take showed.
    assert math.sqrt(2 * shove / THICK_PUCK.mass_kg) == pytest.approx(0.67, abs=0.05)

    # What the creep leaves: one step's travel of standing error.
    crept = stored(CLOSE_CREEP_RATE_RAD * rate)
    assert shove / crept > 300
    assert crept < 0.1 * barrier


def test_a_creeping_close_still_ends_on_the_stall_and_still_squeezes():
    """The FSM contract is unchanged: CLOSE ends squeezing, on a stalled jaw.

    The creep lengthens CLOSE, so the exit rule has to wait for it
    (:func:`~manus.expert.close_steps`) rather than for the ramp -- otherwise a
    jaw stalled on the object early in the creep would hand LIFT a command that
    had never reached the squeeze, and the squeeze would arrive as a step.
    """
    from manus.expert import close_steps

    spec = CYLINDER
    expert = ScriptedGraspExpert(spec, a_placement(spec))
    plant = FakeArm(jaw_stop=spec.contact_angle_rad)
    trace = run(expert, plant)
    report = next(item for item in expert.reports if item.state == CLOSE)
    assert report.exit == "stalled"
    assert report.steps >= close_steps(spec)
    assert report.gripper == pytest.approx(spec.contact_angle_rad, abs=1e-3)
    # ... and the command it left behind is the full squeeze, not the creep.
    assert trace[-1][1][-1] == pytest.approx(spec.contact_angle_rad, abs=1e-3)


# --- The side approach's error budget -----------------------------------------------


def tcp_height_sensitivity(q) -> float:
    """``sum_i |d z_tcp / d q_i|`` at arm pose `q`, metres per radian.

    The worst-case TCP *height* error a per-joint convergence bar admits: every
    joint sitting on the bar with its sign chosen against you. That is not a
    pessimist's bound for this arm -- gravity droops the three pitch joints the
    same way at once, which is exactly the aligned case (the filmed cylinder's
    ADVANCE exit: 16.9, 14.8 and 11.1 mrad, one sign, 10.7 mm of height).
    """
    gradient = np.zeros(kinematics.NUM_ARM_JOINTS)
    for index in range(kinematics.NUM_ARM_JOINTS):
        step = np.zeros(kinematics.NUM_ARM_JOINTS)
        step[index] = 1e-6
        gradient[index] = (
            CHAIN.fk_tcp(q + step)[0][2] - CHAIN.fk_tcp(q - step)[0][2]
        ) / 2e-6
    return float(np.abs(gradient).sum())


def test_the_side_convergence_bar_is_half_the_hands_table_clearance():
    """SIDE_CONVERGE_TOL, re-derived: what the bar admits, against what is there.

    The rule is that a side move may not exit with more vertical TCP error than
    *half* the gap between the hand's lowest material and the table -- so even a
    move that exits exactly on the bar still clears the table by more than
    :data:`~manus.expert.MIN_TIP_CLEARANCE`. Both halves are measured here: the
    clearance off the grasp height and the hand's own geometry, the sensitivity
    off the FK at the worst radius in the region.
    """
    from manus.expert import (
        MIN_TIP_CLEARANCE,
        SIDE_CONVERGE_TOL,
        converge_tol,
        side_table_clearance,
    )

    region = placement_region(CYLINDER)
    worst = max(
        tcp_height_sensitivity(side_plan(radius=float(radius))[0].q_grasp)
        for radius in np.linspace(*region.radius, 5)
    )
    assert worst == pytest.approx(0.826, abs=0.02)
    clearance = side_table_clearance(CYLINDER)
    assert SIDE_CONVERGE_TOL == pytest.approx(0.5 * clearance / worst, abs=5e-4)
    # What that admits, and what is left when it does.
    admitted = SIDE_CONVERGE_TOL * worst
    assert admitted == pytest.approx(0.0061, abs=5e-4)
    assert clearance - admitted >= MIN_TIP_CLEARANCE
    assert converge_tol(CYLINDER) == SIDE_CONVERGE_TOL

    # The old bar admitted more than the whole clearance the old grasp had:
    # 20 mrad at that radius is 15 mm of height against 5.8 mm of gap.
    old = expert_mod.CONVERGE_TOL * worst
    assert old > 0.0058, f"the old bar admitted {old * 1e3:.1f} mm"
    assert SIDE_CONVERGE_TOL < 0.4 * expert_mod.CONVERGE_TOL


def test_a_side_approach_holds_its_waypoint_until_the_droop_is_cancelled():
    """The settle dwell, against a drooping plant: the fix for the filmed exit.

    The filmed ADVANCE exited on the *first* step its ramp completed, with one
    integrator update behind it and 7.4 mm of sag still in the arm. With the
    dwell the same state keeps stepping, the bias keeps growing, and what CLOSE
    freezes is a pose that actually reached the waypoint.
    """
    from manus.expert import ADVANCE, SIDE_SETTLE_STEPS, converge_tol

    config = ExpertConfig()
    expert = ScriptedGraspExpert(CYLINDER, a_placement(CYLINDER), config=config)
    droop = np.full(ARM, 0.02)
    run(expert, FakeArm(droop=droop))
    advance = next(report for report in expert.reports if report.state == ADVANCE)
    assert advance.exit == "converged"
    assert advance.steps >= config.descend_ramp + SIDE_SETTLE_STEPS
    assert advance.joint_error < converge_tol(CYLINDER)
    # The bias is the droop it cancelled, and it got there by having the time.
    assert max(abs(value) for value in advance.bias) == pytest.approx(0.02, abs=3e-3)

    # Without the dwell the same plant exits early, still drooping, and lands
    # outside the bar the clearance asks for.
    hasty = ScriptedGraspExpert(
        CYLINDER, a_placement(CYLINDER), config=dataclasses.replace(config, side_settle_steps=0)
    )
    run(hasty, FakeArm(droop=droop))
    early = next(report for report in hasty.reports if report.state == ADVANCE)
    assert early.steps < advance.steps
    assert early.joint_error > advance.joint_error


def test_only_a_side_approach_dwells():
    """The dwell is scoped to the side approach, so no top-down state moves.

    :func:`~manus.expert.settle_steps` is the whole scope: DESCEND, CLOSE, LIFT
    and HOLD answer zero, which makes the FSM's exit rule for a top-down grasp
    bit-for-bit the one every committed dataset was generated under.
    """
    from manus.expert import settle_steps

    config = ExpertConfig()
    for state in (PREGRASP, DESCEND, CLOSE, LIFT, HOLD):
        assert settle_steps(state, CUBE, config) == 0
    for state in (CLOSE, LIFT, HOLD):
        assert settle_steps(state, CYLINDER, config) == 0
    for state in (PREGRASP, expert_mod.ADVANCE):
        assert settle_steps(state, CYLINDER, config) == config.side_settle_steps
    assert settle_steps(PREGRASP, None, config) == 0

    # And the top-down FSM's step counts are unchanged against a fixed plant.
    plant = FakeArm(droop=np.full(ARM, 0.01))
    expert = ScriptedGraspExpert(CUBE, (0.20, 0.0, 0.0), config=config)
    run(expert, plant)
    dwelling = dataclasses.replace(config, side_settle_steps=99)
    assert [report.steps for report in expert.reports] == [
        report.steps for report in _rerun(CUBE, (0.20, 0.0, 0.0), dwelling)
    ]


def _rerun(spec, placement, config):
    """Run one attempt against a standard drooping plant and return its reports."""
    expert = ScriptedGraspExpert(spec, placement, config=config)
    run(expert, FakeArm(droop=np.full(ARM, 0.01)))
    return expert.reports


# --- The wrist camera, at the poses the plans actually stand in ----------------------
#
# The POV the whole dataset is recorded through is a rigid mount on the gripper
# link, so where it ends up -- and which way up it is -- is a property of the
# *plan*, not of the renderer. All of this is FK plus manus.specs' mount, no
# Isaac app involved, which is the point: the cylinder's side grasp put the
# camera 25 mm under the table for a whole filmed episode and nothing that runs
# on the CPU noticed.


def camera_frame(q):
    """``(position, right, up, view)`` of the wrist camera at arm pose `q`.

    The rotation is in the camera's own OpenGL convention, so its columns are
    the image's right and up axes and *minus* the view direction.
    """
    position, rotation = CHAIN.wrist_camera_pose(q)
    return position, rotation[:, 0], rotation[:, 1], -rotation[:, 2]


def in_view(q, point, margin_deg: float = 0.0) -> bool:
    """Whether `point` falls inside the wrist camera's frustum at pose `q`."""
    position, right, up, view = camera_frame(q)
    offset = np.asarray(point, dtype=float) - position
    depth = float(offset @ view)
    if depth <= 0.0:
        return False
    half_width = 0.5 * specs.WRIST_CAM_APERTURE / specs.WRIST_CAM_FOCAL
    half_height = half_width * specs.WRIST_CAM_HEIGHT / specs.WRIST_CAM_WIDTH
    shrink = math.tan(math.radians(margin_deg))
    return (
        abs(float(offset @ right) / depth) <= half_width - shrink
        and abs(float(offset @ up) / depth) <= half_height - shrink
    )


def test_the_wrist_camera_mount_is_the_one_the_scene_spawns():
    """The sim-free copy of the mount is the copy the sensor is built from."""
    assert specs.WRIST_CAM_PARENT_LINK == specs.LINK_CHAIN[-1] == "gripper_link"
    assert len(specs.WRIST_CAM_QUAT_XYZW) == 4
    # A pure turn about the parent's +X, which is what makes the camera's up
    # axis live in the tool's own (y, z) plane -- see the roll test below.
    x, y, z, w = specs.WRIST_CAM_QUAT_XYZW
    assert (y, z) == (0.0, 0.0)
    assert math.degrees(2 * math.atan2(x, w)) == pytest.approx(-32.7, abs=0.1)
    rotation = kinematics.rotation_from_quat_xyzw(np.array(specs.WRIST_CAM_QUAT_XYZW))
    # ... aimed just past the fingertips, 67 mm out, which is the vendor claim.
    tip = np.array(specs.WRIST_CAM_POS) - 0.067 * rotation[:, 2]
    assert tip[2] == pytest.approx(-0.101, abs=2e-3)
    assert abs(tip[1]) < 0.020


@pytest.mark.parametrize(
    "spec", [CUBE, DIE, DUPLO, THICK_PUCK, CYLINDER], ids=lambda spec: spec.name
)
def test_the_wrist_camera_watches_the_grasp_from_above_the_table(spec):
    """The invariant the cylinder's side grasp broke, checked on both families.

    Three things have to hold at every arm waypoint of a plan, or the episode's
    images are not a view of the task:

    1. the camera is **above the table** -- at the discarded
       :data:`~manus.expert.SIDE_GRASP_ROLL` = pi it was 25 mm *below* the
       ground plane, filming its underside,
    2. it **looks downward**, at the work rather than at the sky, and
    3. the object is **in front of it and inside the frustum** at the grasp
       pose, with room to spare.
    """
    for x, y in region_samples(3, 3, spec=spec):
        plan = plan_grasp(spec, (x, y, 0.4))
        assert plan.ok, plan.reason
        for state in (PREGRASP, expert_mod.approach_state(spec), LIFT):
            q = plan.waypoint(state)
            position, _, _, view = camera_frame(q)
            assert position[2] > 0.02, (
                f"{spec.name} {state}: camera {position[2] * 1e3:.1f} mm above the table"
            )
            assert view[2] < 0.0, f"{spec.name} {state}: the camera is looking up"
        assert in_view(plan.q_grasp, [x, y, expert_mod.grasp_height(spec)], margin_deg=5.0)


def test_the_side_grasps_camera_is_upright_and_the_top_downs_is_the_vendors():
    """Which way up the recorded image is, per family -- and why they differ.

    "Upright" is a statement about gravity: the world's up direction, projected
    into the image plane, has to point up in the image, which is the sign of the
    camera up axis' world z.

    * The **side** family is upright by construction now: +0.842, and its
      projected world-up is the image up axis exactly.
    * The **top-down** family is not, and never was -- it is the vendor mount
      seen from a tool that points at the floor: the camera looks 57 deg down
      over its own boom, its up axis comes out at -0.540, and a falling object
      rises in the frame. Every dataset in the repo is recorded that way and it
      is left alone deliberately.

    And the two cannot be reconciled by re-rolling the mount, which is the
    reason this is a finding rather than a bug: the camera's up axis lives in
    the tool's own (y, z) plane, the two families' tools differ by a quarter
    turn about the tool x, so any roll that lifts one lowers the other. Measured
    below by sweeping the mount roll over a full turn.
    """
    top = plan_grasp(CUBE, (0.19, 0.0, 0.0))
    side, _ = side_plan()
    for plan, spec in ((top, CUBE), (side, CYLINDER)):
        for q in (plan.q_pregrasp, plan.q_grasp):
            _, _, up, view = camera_frame(q)
            projected = np.array([0.0, 0.0, 1.0]) - view * float(view[2])
            projected /= np.linalg.norm(projected)
            # The image's up axis is exactly aligned or exactly opposed to the
            # projected world up: no in-between, in either family.
            assert abs(float(projected @ up)) == pytest.approx(1.0, abs=1e-6)
    assert camera_frame(side.q_grasp)[2][2] == pytest.approx(+0.842, abs=1e-3)
    assert camera_frame(top.q_grasp)[2][2] == pytest.approx(-0.540, abs=1e-3)

    # No mount roll makes both families upright: the product of the two
    # families' up-axis world z is negative at every roll angle.
    base = kinematics.rotation_from_quat_xyzw(np.array(specs.WRIST_CAM_QUAT_XYZW))
    parents = [CHAIN.fk(q)[specs.WRIST_CAM_PARENT_LINK][1] for q in (top.q_grasp, side.q_grasp)]
    for angle in np.linspace(0.0, 2 * math.pi, 73):
        cos, sin = math.cos(angle), math.sin(angle)
        rolled = base @ np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        ups = [float((parent @ rolled)[2, 1]) for parent in parents]
        assert ups[0] * ups[1] <= 1e-9, f"a mount roll of {math.degrees(angle):.0f} deg fixes both"


# --- The success predicate, side-on -------------------------------------------------


def test_the_stall_clause_does_not_know_which_way_the_hand_points():
    """Clause 4 is a jaw angle against a width. Nothing in it is a direction.

    Checked by building the band for the side-grasped cylinder and for a
    top-down object of the same width -- the cube -- and finding them equal.
    """
    side = GraspSuccessMonitor(CYLINDER)
    top = GraspSuccessMonitor(CUBE)
    assert side.stall_band == top.stall_band
    assert CYLINDER.grasp_width_m == CUBE.grasp_width_m


def test_the_in_hand_clause_measures_the_same_distance_side_on():
    """Clause 3 is |object - TCP|, and a seated object sits at the same place.

    The pad centre is ``(lateral, 0, TCP_TO_PAD_CENTRE)`` in the tool's own
    frame whichever way the tool points, so the distance the bar compares
    against is a property of the *hand*, not of the approach: 17.5 mm for a
    30 mm grasp, either way, against a 60 mm bar.
    """
    joints, seated = lift_pose(CYLINDER, CYLINDER.contact_angle_rad)
    monitor = hold(GraspSuccessMonitor(CYLINDER), joints, seated)
    assert monitor.success
    assert monitor.tcp_distance == pytest.approx(
        math.hypot(expert_mod.pad_lateral_offset(CYLINDER), TCP_TO_PAD_CENTRE), abs=1e-6
    )
    assert monitor.tcp_distance == pytest.approx(0.0175, abs=1e-4)
    assert monitor.tcp_distance < 0.35 * monitor.in_hand_radius


def test_a_side_grasped_cylinder_clears_the_height_bar_from_its_own_spawn():
    """Clause 1, side-on: it is lifted from 30 mm, and the bar is 80 mm.

    The bar is the object's rest height plus 5 cm whichever way it was picked
    up, and the side lift delivers ~94 mm of TCP rise -- so the margin is the
    lift's, not the bar's.
    """
    monitor = GraspSuccessMonitor(CYLINDER)
    assert monitor.spawn_z == 0.030
    assert monitor.threshold_z == pytest.approx(0.080)
    plan, _ = side_plan()
    lifted_tcp_z = float(CHAIN.fk_tcp(plan.q_lift)[0][2])
    assert lifted_tcp_z - expert_mod.grasp_height(CYLINDER) == pytest.approx(
        plan.lift_rise, abs=1e-9
    )
    assert lifted_tcp_z > monitor.threshold_z + 0.03


def test_a_side_attempt_that_never_grips_is_classified_the_same_way():
    """classify_outcome is mode-agnostic: same clauses, same names."""
    expert = ScriptedGraspExpert(CYLINDER, a_placement(CYLINDER))
    run(expert, FakeArm())
    monitor = GraspSuccessMonitor(CYLINDER)
    hold(monitor, np.concatenate([expert.plan.q_lift, [0.05]]), np.array([0.4, 0.0, 0.03]))
    assert classify_outcome(expert, monitor) == "no_grasp"


# --- The top-down plan, bit for bit -------------------------------------------------

GOLDEN_TOP_DOWN_PLANS = {
    "cube_3cm": (
        2.1984544613195234,
        (
            1.0216064975872534,
            -0.5388804881491307,
            0.9517017922116668,
            1.1579750227323604,
            0.12714553815510543,
        ),
        (
            1.0216064975872534,
            -0.6149584670171313,
            0.7758126539563803,
            1.4099421398556475,
            0.12714553815510543,
        ),
        (
            1.0216064975872534,
            -0.7815003989298153,
            0.6185497209910995,
            1.0902300816172816,
            0.12714553815510543,
        ),
    ),
    "die_16mm": (
        2.1984544613195234,
        (
            1.0173629789932903,
            -0.43846882150721234,
            0.9307881590022333,
            1.0784769892998756,
            0.12290201956114233,
        ),
        (
            1.0173629789932903,
            -0.5366720427039331,
            0.7722675991449246,
            1.335200770353905,
            0.12290201956114233,
        ),
        (
            1.0173629789932903,
            -0.68790093553994,
            0.6144094175553028,
            1.0148511548867158,
            0.12290201956114233,
        ),
    ),
    "duplo_32x64": (
        2.1984544613195234,
        (
            1.0221874833902813,
            -0.5355466687512505,
            0.9732575660017702,
            1.1330854295443769,
            0.12772652395813333,
        ),
        (
            1.0221874833902813,
            -0.6198091431471462,
            0.8015485320925015,
            1.3890569378495412,
            0.12772652395813333,
        ),
        (
            1.0221874833902813,
            -0.7784652739782671,
            0.640304906028293,
            1.0653694057443903,
            0.12772652395813333,
        ),
    ),
}
"""``(grasp_yaw, q_grasp, q_pregrasp, q_lift)`` recorded from ``plan_grasp`` at
``draw_episode("grasp_cube_v2", 7)`` **before** side grasps existed.

Three objects covering the three yaw-symmetry classes and both ends of the
convergence-tolerance scale. Pinned to the last bit rather than to a tolerance:
adding a second tool family to ``ik_solve`` and a second mode to the expert was
allowed to add plans and forbidden to move them, because these joint angles are
what the 200-attempt gate and every committed dataset were produced with, and a
micron of drift here is a silently different dataset rather than a failing test.
"""


@pytest.mark.parametrize("name", list(GOLDEN_TOP_DOWN_PLANS))
def test_the_top_down_plan_is_unchanged_to_the_last_bit(name):
    expected_yaw, expected_grasp, expected_pregrasp, expected_lift = GOLDEN_TOP_DOWN_PLANS[name]
    plan = plan_grasp(OBJECTS[name], draw_episode("grasp_cube_v2", 7))
    assert plan.ok, plan.reason
    assert plan.grasp_mode == "top"
    assert plan.approach_azimuth is None
    assert plan.grasp_yaw == expected_yaw
    assert tuple(plan.q_grasp.tolist()) == expected_grasp
    assert tuple(plan.q_pregrasp.tolist()) == expected_pregrasp
    assert tuple(plan.q_lift.tolist()) == expected_lift


def test_the_side_planner_never_returns_a_pose_the_arm_cannot_hold():
    """Even nonsense placements come back clamped and finite, not as an exception.

    The generator runs infeasible attempts on purpose (they are recorded as
    ``ik_infeasible`` outcomes rather than skipped), so a plan for a placement
    on top of the robot has to be a *plan*.
    """
    lower = np.array([specs.JOINT_LIMITS[n][0] for n in kinematics.ARM_JOINT_NAMES])
    upper = np.array([specs.JOINT_LIMITS[n][1] for n in kinematics.ARM_JOINT_NAMES])
    for placement in ((0.0, 0.0, 0.0), (0.039, 0.0, 0.0), (1.5, 0.0, 0.0), (-0.5, -0.5, 2.0)):
        plan = plan_grasp(CYLINDER, placement)
        assert not plan.ok and plan.reason.startswith("ik_")
        for q in (plan.q_pregrasp, plan.q_grasp, plan.q_lift):
            assert np.isfinite(q).all()
            assert np.all(q >= lower - 1e-9) and np.all(q <= upper + 1e-9)
