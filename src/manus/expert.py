"""Scripted grasp expert: the FSM that turns a placement into a grasp.

Sim-free by construction -- it plans with :mod:`manus.kinematics` and talks to
the caller in joint-space dictionaries, so every transition in
:class:`ScriptedGraspExpert` is exercised on the CPU by
``tests/test_expert_logic.py`` without an Isaac app anywhere near it. The
driver (``scripts/demo_expert.py``, ``scripts/gen_workspace_map.py --gate``,
and the Step 9 dataset generator) owns the simulator; the expert owns the plan.

The loop the driver runs is::

    expert.reset(placement, q_current=measured)
    while not expert.done:
        targets = expert.step(measured)     # dict, all six joints, radians
        write targets; step physics; read measured

States, in order -- :data:`STATE_SEQUENCE`:

``PREGRASP``
    Tool vertical, TCP :data:`ExpertConfig.hover_height` above the grasp point,
    jaws open. Reached from wherever the arm happens to be.
``DESCEND``
    Straight down to the grasp pose: the *jaws* around the object, which is not
    the TCP on the object -- see :func:`tcp_target` and :func:`grasp_height`.
``CLOSE``
    Arm frozen, jaws driven to ``spec.close_target_rad`` -- deliberately past
    contact, so the servo squeezes against its effort limit.
``LIFT``
    Joint-space retraction: the three pitch joints follow the TCP-height
    gradient to a raised pose while shoulder_pan and wrist_roll hold still.
    **No verticality constraint** -- the tool-vertical family tops out at
    0.0903 m TCP height, which is not enough clearance, so the tool is allowed
    to tilt (see :func:`plan_lift`).
``HOLD``
    Everything still for :data:`ExpertConfig.hold_steps`, which is what the
    success predicate's "sustained" is measured over.

Three behaviours are worth knowing before reading the code:

**The tool frame is not the grasp frame.** The jaws straddle the *wrist_roll
axis*, which the TCP hangs 7.9 mm off, and the static finger fills everything on
one side of it. Aiming the TCP at the object -- the obvious reading of "grasp at
(x, y)" -- lands that finger on the object's lid and closes the hand on air.
:func:`tcp_target` aims the jaws instead, which makes the grasp yaw a *position*
decision as well as an orientation one; see the jaw-geometry block below.

**Servo-to-converge.** An arm state ends when the measured joints actually
reach the waypoint (``max |measured - waypoint| < CONVERGE_TOL``), not when a
step counter says so; a per-state budget only exists so a wedged attempt cannot
hang the generator, and every expiry is recorded in
:attr:`ScriptedGraspExpert.timeouts`.

**Droop compensation.** Under the vendored PD gains (kp 17.8, no gravity
feed-forward) a commanded pose is held low, which is a large fraction of a 30 mm
cube. The expert therefore integrates the measured joint error into a per-joint
bias added to the command (:attr:`ScriptedGraspExpert.bias`) once the ramp has
finished and the error is small enough to be droop rather than travel. The bias
*is* the droop measurement: ``commanded - measured`` at convergence, which over
the Step 8 gate ran 9 mrad mean / 80 mrad worst per joint, holding the TCP
within 4-9 mm of every waypoint.

All angles are radians, all lengths metres.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from manus import kinematics, specs
from manus.control import GRIPPER_OPEN, clamp_targets
from manus.kinematics import TCP_TO_PAD_CENTRE, KinematicChain, ik_solve
from manus.objects import ObjectSpec
from manus.randomize import EpisodeDraw

__all__ = [
    "CONVERGE_TOL",
    "STATE_SEQUENCE",
    "ExpertConfig",
    "GraspPlan",
    "GraspSuccessMonitor",
    "ScriptedGraspExpert",
    "StateReport",
    "classify_outcome",
    "grasp_height",
    "grasp_yaw_candidates",
    "joint_vector",
    "plan_grasp",
    "plan_lift",
    "yaw_symmetry",
]

_CHAIN = KinematicChain()
"""Shared FK instance; :class:`~manus.kinematics.KinematicChain` is stateless."""

# The expert speaks specs.JOINT_NAMES order on the wire and kinematics'
# ARM_JOINT_NAMES order internally. They agree on the first five entries --
# pinned here rather than assumed, since a re-order would silently swap joints.
if kinematics.ARM_JOINT_NAMES != specs.JOINT_NAMES[: kinematics.NUM_ARM_JOINTS]:  # pragma: no cover
    raise ImportError(
        "arm joint order drifted: "
        f"{kinematics.ARM_JOINT_NAMES} vs {specs.JOINT_NAMES[: kinematics.NUM_ARM_JOINTS]}"
    )

GRIPPER_NAME = "gripper"
"""Name of the off-chain jaw joint in :data:`manus.specs.JOINT_NAMES`."""

GRIPPER_INDEX = specs.JOINT_NAMES.index(GRIPPER_NAME)
"""Column of the jaw joint in a six-joint measurement vector."""

NUM_JOINTS = len(specs.JOINT_NAMES)
"""Width of the measurement vectors :meth:`ScriptedGraspExpert.step` accepts."""

ARM_LOWER: np.ndarray = np.array(
    [specs.JOINT_LIMITS[name][0] for name in kinematics.ARM_JOINT_NAMES]
)
ARM_UPPER: np.ndarray = np.array(
    [specs.JOINT_LIMITS[name][1] for name in kinematics.ARM_JOINT_NAMES]
)
"""Arm joint travel in ``q`` order: what :func:`plan_lift` clamps against."""

# --- States -------------------------------------------------------------------

PREGRASP = "PREGRASP"
DESCEND = "DESCEND"
CLOSE = "CLOSE"
LIFT = "LIFT"
HOLD = "HOLD"
DONE = "DONE"

STATE_SEQUENCE: tuple[str, ...] = (PREGRASP, DESCEND, CLOSE, LIFT, HOLD, DONE)
"""Every state, in the order the FSM walks them. ``DONE`` is terminal."""

ARM_STATES: frozenset[str] = frozenset({PREGRASP, DESCEND, LIFT})
"""States that move the arm and therefore end on joint convergence."""

CONVERGE_TOL: float = 0.02
"""Default ``max |measured - waypoint|`` (radians, arm joints) ending a move.

Roughly 1.1 deg per joint, which lands the TCP within ~2 mm of the waypoint --
comfortably inside the ~7 mm of slack a 30 mm cube leaves in 34 mm of jaw
travel, and reachable only because the droop bias exists (raw steady-state
droop is 2-4x this on the shoulder).
"""


@dataclass(frozen=True)
class ExpertConfig:
    """Tunables of the FSM. Defaults are the tuned Step 7 values.

    Step counts are *control* steps -- the expert is rate-agnostic, but the
    pipeline drives it at ``recorder.CONTROL_HZ`` = 30 Hz, so 30 steps is one
    second of simulated time.

    Attributes:
        hover_height: PREGRASP TCP height above the grasp point, metres.
        converge_tol: Arm convergence tolerance, radians (see :data:`CONVERGE_TOL`).
        state_budget: Per-state step ceiling; expiry advances the FSM anyway and
            is recorded in :attr:`ScriptedGraspExpert.timeouts`.
        hold_steps: Length of the terminal HOLD.
        gripper_open: Jaw target while approaching, radians.
        gripper_stall_tol: Peak-to-peak jaw travel over
            :attr:`gripper_stall_window` steps below which CLOSE is deemed
            squeezed out, radians.
        gripper_stall_window: Length of that window, in steps.
        lift_rise: TCP rise :func:`plan_lift` aims for, metres.
        min_lift_rise: Rise below which the plan is declared infeasible, metres.
            The success predicate needs 0.05 m of *object* rise, so this is the
            floor with the tolerance and droop left over.
        pregrasp_ramp: Steps spent interpolating into the PREGRASP waypoint.
        descend_ramp: Steps spent interpolating down to the grasp pose.
        close_ramp: Steps spent driving the jaws to the close target. **The
            single most sensitive number here.** The jaws travel ~1.45 rad from
            open to the close target, so 20 steps is 2.2 rad/s of jaw closing
            speed, or ~90 mm/s at the pads -- fast enough that the fingers'
            inward taper flicks a 30 mm cube out of the hand like a watermelon
            seed. Two of 200 gate attempts failed exactly that way (seen on
            video: the cube leaves the frame upward while the jaws close on
            nothing). 60 steps is 0.7 rad/s and the same grip once seated:
            16/16 on the probe set that contained both failures, and no slips
            among the low-friction attempts a *weaker* squeeze cost instead.

        lift_ramp: Steps spent retracting to the lift pose.
        droop_gain: Integral gain folding measured joint error into the command.
            Zero disables droop compensation entirely.
        droop_engage: Per-joint error below which the integrator runs, radians --
            above it the joint is still travelling (or jammed) and integrating
            would wind up.
        droop_leak: Per-step factor the bias of a *non*-settling joint decays by,
            so a bias inherited from a jammed state cannot deadlock the next one.
        droop_limit: Absolute cap on the per-joint bias, radians. Past
            ``effort_limit / kp`` = 0.19 rad the servo is already saturated, so
            more bias buys no more torque; the cap is only a little above that.
    """

    hover_height: float = 0.03
    converge_tol: float = CONVERGE_TOL
    state_budget: int = 240
    hold_steps: int = 45
    gripper_open: float = GRIPPER_OPEN
    gripper_stall_tol: float = 0.002
    gripper_stall_window: int = 15
    lift_rise: float = 0.09
    min_lift_rise: float = 0.06
    pregrasp_ramp: int = 45
    descend_ramp: int = 30
    close_ramp: int = 60
    lift_ramp: int = 30
    droop_gain: float = 0.12
    droop_engage: float = 0.20
    droop_leak: float = 0.97
    droop_limit: float = 0.30

    def ramp_steps(self, state: str) -> int:
        """Ramp length of `state`, in steps (1 for states that do not ramp)."""
        return {
            PREGRASP: self.pregrasp_ramp,
            DESCEND: self.descend_ramp,
            CLOSE: self.close_ramp,
            LIFT: self.lift_ramp,
        }.get(state, 1)


DEFAULT_CONFIG = ExpertConfig()
"""The configuration :class:`ScriptedGraspExpert` uses unless told otherwise."""


# --- Planning -----------------------------------------------------------------


def joint_vector(measured: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
    """Normalise a measurement into a (6,) array in :data:`specs.JOINT_NAMES` order.

    Args:
        measured: Either a mapping carrying every joint name, or a sequence of
            six numbers already in ``specs.JOINT_NAMES`` order. Anything numpy
            can turn into a float array works, including a CPU torch tensor;
            move CUDA tensors across yourself.

    Returns:
        The (6,) measurement, radians.

    Raises:
        KeyError: A mapping is missing a joint.
        ValueError: A sequence is the wrong length or holds non-finite values.
    """
    if isinstance(measured, Mapping):
        missing = [name for name in specs.JOINT_NAMES if name not in measured]
        if missing:
            raise KeyError(f"measurement is missing joints: {missing}")
        vector = np.array([float(measured[name]) for name in specs.JOINT_NAMES])
    else:
        vector = np.asarray(measured, dtype=float).reshape(-1)
        if vector.shape != (NUM_JOINTS,):
            raise ValueError(
                f"measurement must have {NUM_JOINTS} entries in {specs.JOINT_NAMES} order, "
                f"got shape {np.shape(measured)}"
            )
    if not np.isfinite(vector).all():
        raise ValueError(f"measurement is not finite: {vector}")
    return vector


def _wrap(angle: float) -> float:
    """Fold an angle (radians) into [-pi, pi)."""
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def _placement(
    placement: EpisodeDraw | Sequence[float],
) -> tuple[float, float, float]:
    """``(x, y, yaw)`` of a placement given as a draw or a bare triple."""
    if isinstance(placement, EpisodeDraw):
        return (placement.object_x, placement.object_y, placement.object_yaw)
    values = tuple(float(value) for value in placement)
    if len(values) != 3:
        raise ValueError(f"placement must be an EpisodeDraw or (x, y, yaw), got {placement!r}")
    return values  # type: ignore[return-value]


# --- Jaw geometry --------------------------------------------------------------
# Measured off the vendored meshes (assets/so101/urdf/assets/*.stl) carried
# through the URDF origins into the TCP frame, where +x is the direction the
# static jaw sits in and +z is the approach axis (pointing out of the jaws).
# tests/test_expert_logic.py re-derives all three numbers from the STLs and the
# URDF, so a re-exported model cannot drift away from them silently.
#
# The shape of the hand is the thing that makes this expert non-obvious: the
# jaws are centred on the *wrist_roll axis*, which the TCP hangs 7.9 mm off, and
# the fingers are ~50 mm long with a stepped inner face. So aiming the TCP at
# the object -- the obvious thing -- lands the static finger squarely on the
# object's lid. Everything below exists to aim the *jaws* instead.

JAW_FIXED_FACE_X: float = 0.0
"""Innermost point of the static jaw over the engaged band, TCP-frame x, metres.

The face is stepped rather than flat -- 0.000 m at the fingertip, 0.002 m about
10 mm back, 0.0037 m about 20 mm back -- and it is the *innermost* step that
decides whether the object can be lowered past the finger, so that is what this
records. (It comes out at exactly zero: the vendor put the tool frame on the
static finger's gripping plane.) One consequence is unavoidable: an object
placed clear of this face still gets seated ~2 mm further in by the closing
jaw, because the step it finally rests against protrudes that much further.
"""

JAW_TIP_Z: float = 0.0063
"""Fingertip position along the approach axis, TCP-frame z, metres.

The TCP sits 6.3 mm *behind* the fingertips, so the tool can be driven to a
height of ``object_z + TCP_TO_PAD_CENTRE`` without the fingers touching ground.
"""

JAW_CLEARANCE: float = 0.002
"""Gap left between the object and the static jaw on the way down, metres.

Pure descent margin. The static jaw cannot move out of the way, so the object
has to be placed clear of it -- and the descent carries the full positioning
error of the arm, ~2 mm at the convergence tolerance. The moving jaw then
pushes the object back across this gap as it closes, so more clearance is a
safer approach and a bigger shove at contact.
"""


MIN_TIP_CLEARANCE: float = 0.005
"""Gap left between the static fingertips and the table at the grasp, metres.

Only binds on objects too short to centre the pads on: the pads sit
:data:`~manus.kinematics.TCP_TO_PAD_CENTRE` below the TCP and the fingertips
:data:`JAW_TIP_Z` below it, so centring on an object shorter than
``2 * (JAW_TIP_Z - TCP_TO_PAD_CENTRE)`` = 4.6 mm would drive the tips into the
ground outright, and anything under 14.6 mm leaves less than this clearance.
The 15 mm domino scrapes past by 0.4 mm; the 10 mm puck is the only catalogue
object :func:`grasp_height` actually has to raise. 5 mm is the compromise it
spends there: the tips clear the table by 5 mm against a ~2 mm droop residual at
convergence, and the pads still cover the puck's top half. Both directions cost
something real -- lower and a sagging finger digs into the table, higher and the
grip creeps towards the puck's top edge -- so this is the constant to move first
if the puck misbehaves.
"""

YAW_MATCH_TOL: float = math.radians(2.0)
"""How far the solved tool yaw may sit from the requested one, radians.

Wide enough to swallow ``ik_solve``'s own 1 deg yaw tolerance, far narrower
than the pi flip it is there to catch.
"""


def pad_lateral_offset(spec: ObjectSpec) -> float:
    """Where the object's centre should sit along TCP-frame x, metres.

    Negative: the object is offset *away* from the static jaw, by its own half
    width plus :data:`JAW_CLEARANCE`, so that the descent passes the static
    finger and the closing moving finger seats the object against it.
    """
    return JAW_FIXED_FACE_X - 0.5 * spec.grasp_width_m - JAW_CLEARANCE


def grasp_height(spec: ObjectSpec) -> float:
    """Height above the table the jaw pads centre on for `spec`, metres.

    The object's own mid-height, which puts the pads across the widest part of
    it -- raised, for an object short enough to need it, until the fingertips
    clear the table by :data:`MIN_TIP_CLEARANCE`. Nothing in the catalogue but
    the 10 mm puck is short enough to be raised at all, and it is raised 2.3 mm.
    """
    lowest_centre = JAW_TIP_Z - TCP_TO_PAD_CENTRE + MIN_TIP_CLEARANCE
    return max(spec.spawn_z, lowest_centre)


def tcp_target(
    object_xy: tuple[float, float], object_z: float, grasp_yaw: float, spec: ObjectSpec
) -> np.ndarray:
    """TCP position (3,) placing the jaws around `spec` at `object_xy`.

    Inverts the object's pose in the tool frame: with the tool vertical at yaw
    ``psi``, the tool's own +x axis points along ``(cos psi, sin psi, 0)`` in
    the world and its +z axis straight down, so an object sitting at TCP-frame
    ``(lateral, 0, TCP_TO_PAD_CENTRE)`` is at
    ``tcp + (lateral cos psi, lateral sin psi, -TCP_TO_PAD_CENTRE)``.

    Note the yaw dependence: the grasp yaw does not merely spin the wrist, it
    swings *where the tool has to stand*, by up to :func:`pad_lateral_offset`
    (16 mm). That is why the yaw branch has to be chosen before the IK rather
    than after it, and why a solver that silently pi-flips the yaw would put the
    tool 32 mm off target -- see :func:`plan_grasp`.
    """
    lateral = pad_lateral_offset(spec)
    return np.array(
        [
            object_xy[0] - lateral * math.cos(grasp_yaw),
            object_xy[1] - lateral * math.sin(grasp_yaw),
            object_z + TCP_TO_PAD_CENTRE,
        ]
    )


YAW_PERIOD_RAD: dict[str, float] = {"quarter": math.pi / 2, "half": math.pi, "free": 0.0}
"""Yaw period, in radians, of each :attr:`~manus.objects.ObjectSpec.yaw_symmetry` class."""


def yaw_symmetry(spec: ObjectSpec) -> float:
    """Rotation about +z (radians) that maps `spec` onto itself, 0 if continuous.

    Reads the object's declared symmetry class rather than re-deriving it:
    :class:`~manus.objects.ObjectSpec` validates the declaration against the
    geometry at construction, so the catalogue is the single place a wrong
    answer can enter, and it cannot enter quietly.

    A square-section cuboid answers a quarter turn, a rectangular one a half
    turn (only two of the four quarter turns put the jaws across its short
    axis), and anything round answers 0 -- continuous, no constraint at all.
    """
    return YAW_PERIOD_RAD[spec.yaw_symmetry]


def grasp_yaw_candidates(
    spec: ObjectSpec, object_yaw: float, tool_yaw: float
) -> tuple[float, ...]:
    """Grasp yaws to try for `spec` at `object_yaw`, best first.

    A grasp yaw has to line the jaws up across the object's grasp axis, so for
    an object with a yaw period the admissible set is
    ``object_yaw + k * yaw_symmetry(spec)`` -- four branches for a cube, two for
    a domino (the other two would ask the jaws to span its 40 mm length), and
    for a round object no constraint at all, in which case the *tool's own*
    current yaw is taken as the base instead. The preferred branch is the one
    nearest the tool's current yaw, which is the cheapest wrist_roll travel and
    keeps that joint (the one with the least margin: 320 deg of travel, so a
    40 deg band of tool yaws is unreachable at any given pan) furthest from its
    stops.

    All **four** quarter turns are distinct here even though parallel jaws are
    pi-periodic, because this hand is not symmetric: one jaw is fixed to the
    wrist and only the other moves, so a pi flip puts the static finger on the
    opposite side of the object and moves the tool 32 mm
    (:func:`pad_lateral_offset` twice over). The extra branches are also what
    make the far corners of the region reachable -- the offset can point inward
    instead of outward. That is why a round object gets quarter turns off the
    tool yaw as *fallbacks* rather than the single candidate its symmetry would
    justify: they are placement branches, not symmetry branches, and dropping
    them would forfeit reachability at the region's corners for nothing.

    Args:
        spec: Object being grasped.
        object_yaw: Object yaw about world +z, radians.
        tool_yaw: Current :meth:`KinematicChain.tool_yaw`, radians.

    Returns:
        Grasp yaws in preference order, radians: nearest branch first.
    """
    period = yaw_symmetry(spec)
    base = object_yaw if period else tool_yaw
    step = period or math.pi / 2
    candidates = [base + index * step for index in range(round(2 * math.pi / step))]
    return tuple(sorted(candidates, key=lambda yaw: abs(_wrap(yaw - tool_yaw))))


def plan_lift(
    q_grasp: np.ndarray,
    rise: float = DEFAULT_CONFIG.lift_rise,
    *,
    step: float = 0.01,
    max_steps: int = 400,
) -> tuple[np.ndarray, float]:
    """Retract from `q_grasp` to a raised pose, in joint space.

    Gradient flow of TCP height over the three pitch joints (shoulder_lift,
    elbow_flex, wrist_flex), clamped to ``specs.JOINT_LIMITS`` at every step,
    with shoulder_pan and wrist_roll held: the object stays over the spot it was
    picked from and the wrist does not spin the grasp. Deliberately *not*
    tool-vertical -- the vertical-tool family caps TCP height at 0.0903 m, only
    ~7 cm above a cube grasp, so insisting on it would forfeit most of the
    region (measured: 20% of the region has no vertical-tool IK solution at the
    lift height). Following the height gradient instead reaches >= 0.09 m
    everywhere in :data:`~manus.kinematics.GRASP_REGION`, at the cost of tilting
    the tool ~30 deg, which the squeeze holds against easily.

    Args:
        q_grasp: Shape (5,) arm pose the lift starts from, radians.
        rise: TCP height gain to stop at, metres.
        step: Joint-space step per gradient iteration, radians.
        max_steps: Iteration ceiling (a joint limit can stall the flow early).

    Returns:
        ``(q_lift, achieved_rise)`` -- the (5,) raised pose in radians and the
        TCP rise it delivers by FK, in metres. The rise falls short of `rise`
        only when the joint limits stop the flow.
    """
    q = np.clip(np.asarray(q_grasp, dtype=float), ARM_LOWER, ARM_UPPER)
    start_height = float(_CHAIN.fk_tcp(q)[0][2])
    pitch = (1, 2, 3)
    for _ in range(max_steps):
        if float(_CHAIN.fk_tcp(q)[0][2]) - start_height >= rise:
            break
        gradient = np.zeros(kinematics.NUM_ARM_JOINTS)
        for index in pitch:
            delta = np.zeros(kinematics.NUM_ARM_JOINTS)
            delta[index] = 1e-5
            gradient[index] = (
                float(_CHAIN.fk_tcp(q + delta)[0][2]) - float(_CHAIN.fk_tcp(q - delta)[0][2])
            ) / 2e-5
        norm = float(np.linalg.norm(gradient))
        if norm < 1e-9:  # stationary point: no pitch motion raises the tool
            break
        stepped = np.clip(
            q + step * gradient / norm, ARM_LOWER, ARM_UPPER
        )
        if np.array_equal(stepped, q):  # every free joint is against its limit
            break
        q = stepped
    return q, float(_CHAIN.fk_tcp(q)[0][2]) - start_height


@dataclass(frozen=True)
class GraspPlan:
    """The waypoints one attempt is driven through, all in radians / metres.

    Attributes:
        grasp_yaw: Tool yaw the grasp is planned at.
        q_pregrasp: Arm pose hovering above the object.
        q_grasp: Arm pose at grasp height.
        q_lift: Arm pose after the retraction.
        tcp_pregrasp: TCP position the pregrasp pose was solved for.
        tcp_grasp: TCP position the grasp pose was solved for.
        lateral_offset: TCP-frame x the object centre was aimed at, metres
            (:func:`pad_lateral_offset`).
        lift_rise: TCP height gain from :attr:`q_grasp` to :attr:`q_lift`.
        close_target: Jaw target used by CLOSE.
        ik_converged: Whether both IK solves met their tolerances.
        reason: ``""`` when :attr:`ok`, else why the plan is not trustworthy.
    """

    grasp_yaw: float
    q_pregrasp: np.ndarray
    q_grasp: np.ndarray
    q_lift: np.ndarray
    tcp_pregrasp: np.ndarray
    tcp_grasp: np.ndarray
    lateral_offset: float
    lift_rise: float
    close_target: float
    ik_converged: bool
    reason: str = ""

    @property
    def ok(self) -> bool:
        """Whether the plan is feasible: IK converged and the lift clears."""
        return self.reason == ""

    def waypoint(self, state: str) -> np.ndarray:
        """Arm waypoint of an arm state (`PREGRASP`, `DESCEND` or `LIFT`)."""
        return {PREGRASP: self.q_pregrasp, DESCEND: self.q_grasp, LIFT: self.q_lift}[state]


def plan_grasp(
    spec: ObjectSpec,
    placement: EpisodeDraw | Sequence[float],
    q_current: np.ndarray | None = None,
    config: ExpertConfig = DEFAULT_CONFIG,
) -> GraspPlan:
    """Solve the three waypoints for grasping `spec` at `placement`.

    Branches are tried in :func:`grasp_yaw_candidates` order and the first one
    that survives *three* checks is taken: both IK solves converge, and the yaw
    they actually deliver is the yaw that was asked for. That last check is not
    paranoia -- :func:`~manus.kinematics.ik_solve` is documented to solve the
    pi-flipped pose when wrist_roll cannot reach the requested yaw, which is a
    free substitution for a symmetric gripper and a 32 mm miss for this one
    (:func:`tcp_target`). The flipped yaw is a candidate in its own right, with
    its own tool position, so rejecting the substitution loses nothing.

    The two IK solves are run *cold* (no warm start) even though
    :func:`~manus.kinematics.ik_solve` accepts one: inside the grasp region the
    analytic seed is exact to ~0.02 mm and the refinement never runs, whereas a
    warm start stops as soon as it is inside the 1 mm tolerance. Cold is both
    more accurate here and independent of the measured pose, so two attempts at
    the same placement plan identically.

    Args:
        spec: Object to grasp; supplies the grasp height and the close target.
        placement: The attempt's :class:`~manus.randomize.EpisodeDraw`, or a
            bare ``(x, y, yaw)`` in metres/radians.
        q_current: Shape (5,) arm pose the attempt starts from, radians; used
            only to pick the grasp-yaw branch. None means the home pose.
        config: Tunables; :attr:`~ExpertConfig.hover_height`,
            :attr:`~ExpertConfig.lift_rise` and
            :attr:`~ExpertConfig.min_lift_rise` are read.

    Returns:
        The :class:`GraspPlan`. An infeasible request still comes back as a
        plan -- with ``ok`` False and ``reason`` set, and waypoints holding
        ``ik_solve``'s best in-limit effort -- so a caller can run the attempt
        anyway and report an honest failure rather than a skipped sample.
    """
    x, y, object_yaw = _placement(placement)
    start = (
        np.zeros(kinematics.NUM_ARM_JOINTS)
        if q_current is None
        else np.asarray(q_current, dtype=float).reshape(-1)[: kinematics.NUM_ARM_JOINTS]
    )
    best: GraspPlan | None = None
    for grasp_yaw in grasp_yaw_candidates(spec, object_yaw, _CHAIN.tool_yaw(start)):
        tcp_grasp = tcp_target((x, y), grasp_height(spec), grasp_yaw, spec)
        tcp_pregrasp = tcp_grasp + np.array([0.0, 0.0, config.hover_height])
        q_pregrasp, pregrasp_ok = ik_solve(tcp_pregrasp, grasp_yaw)
        q_grasp, grasp_ok = ik_solve(tcp_grasp, grasp_yaw)
        q_lift, lift_rise = plan_lift(q_grasp, config.lift_rise)
        flipped = max(
            abs(_wrap(_CHAIN.tool_yaw(q_pregrasp) - grasp_yaw)),
            abs(_wrap(_CHAIN.tool_yaw(q_grasp) - grasp_yaw)),
        ) > YAW_MATCH_TOL
        reason = ""
        if not (pregrasp_ok and grasp_ok):
            missed = "pregrasp" if not pregrasp_ok else "grasp"
            reason = f"ik_{missed}_unreachable"
        elif flipped:
            reason = "ik_solved_the_flipped_yaw"
        elif lift_rise < config.min_lift_rise:
            reason = f"lift_rise_{lift_rise * 1e3:.0f}mm_below_minimum"
        plan = GraspPlan(
            grasp_yaw=float(grasp_yaw),
            q_pregrasp=q_pregrasp,
            q_grasp=q_grasp,
            q_lift=q_lift,
            tcp_pregrasp=tcp_pregrasp,
            tcp_grasp=tcp_grasp,
            lateral_offset=pad_lateral_offset(spec),
            lift_rise=lift_rise,
            close_target=spec.close_target_rad,
            ik_converged=bool(pregrasp_ok and grasp_ok),
            reason=reason,
        )
        if plan.ok:
            return plan
        if best is None:
            best = plan
    assert best is not None  # grasp_yaw_candidates never returns empty
    return best


# --- The FSM ------------------------------------------------------------------


@dataclass(frozen=True)
class StateReport:
    """What one state did, recorded as the FSM leaves it.

    Attributes:
        state: The state's name.
        steps: Steps spent in it.
        exit: ``"converged"``, ``"stalled"``, ``"elapsed"`` (HOLD ran its
            course) or ``"timeout"`` (the budget expired first).
        joint_error: ``max |measured - waypoint|`` at exit, radians; 0 for
            states with no arm waypoint.
        joint_errors: The same, per arm joint, in
            :data:`~manus.kinematics.ARM_JOINT_NAMES` order.
        bias: Droop bias in force at exit, per arm joint, radians.
        tcp_error: Distance from the measured TCP to the waypoint's TCP by FK,
            metres; None for states with no arm waypoint.
        gripper: Measured jaw angle at exit, radians.
    """

    state: str
    steps: int
    exit: str
    joint_error: float
    joint_errors: tuple[float, ...]
    bias: tuple[float, ...]
    tcp_error: float | None
    gripper: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping of the report."""
        return {
            "state": self.state,
            "steps": self.steps,
            "exit": self.exit,
            "joint_error": self.joint_error,
            "joint_errors": list(self.joint_errors),
            "bias": list(self.bias),
            "tcp_error": self.tcp_error,
            "gripper": self.gripper,
        }


class ScriptedGraspExpert:
    """Waypoint FSM that grasps one object at one placement.

    Construct once per object (planning is cheap but not free), then
    :meth:`reset` per attempt and call :meth:`step` once per control step until
    :attr:`done`.

    Args:
        spec: Object to grasp.
        placement: Optional placement to plan immediately -- an
            :class:`~manus.randomize.EpisodeDraw` or ``(x, y, yaw)``. Passing it
            here is the same as passing it to :meth:`reset`.
        config: Tunables; see :class:`ExpertConfig`.

    Attributes:
        spec: The object spec handed in.
        config: The configuration in force.
    """

    def __init__(
        self,
        spec: ObjectSpec,
        placement: EpisodeDraw | Sequence[float] | None = None,
        *,
        config: ExpertConfig = DEFAULT_CONFIG,
    ) -> None:
        self.spec = spec
        self.config = config
        self._plan: GraspPlan | None = None
        self._state: str = PREGRASP
        self._state_step: int = 0
        self._total_steps: int = 0
        self._entry_arm: np.ndarray | None = None
        self._entry_gripper: float = 0.0
        self._bias = np.zeros(kinematics.NUM_ARM_JOINTS)
        self._frozen_arm: np.ndarray | None = None
        self._last_arm_command = np.zeros(kinematics.NUM_ARM_JOINTS)
        self._gripper_history: deque[float] = deque(maxlen=config.gripper_stall_window + 1)
        self._reports: list[StateReport] = []
        self._timeouts: list[str] = []
        self._started = False
        if placement is not None:
            self.reset(placement)

    # -- lifecycle -------------------------------------------------------------

    def reset(
        self,
        placement: EpisodeDraw | Sequence[float] | None = None,
        q_current: np.ndarray | Sequence[float] | None = None,
    ) -> GraspPlan:
        """Re-plan and rewind the FSM to PREGRASP.

        Args:
            placement: The attempt's placement; None re-uses the one already
                planned (and raises if there is none).
            q_current: Optional (5,) arm pose or (6,) measurement the attempt
                starts from, radians. Only the grasp-yaw branch depends on it;
                None means the home pose.

        Returns:
            The :class:`GraspPlan` now in force -- check :attr:`GraspPlan.ok`
            before trusting the attempt.
        """
        if placement is None and self._plan is None:
            raise ValueError("no placement to reset to; pass one here or to the constructor")
        if placement is not None:
            start = None
            if q_current is not None:
                start = np.asarray(q_current, dtype=float).reshape(-1)[
                    : kinematics.NUM_ARM_JOINTS
                ]
            self._plan = plan_grasp(self.spec, placement, start, self.config)

        self._state = PREGRASP
        self._state_step = 0
        self._total_steps = 0
        self._entry_arm = None
        self._entry_gripper = 0.0
        self._bias = np.zeros(kinematics.NUM_ARM_JOINTS)
        self._frozen_arm = None
        self._last_arm_command = np.zeros(kinematics.NUM_ARM_JOINTS)
        self._gripper_history.clear()
        self._reports = []
        self._timeouts = []
        self._started = True
        assert self._plan is not None
        return self._plan

    # -- read-only surface -----------------------------------------------------

    @property
    def state(self) -> str:
        """Current state, one of :data:`STATE_SEQUENCE`."""
        return self._state

    @property
    def done(self) -> bool:
        """Whether the FSM has finished (``state == "DONE"``)."""
        return self._state == DONE

    @property
    def plan(self) -> GraspPlan:
        """The plan in force. Raises if :meth:`reset` has not run."""
        if self._plan is None:
            raise RuntimeError("expert has no plan yet; call reset(placement) first")
        return self._plan

    @property
    def timeouts(self) -> list[str]:
        """States whose step budget expired, in the order they did, with repeats."""
        return list(self._timeouts)

    @property
    def reports(self) -> list[StateReport]:
        """One :class:`StateReport` per state left so far, in order."""
        return list(self._reports)

    @property
    def bias(self) -> np.ndarray:
        """Current droop bias per arm joint, radians (commanded minus waypoint)."""
        return self._bias.copy()

    @property
    def total_steps(self) -> int:
        """Steps issued since :meth:`reset`."""
        return self._total_steps

    @property
    def state_step(self) -> int:
        """Steps issued in the current state."""
        return self._state_step

    # -- the loop --------------------------------------------------------------

    def step(
        self, measured_joint_pos: Mapping[str, float] | Sequence[float] | np.ndarray
    ) -> dict[str, float]:
        """Advance the FSM by one control step and return the joint targets.

        The measurement is the state that resulted from the *previous* command,
        so each call first decides whether the state it is in has finished, then
        issues the command for whichever state is current afterwards. Calling
        after :attr:`done` is harmless: the final hold command is repeated.

        Args:
            measured_joint_pos: All six measured joint positions, as a mapping
                keyed by :data:`manus.specs.JOINT_NAMES` or a sequence in that
                order, radians.

        Returns:
            All six joint targets, radians, clamped to ``specs.JOINT_LIMITS``.
        """
        if not self._started:
            raise RuntimeError("call reset(placement) before step()")
        measured = joint_vector(measured_joint_pos)
        arm = measured[: kinematics.NUM_ARM_JOINTS]
        gripper = float(measured[GRIPPER_INDEX])

        if self._entry_arm is None:  # first step of the attempt
            self._enter(self._state, arm, gripper)
        self._gripper_history.append(gripper)

        exit_reason = self._exit_reason(arm)
        if exit_reason is not None:
            self._leave(exit_reason, arm, gripper)

        targets = self._command(arm, gripper)
        self._state_step += 1
        self._total_steps += 1
        return targets

    # -- internals -------------------------------------------------------------

    def _enter(self, state: str, arm: np.ndarray, gripper: float) -> None:
        """Latch the pose a state starts ramping from."""
        self._state = state
        self._state_step = 0
        self._entry_arm = arm.copy()
        self._entry_gripper = gripper
        self._gripper_history.clear()

    def _leave(self, exit_reason: str, arm: np.ndarray, gripper: float) -> None:
        """Record the state being left and enter the next one."""
        waypoint = self._waypoint()
        if waypoint is None:
            errors: tuple[float, ...] = ()
            tcp_error = None
        else:
            errors = tuple(float(value) for value in np.abs(arm - waypoint))
            tcp_error = float(
                np.linalg.norm(_CHAIN.fk_tcp(arm)[0] - _CHAIN.fk_tcp(waypoint)[0])
            )
        self._reports.append(
            StateReport(
                state=self._state,
                steps=self._state_step,
                exit=exit_reason,
                joint_error=max(errors) if errors else 0.0,
                joint_errors=errors,
                bias=tuple(float(value) for value in self._bias),
                tcp_error=tcp_error,
                gripper=gripper,
            )
        )
        if exit_reason == "timeout":
            self._timeouts.append(self._state)
        if self._state == DESCEND:
            # CLOSE holds the arm exactly where DESCEND left it -- including the
            # droop bias -- so the jaws close on the pose that was reached, not
            # on a recomputed one that would nudge the arm mid-grasp.
            self._frozen_arm = self._last_arm_command.copy()
        if self._state == LIFT:
            self._frozen_arm = self._last_arm_command.copy()
        self._enter(STATE_SEQUENCE[STATE_SEQUENCE.index(self._state) + 1], arm, gripper)

    def _waypoint(self) -> np.ndarray | None:
        """Arm waypoint of the current state, or None if it does not move the arm."""
        if self._state in ARM_STATES:
            return self.plan.waypoint(self._state)
        return None

    def _ramp(self) -> float:
        """Fraction of the current state's ramp already commanded, in [0, 1]."""
        span = max(1, self.config.ramp_steps(self._state))
        return min(1.0, self._state_step / span)

    def _exit_reason(self, arm: np.ndarray) -> str | None:
        """Why the current state should end now, or None to stay in it."""
        if self._state == DONE:
            return None
        budget_spent = self._state_step >= self.config.state_budget
        if self._state == HOLD:
            return "elapsed" if self._state_step >= self.config.hold_steps else None
        if self._state in ARM_STATES:
            waypoint = self.plan.waypoint(self._state)
            converged = float(np.abs(arm - waypoint).max()) < self.config.converge_tol
            if self._ramp() >= 1.0 and converged:
                return "converged"
            return "timeout" if budget_spent else None
        # CLOSE: the jaws stop moving once they are squeezing the object (or
        # each other) -- a position proxy for effort saturation, since the
        # implicit actuator's applied torque is not part of the contract here.
        if self._ramp() >= 1.0 and self._gripper_stalled():
            return "stalled"
        return "timeout" if budget_spent else None

    def _gripper_stalled(self) -> bool:
        """Whether the jaws have stopped moving over the stall window."""
        window = self.config.gripper_stall_window
        if len(self._gripper_history) <= window:
            return False
        recent = list(self._gripper_history)[-(window + 1) :]
        return (max(recent) - min(recent)) < self.config.gripper_stall_tol

    def _command(self, arm: np.ndarray, gripper: float) -> dict[str, float]:
        """Targets for the current state at the current step."""
        assert self._entry_arm is not None
        alpha = min(1.0, (self._state_step + 1) / max(1, self.config.ramp_steps(self._state)))
        config = self.config

        if self._state in ARM_STATES:
            waypoint = self.plan.waypoint(self._state)
            self._update_bias(arm, waypoint, alpha)
            arm_command = self._entry_arm + alpha * (waypoint - self._entry_arm) + self._bias
        else:
            # CLOSE / HOLD / DONE: whatever the previous state finished holding.
            arm_command = (
                self._frozen_arm if self._frozen_arm is not None else self._last_arm_command
            )

        if self._state == PREGRASP:
            gripper_command = self._entry_gripper + alpha * (
                config.gripper_open - self._entry_gripper
            )
        elif self._state == DESCEND:
            gripper_command = config.gripper_open
        elif self._state == CLOSE:
            gripper_command = self._entry_gripper + alpha * (
                self.plan.close_target - self._entry_gripper
            )
        else:  # LIFT, HOLD, DONE -- keep squeezing
            gripper_command = self.plan.close_target

        self._last_arm_command = np.asarray(arm_command, dtype=float).copy()
        targets = {
            name: float(value)
            for name, value in zip(kinematics.ARM_JOINT_NAMES, arm_command, strict=True)
        }
        targets[GRIPPER_NAME] = float(gripper_command)
        return clamp_targets(targets)

    def _update_bias(self, arm: np.ndarray, waypoint: np.ndarray, alpha: float) -> None:
        """Fold the standing joint error into the droop bias (see the module docstring).

        Gated per joint, and leaky. Both halves are load-bearing:

        * *Per joint*, because a single joint jammed against the object holds an
          error far bigger than droop while its neighbours are quietly settling;
          a shared gate would freeze the lot.
        * *Leaky*, because a bias that only ever grows deadlocks. A joint that
          takes a large error into the next state -- exactly what happens when
          the previous state timed out -- would keep the error above the gate
          with its own stale offset, and the gate would keep the offset from
          being corrected. Bleeding it away restores the arm to plain PD control
          within a couple of seconds, and the integrator then re-engages by
          itself. (Found in sim: a jammed DESCEND left a 0.3 rad bias that LIFT
          could not shed, and the arm never moved again.)
        """
        if self.config.droop_gain == 0.0 or alpha < 1.0:
            return  # still ramping: the error is travel, not droop
        error = waypoint - arm
        settling = np.abs(error) < self.config.droop_engage
        self._bias = np.clip(
            np.where(
                settling,
                self._bias + self.config.droop_gain * error,
                self._bias * self.config.droop_leak,
            ),
            -self.config.droop_limit,
            self.config.droop_limit,
        )

    # -- reporting -------------------------------------------------------------

    def telemetry(self) -> dict[str, Any]:
        """JSON-ready summary of the attempt so far (plan, states, timeouts)."""
        plan = self.plan
        return {
            "object": self.spec.name,
            "grasp_yaw": plan.grasp_yaw,
            "ik_converged": plan.ik_converged,
            "plan_ok": plan.ok,
            "plan_reason": plan.reason,
            "lift_rise": plan.lift_rise,
            "close_target": plan.close_target,
            "state": self._state,
            "total_steps": self._total_steps,
            "timeouts": list(self._timeouts),
            "states": [report.to_dict() for report in self._reports],
        }


# --- Success --------------------------------------------------------------------

SUCCESS_LIFT_M: float = 0.05
"""Object rise above its spawn height that counts as lifted, metres."""

SUCCESS_SUSTAIN_STEPS: int = 30
"""Consecutive steps the lift has to hold for."""

GRIPPER_HELD_MAX_RAD: float = 1.0
"""Jaw angle at or below which the gripper counts as closed on something.

The jaws are commanded to ``close_target_rad`` (below contact) from CLOSE
onwards and stall wherever the object stops them: measured over the 200-attempt
Step 8 gate, a held 30 mm cube stops them at 0.27-0.35 rad, and an empty hand
runs on down to the 0.05 rad target. This bar sits above the loaded stall and
well below :data:`manus.control.GRIPPER_OPEN` (1.5), so a hand that dropped the
object cannot satisfy the predicate while a real grasp always can.
"""


class GraspSuccessMonitor:
    """The success predicate: object lifted 5 cm, held 30 steps, jaws closed.

    Stateful because "sustained" means *consecutive* steps: any step where the
    object drops back below the bar, or the jaws are open, restarts the count.
    Once :attr:`success` latches True it stays True -- a grasp that succeeded
    and was then thrown away by a later state still succeeded at the moment the
    predicate was met, and the episode is cut there by the driver anyway.

    Args:
        spawn_z: Height the object rests at, metres (``ObjectSpec.spawn_z``).
        lift: Rise required above `spawn_z`, metres.
        sustain: Consecutive steps required.
        gripper_max: Jaw angle at or below which the gripper counts as closed.
    """

    def __init__(
        self,
        spawn_z: float,
        *,
        lift: float = SUCCESS_LIFT_M,
        sustain: int = SUCCESS_SUSTAIN_STEPS,
        gripper_max: float = GRIPPER_HELD_MAX_RAD,
    ) -> None:
        self.spawn_z = float(spawn_z)
        self.lift = float(lift)
        self.threshold_z = self.spawn_z + self.lift
        self.sustain = int(sustain)
        self.gripper_max = float(gripper_max)
        self.streak = 0
        self.best_streak = 0
        self.peak_z = -np.inf
        self.success = False

    def update(self, object_z: float, gripper_pos: float) -> bool:
        """Fold in one step's privileged state; returns :attr:`success`."""
        height = float(object_z)
        self.peak_z = max(self.peak_z, height)
        if height >= self.threshold_z and float(gripper_pos) <= self.gripper_max:
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            if self.streak >= self.sustain:
                self.success = True
        else:
            self.streak = 0
        return self.success

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary of what the predicate saw."""
        return {
            "success": self.success,
            "peak_z": None if self.peak_z == -np.inf else float(self.peak_z),
            "threshold_z": self.threshold_z,
            "best_streak": self.best_streak,
            "sustain": self.sustain,
        }


NUDGE_M: float = 0.02
"""Object rise that separates "the jaws moved it" from "the jaws missed", metres."""


def classify_outcome(
    expert: ScriptedGraspExpert, monitor: GraspSuccessMonitor, *, nudge: float = NUDGE_M
) -> str:
    """Name what happened, from the expert's telemetry and the height trace.

    The taxonomy the gate report is grouped by. Order matters -- an attempt can
    satisfy several clauses, and the first one that fires is the most specific
    explanation:

    ``success``
        The predicate was met.
    ``ik_infeasible``
        The plan itself was never solvable (edge of region, or no lift). The
        attempt still ran, so this is an outcome rather than a skip.
    ``slipped``
        The object cleared the bar at some point but did not stay there --
        squeeze, friction or the lift tilt.
    ``short_lift``
        It came up but never reached the bar: the arm is holding it too low,
        or the retraction stalled against a joint limit.
    ``timeout``
        It barely moved *and* a state ran out of budget -- the arm never got
        where it was going, so no grasp was properly attempted.
    ``no_grasp``
        It barely moved and every state converged: the jaws closed on air or
        knocked the object aside. The interesting failure.

    Args:
        expert: The expert that ran the attempt (after it finished).
        monitor: The success monitor fed that attempt's privileged state.
        nudge: Rise above which the object counts as having been moved, metres.

    Returns:
        One of the names above.
    """
    if monitor.success:
        return "success"
    if not expert.plan.ok:
        return "ik_infeasible"
    peak = monitor.peak_z
    if peak >= monitor.threshold_z:
        return "slipped"
    if peak >= monitor.spawn_z + nudge:
        return "short_lift"
    if expert.timeouts:
        return "timeout"
    return "no_grasp"
