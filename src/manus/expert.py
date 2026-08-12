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
    Tool vertical, jaws open, TCP at :func:`pregrasp_height` -- the fixed
    stand-off above the grasp point, or higher if that would put the fingers
    inside a tall object. Reached from wherever the arm happens to be.
``DESCEND``
    Straight down to the grasp pose: the *jaws* around the object, which is not
    the TCP on the object -- see :func:`tcp_target` and :func:`grasp_height`.
    Also not the object's mid-height on anything too short or too tall for the
    hand's parallel band (:data:`JAW_PARALLEL_REACH`).
``SEAT``
    **Only for an object that asks for it** (:func:`seats`), between the
    approach and CLOSE, and it is the one state that exists to stop the jaws
    doing something rather than to move the arm somewhere new. The approach
    deliberately leaves the object :data:`JAW_CLEARANCE` clear of the static
    pad so the hand can get past it; SEAT closes that 2 mm with the *arm*, at
    :data:`SEAT_CREEP_RATE_M`, jaws still wide, so the moving jaw finds the
    object already against its stop. See :data:`SEAT_GAP_M` for why the arm can
    be trusted to touch an object the jaw cannot.
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

**Side grasps** (``spec.grasp_mode == "side"``, :data:`SIDE_STATE_SEQUENCE`)
walk the same five states with DESCEND replaced by ADVANCE, and they are a
different plan rather than the same plan rotated:

``PREGRASP``
    Tool **flat**, approach axis horizontal and pointing radially outward, jaws
    open and level; standing :attr:`ExpertConfig.side_retract` back from the
    grasp along that axis, at the grasp's own height -- the *cup* height, two
    thirds of the way up the object (:func:`grasp_height`). The hand never
    passes over the object.
``ADVANCE``
    Straight out along the approach axis onto the grasp pose, so the object ends
    up between the pads -- the same ``(lateral, 0, TCP_TO_PAD_CENTRE)`` in the
    tool's own frame as a top-down grasp, which in the world is now *tangential*
    and *radial* rather than lateral and vertical (:func:`side_tcp_target`).
    Then it **waits**: a side move may not exit until it has held the waypoint
    for :data:`SIDE_SETTLE_STEPS` past its ramp, because what a side approach's
    leftover droop is spent out of is the hand's own table clearance
    (:func:`side_table_clearance`, :data:`SIDE_CONVERGE_TOL`).
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

SEAT, CLOSE, LIFT and HOLD are shared verbatim: the jaws do not know which way
the hand is pointing, and the lift is a pitch retraction either way -- it tips a
side-held cylinder as it raises it, which the squeeze holds against just as it
holds against the ~30 deg the top-down lift already tilts through. SEAT is
shared because it is defined in the *tool* frame too -- it always moves along
the tool's own -x, which is a horizontal slide across the object's face in
either mode (straight sideways when the tool is vertical, tangential when it is
flat), never along the approach.

Three behaviours are worth knowing before reading the code:

**The tool frame is not the grasp frame.** The jaws straddle the *wrist_roll
axis*, which the TCP hangs 7.9 mm off, and the static finger fills everything on
one side of it. Aiming the TCP at the object -- the obvious reading of "grasp at
(x, y)" -- lands that finger on the object's lid and closes the hand on air.
:func:`tcp_target` aims the jaws instead, which makes the grasp yaw a *position*
decision as well as an orientation one; see the jaw-geometry block below.

**Servo-to-converge.** An arm state ends when the measured joints actually
reach the waypoint (``max |measured - waypoint| < converge_tol(spec)``), not
when a step counter says so; a per-state budget only exists so a wedged attempt
cannot hang the generator, and every expiry is recorded in
:attr:`ScriptedGraspExpert.timeouts`. The bar is object-scaled because the
error it admits is a fixed number of millimetres at the tool and objects are
not a fixed number of millimetres wide -- see :func:`converge_tol`.

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

from manus import kinematics, objects, specs
from manus.control import GRIPPER_OPEN, clamp_targets
from manus.kinematics import TCP_TO_PAD_CENTRE, KinematicChain, ik_solve
from manus.objects import ObjectSpec
from manus.randomize import EpisodeDraw

__all__ = [
    "CLOSE_CREEP_LEAD_RAD",
    "CLOSE_CREEP_RATE_RAD",
    "CONVERGE_TOL",
    "FREEZE_STATES",
    "JAWS_OPEN_STATES",
    "SEAT",
    "SEAT_CONTACT_REF_STEP",
    "SEAT_CONTACT_STEPS",
    "SEAT_CONTACT_TOL",
    "SEAT_CONVERGE_TOL",
    "SEAT_CREEP_RATE_M",
    "SEAT_GAP_M",
    "SEAT_SETTLE_STEPS",
    "SIDE_CONVERGE_TOL",
    "SIDE_SETTLE_STEPS",
    "SIDE_STATE_SEQUENCE",
    "STATE_SEQUENCE",
    "ExpertConfig",
    "GraspPlan",
    "GraspSuccessMonitor",
    "ScriptedGraspExpert",
    "StateReport",
    "classify_outcome",
    "close_command",
    "close_ramp_steps",
    "close_steps",
    "converge_tol",
    "grasp_height",
    "grasp_yaw_candidates",
    "is_side_grasp",
    "jaw_depth",
    "joint_vector",
    "plan_grasp",
    "plan_lift",
    "pregrasp_height",
    "seat_ramp_steps",
    "seat_stroke",
    "seats",
    "settle_steps",
    "side_body_behind_tcp",
    "state_budget",
    "side_table_clearance",
    "side_tcp_target",
    "state_sequence",
    "tip_clearance",
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
ADVANCE = "ADVANCE"
SEAT = "SEAT"
CLOSE = "CLOSE"
LIFT = "LIFT"
HOLD = "HOLD"
DONE = "DONE"

STATE_SEQUENCE: tuple[str, ...] = (PREGRASP, DESCEND, CLOSE, LIFT, HOLD, DONE)
"""Every state of a **top-down** grasp, in the order the FSM walks them.

``DONE`` is terminal. See :data:`SIDE_STATE_SEQUENCE` for the side grasp's,
:data:`SEAT` for the optional extra state an object can ask for, and
:func:`state_sequence` for the accessor that assembles the right one."""

SIDE_STATE_SEQUENCE: tuple[str, ...] = (PREGRASP, ADVANCE, CLOSE, LIFT, HOLD, DONE)
"""Every state of a **side** grasp: :data:`STATE_SEQUENCE` with DESCEND replaced.

The two are the same shape and the same code -- one waypoint move onto the
grasp pose, with the jaws held open -- but they are not the same motion, and
the name is what a recorded episode carries into the dataset. ADVANCE pushes the
flat hand *outward along the table* onto a standing object; DESCEND lowers it
onto one lying under it. A policy reading the two as one state would be reading
two different action distributions as one."""

ARM_STATES: frozenset[str] = frozenset({PREGRASP, DESCEND, ADVANCE, SEAT, LIFT})
"""States that move the arm and therefore end on joint convergence."""

APPROACH_STATES: frozenset[str] = frozenset({DESCEND, ADVANCE})
"""The per-mode move onto the grasp pose. Where CLOSE freezes the arm unless the
object asks for a :data:`SEAT` (:data:`FREEZE_STATES`)."""

JAWS_OPEN_STATES: frozenset[str] = APPROACH_STATES | {SEAT}
"""States that hold the jaws at :attr:`~ExpertConfig.gripper_open`.

SEAT is one of them and that is the whole point of it: the *arm* closes the last
two millimetres onto the static pad with the jaws still wide, so the object is
already against its stop before the moving jaw is asked to travel at all."""

FREEZE_STATES: frozenset[str] = APPROACH_STATES | {SEAT, LIFT}
"""States whose last arm command is latched and held by the state after them.

CLOSE holds whatever the arm last commanded, and with a SEAT in the sequence
that has to be the *seated* pose rather than the approach's -- otherwise the
2 mm the seat just closed would be handed straight back."""

CONVERGE_TOL: float = 0.02
"""``max |measured - waypoint|`` (radians, arm joints) ending a move on a
:data:`~manus.objects.REFERENCE_WIDTH_M` object -- the 30 mm cube.

Roughly 1.1 deg per joint, and reachable only because the droop bias exists
(raw steady-state droop is 2-4x this on the shoulder).

**It is a bar on the joints, not on the tool, and the two are not the same
thing.** Measured on the three filmed previews (Step 21,
``runs/object_previews/*_demo.json``): DESCEND exits on the *first* step its
ramp completes, with the largest joint error at 13-15 of the 20 mrad allowed --
and a TCP that is 5.3-5.9 mm from the waypoint, because a state that exits the
instant it is allowed to exits with the arm still moving. Only 2.2-2.5 mm of
that is vertical; the rest is lateral. CLOSE then freezes the arm exactly
there, so whatever is left is what the jaws close around. 5.9 mm is 20% of the
cube's grasp and it grasps anyway; it is 37% of the 16 mm die's, which lands
the die ~5 mm off the pads' own centreline. Hence :func:`converge_tol`, which
scales this bar by the object rather than loosening or tightening it for
everyone.
"""

SIDE_CONVERGE_TOL: float = 0.0074
"""``max |measured - waypoint|`` (radians) ending a **side** move. **Derived.**

A side grasp does not spend its convergence residual on the same thing a
top-down one does. Coming down onto an object, a millimetre of TCP error is a
millimetre off the pads' centreline; coming *along the table*, the vertical part
of it comes straight out of :func:`side_table_clearance` -- and when that budget
is gone the hand is standing on the table, which is what the filmed cylinder did
(7.4 mm of droop against 5.8 mm of clearance). So the bar is set from the
clearance rather than from the object's width:

    tol = (clearance / 2) / max_region |d z_tcp / d q|

Both halves are measured. The *worst-case* vertical TCP error a per-joint bar of
``tol`` admits is ``tol * sum_i |d z_tcp / d q_i|``, which over
:data:`~manus.kinematics.SIDE_GRASP_REGION` peaks at **0.826 m/rad** at the outer
edge -- and that worst case is the realistic one here, because gravity droops
all three pitch joints the same way (the filmed ADVANCE exit: 16.9/14.8/11.1
mrad, all the same sign, 10.7 mm of z). Half the 12.2 mm clearance at the cup
grasp is 6.1 mm, so the bar is 7.4 mrad -- and half a budget is the right share
because the other half is what the hand keeps: at exactly this bar the lowest
material still clears the table by 6.1 mm, more than :data:`MIN_TIP_CLEARANCE`.

It is a third of :data:`CONVERGE_TOL`, and it is reachable only because ADVANCE
now dwells for :data:`SIDE_SETTLE_STEPS` after its ramp instead of exiting the
first step it is allowed to (see :attr:`ExpertConfig.side_settle_steps`); the
old bar was met *while the arm was still moving*.
"""

SIDE_SETTLE_STEPS: int = 30
"""Steps a side approach holds its waypoint after the ramp, before it may exit.

One second at 30 Hz, and it is the fix for the half of the cylinder's failure
the convergence bar does not reach. The droop integrator
(:meth:`ScriptedGraspExpert._update_bias`) only runs once the ramp is done, and
folds :attr:`~ExpertConfig.droop_gain` = 12% of the standing error into the
command each step -- so it needs *steps*, and the filmed ADVANCE gave it none:
it exited on the first step its ramp completed, with 6 mrad of bias integrated
and 7.4 mm of sag still in the arm. Thirty steps of 12% takes 20 mrad of droop
to 0.4 mrad, which is deep inside :data:`SIDE_CONVERGE_TOL`; the state budget is
240, so the dwell costs a third of it at worst.

Zero for the top-down family, deliberately: DESCEND's exit rule, and therefore
every dataset already generated with it, is unchanged (:func:`settle_steps`).
"""

# --- SEAT: closing the last two millimetres with the arm ----------------------
# The measured mechanism this exists for is written up in close_command(): the
# object is *not* against the static pad when the jaws start moving -- it stands
# JAW_CLEARANCE (2 mm) clear of it by design, so the approach can pass the static
# finger -- and the moving jaw has to shove it across that gap. While it shoves,
# the commanded jaw runs on into the object and the position servo stores
# 0.5 * kp * gap^2 / JAW_WIDTH_PER_RAD^2 = 6.7 mJ, which is dumped at capture.
#
# Creeping through the shove (CLOSE_CREEP_RATE_RAD) reduced the *energy* by 340x
# and still lost both objects, because the energy was never the whole story: a
# one-sided push of 0.29 N tips the standing cylinder at any speed, and 0.29 N
# is only 0.086 mm of blocked jaw travel. There is no jaw speed at which a
# one-sided push is gentle. The only closure that is gentle is one where the
# push is *reacted by the static pad* from the first newton -- i.e. no gap.
#
# So the gap is closed by the arm instead, and the arm is a far weaker spring
# than the jaw at the same displacement, which is what makes this safe. Both
# stiffnesses are the same servo seen through different geometry -- kp divided
# by the square of how far the thing being pushed moves per radian:
#
#   jaw   kp / JAW_WIDTH_PER_RAD^2 = 17.8 / 0.0727^2 = 3.4 N/mm, everywhere
#   arm   kp / sum_i (d x_seat / d q_i)^2, swept over each object's own
#         placement region (63 points each, tests/test_expert_logic.py):
#           cylinder, seat tangential, shoulder_pan alone   0.10-0.15 N/mm
#           puck, seat partly radial, three pitch joints    0.23-1.46 N/mm
#                                                           (median 0.37)
#
# So the arm is 2-30x softer than the jaw at the same overshoot, and on the
# object that actually topples -- the cylinder, at 0.29 N -- it is softer by 23x
# and uniformly so: it takes 2.0-2.9 mm of arm overshoot to reach the force that
# 0.09 mm of jaw overshoot reaches. The seat can therefore afford to be aimed
# *at* the object and to arrive with the arm's own ~1 mm of residual, which the
# jaw never could. The puck's stiffest corner is the one place that margin is
# thin, and there SEAT_CONTACT_TOL is what carries it: what the watchdog bounds
# is the *travel*, and 0.6 mm of it against a 40 mm disc that slides at 0.29 N
# is a disc that slides half a millimetre -- towards the static pad it is being
# seated against, which is where it was going anyway.

SEAT_GAP_M: float = 0.0
"""Gap the SEAT state aims the static pad at, metres. Zero: kiss contact.

Zero rather than a small positive number because the residual is symmetric and
the two sides of it cost very different things. The arm arrives within about a
millimetre of any waypoint it converges on (measured: 0.97 mm of TCP error at
the filmed ADVANCE exit), so aiming at zero lands somewhere in +/-1 mm:

* on the *interference* side the pad presses the object with 0.1-1.5 N per
  millimetre of overshoot depending on the object and where in the region it
  sits -- and the :data:`SEAT_CONTACT_TOL` watchdog ends the state as soon as it
  sees the arm being blocked, which caps the overshoot at 0.95 mm (measured, a
  seat blocked dead at its first step) and so the press at 0.12 N on the
  cylinder, against the 0.29 N that tips it;
* on the *gap* side whatever is left is what the moving jaw still has to shove
  the object across, and the stored spring goes as the square of it: 6.7 mJ at
  the full 2 mm clearance, 1.7 mJ at 1 mm, 0.15 mJ at 0.3 mm.

Aiming at a positive gap would trade the cheap side for the expensive one.
"""

SEAT_CREEP_RATE_M: float = 5e-5
"""Tool speed through the seat, metres per control step -- 1.5 mm/s at 30 Hz.

Set by what the arm's stiffness does to a *blocked* seat: at 0.1-0.3 N/mm the
press grows 5-13 mN per step of blocked travel, so the
:data:`SEAT_CONTACT_TOL` watchdog (0.68 mm of excess tracking error, plus
:data:`SEAT_CONTACT_STEPS` of confirmation) fires around 0.1 N -- a third of the
cylinder's tipping threshold -- and 13 steps is long enough for the reading to
be a reading rather than one sample of noise. Half the jaw's own creep rate
(:data:`CLOSE_CREEP_RATE_RAD` is 0.11 mm of gap per step), which is the right
side to err on: this is the motion that is *meant* to touch.
"""

SEAT_APPROACH_SETTLE_STEPS: int = 30
"""Steps a **seating object's approach** dwells before SEAT is allowed to start.

The precondition the seat cannot do without, and the reason the earlier
1 mm-clearance experiment failed rather than a detail of it: a seat is only as
well aimed as the pose it is aimed from, and a top-down DESCEND exits *the
first step its ramp is done*, with 13-15 mrad still standing and the TCP
5.3-5.9 mm from its waypoint (:data:`CONVERGE_TOL`). Creeping the last two
millimetres out of a pose that is still six millimetres from where it thinks it
is seats nothing.

Thirty steps for the same reason :data:`SIDE_SETTLE_STEPS` is thirty -- that is
what the droop integrator needs to take 20 mrad to 0.4 -- and it buys the
contact watchdog its signal-to-noise as well: with the bias already converged,
the only thing that can move the tracking error during SEAT is the object.
Measured on the fake plant, a seat started from an unsettled DESCEND trips
:data:`SEAT_CONTACT_TOL` on the integrator's own step change, ~2.2 mrad, with
nothing in front of the pad at all.

The side family already dwells this long at ADVANCE, so this only actually adds
a dwell to a *top-down* seating object -- the 40 mm puck. Nothing that does not
seat is affected (:func:`settle_steps`).
"""

SEAT_SETTLE_STEPS: int = 20
"""Steps SEAT holds the seat pose after its ramp before it may exit.

The same job :data:`SIDE_SETTLE_STEPS` does for the approach -- give the droop
integrator steps to spend -- but a fifth of the size, because SEAT inherits an
already-converged bias and only moves the tool 2 mm. It is also the window a
*blocked* seat winds up over, which is the reason not to make it larger: the
integrator folds in 12% of the standing error per step, so a seat that ends up
pressing the object doubles that press over ~20 steps and no more.
"""

SEAT_CONVERGE_TOL: float = 0.004
"""``max |measured - waypoint|`` (radians) ending a SEAT that never touched.

4 mrad, which is 1.6-1.8 mm at the tool in either mode. Chosen to be reachable
rather than aspirational: the arm's own floor with the integrator running is
about 2 mrad (measured, the filmed PREGRASP and ADVANCE exits at 1.98 and
2.15 mrad), so 4 mrad is met a few steps after the ramp and 1.5 mrad would not
be met at all. It is deliberately *tighter* than either approach bar -- 20 mrad
top-down, 7.4 mrad side -- because the whole point of the state is that the
last two millimetres are aimed rather than thrown: a puck whose DESCEND exited
5.8 mm from its waypoint has to be brought in before the jaws are trusted.
"""

SEAT_CONTACT_TOL: float = 0.0015
"""Excess joint tracking error that counts as the pad having found the object, radians.

1.5 mrad. *Excess* over the state's own reference, sampled a few steps into the
creep (:data:`SEAT_CONTACT_REF_STEP`), so what it measures is the change since
the seat started moving rather than the standing droop -- which is several times
larger and is exactly what the bias is already cancelling. A free arm tracks the
creep with a lag that is small and constant; a blocked one banks the whole
commanded travel, every step, until this fires.

Both sides of the bar are measured on the fake plant
(``tests/test_expert_logic.py``), and it sits between them by about a factor of
two on the worse object:

* **free**, seat started from a settled approach: 0.23 mrad of excess on the
  cylinder, 0.85 mrad on the puck, over the whole state;
* **blocked**, the pad stopped dead at three different points of the creep:
  trips 13 steps later on the cylinder and 5 on the puck -- the puck's 2 mm
  costs four times the joint travel, so it banks error four times faster --
  which is 0.65 mm and 0.25 mm of blocked creep, or 0.09 N and 0.06 N standing
  on the object. Both an order below what moves either one.
"""

SEAT_CONTACT_LAG_STEPS: float = 3.0
"""Steps of commanded travel the watchdog allows a free arm to lag by.

The other half of the bar, and the half that has to be *per object*: a position
servo tracking a ramp sits a fixed fraction of a step's travel behind it, so the
lag scales with the commanded joint rate -- and the two seating objects do not
share one. The same 2 mm of tool motion costs 5.5 mrad of shoulder_pan on the
side-grasped cylinder (whose seat is tangential) and 20 mrad across the three
pitch joints on the puck (whose seat is partly radial), four times as fast, so a
bar that suits one is either deaf or trigger-happy on the other.

So the bar is ``SEAT_CONTACT_TOL + SEAT_CONTACT_LAG_STEPS * (travel per step)``:
1.9 mrad on the cylinder and 3.0 on the puck. Three steps rather than one
because the plant's own transient at the start of a ramp takes a couple of steps
to settle; more than three and the puck's bar would be looser than the contact
it is looking for.
"""

SEAT_CONTACT_STEPS: int = 3
"""Consecutive steps over the bar before contact is declared.

Three, so a single noisy sample cannot end the state early. Costs three steps of
commanded travel (0.15 mm, under 0.05 N) over the bare threshold.
"""

SEAT_CONTACT_REF_STEP: int = 2
"""Step of SEAT whose tracking error is taken as the watchdog's zero.

Not step 0 -- the plant needs a step or two to pick the ramp up -- and not much
later than that either, because the reference is a **blind spot**: contact that
is already there when it is sampled gets absorbed into it and then has to be
re-earned. Two steps is 0.1 mm of it. The lag term
(:data:`SEAT_CONTACT_LAG_STEPS`) is what buys the reference the right to be
this early; without it the reference had to be taken at step 5, and a seat that
started already touching pressed the puck with 0.38 N -- past the 0.29 N that
slides it -- before the watchdog could re-earn its threshold.
"""


@dataclass(frozen=True)
class ExpertConfig:
    """Tunables of the FSM. Defaults are the tuned Step 7 values.

    Step counts are *control* steps -- the expert is rate-agnostic, but the
    pipeline drives it at ``recorder.CONTROL_HZ`` = 30 Hz, so 30 steps is one
    second of simulated time.

    Attributes:
        hover_height: PREGRASP TCP height above the grasp point, metres.
            Top-down only; a side grasp stands off *radially* instead, by
            :attr:`side_retract`, at the grasp's own height.
        side_retract: How far back along the approach axis a side grasp's
            PREGRASP sits, metres. 40 mm, which puts the open fingertips 22.7 mm
            clear of a 30 mm cylinder's near face (:data:`JAW_TIP_Z` eats 6.3 of
            the 40, and the object's near face is 11 mm short of the TCP) --
            room for the arm's own 5-6 mm of convergence residual and for the
            object to be a little wider than declared. It is also the whole of
            the ADVANCE stroke, which is why it is not larger: every millimetre
            is a millimetre the hand travels with the jaws already open and the
            object in front of them.
        hover_margin: Clearance PREGRASP keeps between the lowest jaw material
            and the *top of the object*, metres, when that is the binding
            constraint (:func:`pregrasp_height`). Measured against the 4-6 mm of
            TCP error the arm actually converges to at PREGRASP -- droop the
            integrator has not finished cancelling -- so 4 mm of nominal margin
            can be entirely eaten by it and 8 mm leaves a real 2-4 mm.
        converge_tol: Arm convergence tolerance for a
            :data:`~manus.objects.REFERENCE_WIDTH_M` object, radians; narrower
            objects get a proportionally tighter bar (:func:`converge_tol`, and
            see :data:`CONVERGE_TOL` for why the bar is object-dependent).
            Top-down only -- a side grasp's bar comes out of its table
            clearance instead, see :attr:`side_converge_tol`.
        side_converge_tol: Arm convergence tolerance for a **side** move,
            radians (:data:`SIDE_CONVERGE_TOL`). Not width-scaled: what a side
            approach's residual is spent out of is the hand's table clearance,
            which is a property of the grasp height, not of the object's width.
        side_settle_steps: Steps a side approach holds its waypoint after the
            ramp before it is allowed to exit, so the droop integrator can
            actually cancel the sag it is measuring (:data:`SIDE_SETTLE_STEPS`).
            Top-down states never dwell.
        seat_gap: Gap the SEAT state aims the static pad at, metres
            (:data:`SEAT_GAP_M`). Only an object with
            :attr:`~manus.objects.ObjectSpec.seat_close` set has a SEAT at all.
        seat_creep_rate: Tool speed through the seat, metres per step
            (:data:`SEAT_CREEP_RATE_M`); with :attr:`seat_gap` it is what sets
            SEAT's ramp length, since the stroke is fixed by the geometry.
        seat_approach_settle: Steps the approach of a seating object dwells
            before SEAT may start (:data:`SEAT_APPROACH_SETTLE_STEPS`) -- the
            seat's precondition, not a refinement of it.
        seat_settle_steps: Steps SEAT dwells after its ramp
            (:data:`SEAT_SETTLE_STEPS`).
        seat_converge_tol: Convergence bar ending a SEAT that never found the
            object, radians (:data:`SEAT_CONVERGE_TOL`).
        seat_contact_tol: Excess tracking error at which SEAT declares contact
            and stops pushing, radians (:data:`SEAT_CONTACT_TOL`).
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
        close_ramp: Steps spent driving the jaws to the close target, or None --
            the default -- to take it from the object's mass, via
            :attr:`~manus.objects.ObjectSpec.close_ramp_steps`. **The single
            most sensitive number here.** The jaws travel ~1.45 rad from
            open to the close target, so 20 steps is 2.2 rad/s of jaw closing
            speed, or ~90 mm/s at the pads -- fast enough that the fingers'
            inward taper flicks a 30 mm cube out of the hand like a watermelon
            seed. Two of 200 gate attempts failed exactly that way (seen on
            video: the cube leaves the frame upward while the jaws close on
            nothing). 60 steps is 0.7 rad/s and the same grip once seated:
            16/16 on the probe set that contained both failures, and no slips
            among the low-friction attempts a *weaker* squeeze cost instead.
            That is the 60 g cube's number, and it is what the mass rule
            returns for it; a 5 g die needs 150. Setting this pins one ramp on
            whatever is being grasped, which is a tuning override rather than a
            default -- ``scripts/demo_expert.py --close-ramp`` is why it exists.

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
    hover_margin: float = 0.008
    side_retract: float = 0.040
    converge_tol: float = CONVERGE_TOL
    side_converge_tol: float = SIDE_CONVERGE_TOL
    side_settle_steps: int = SIDE_SETTLE_STEPS
    seat_gap: float = SEAT_GAP_M
    seat_creep_rate: float = SEAT_CREEP_RATE_M
    seat_approach_settle: int = SEAT_APPROACH_SETTLE_STEPS
    seat_settle_steps: int = SEAT_SETTLE_STEPS
    seat_converge_tol: float = SEAT_CONVERGE_TOL
    seat_contact_tol: float = SEAT_CONTACT_TOL
    state_budget: int = 240
    hold_steps: int = 45
    gripper_open: float = GRIPPER_OPEN
    gripper_stall_tol: float = 0.002
    gripper_stall_window: int = 15
    lift_rise: float = 0.09
    min_lift_rise: float = 0.06
    pregrasp_ramp: int = 45
    descend_ramp: int = 30
    close_ramp: int | None = None
    lift_ramp: int = 30
    droop_gain: float = 0.12
    droop_engage: float = 0.20
    droop_leak: float = 0.97
    droop_limit: float = 0.30

    def ramp_steps(self, state: str, spec: ObjectSpec | None = None) -> int:
        """Ramp length of `state`, in steps (1 for states that do not ramp).

        CLOSE is the only state whose ramp depends on what is being grasped:
        with :attr:`close_ramp` left at None it comes from `spec`, so pass the
        object whenever one is in hand. Without a spec an unset
        :attr:`close_ramp` falls back to the tuned reference ramp. SEAT's is
        derived rather than tuned -- the stroke is the same two millimetres for
        every object, so the ramp is just that at the creep rate
        (:func:`seat_ramp_steps`).
        """
        if state == CLOSE:
            return close_ramp_steps(spec, self)
        if state == SEAT:
            return seat_ramp_steps(self)
        return {
            PREGRASP: self.pregrasp_ramp,
            # ADVANCE is DESCEND's side-mode twin -- the same move onto the
            # grasp pose over the same distance (a 40 mm radial push against a
            # 30 mm drop), so it shares the knob rather than adding a second one.
            DESCEND: self.descend_ramp,
            ADVANCE: self.descend_ramp,
            LIFT: self.lift_ramp,
        }.get(state, 1)


DEFAULT_CONFIG = ExpertConfig()
"""The configuration :class:`ScriptedGraspExpert` uses unless told otherwise."""


def close_ramp_steps(spec: ObjectSpec | None, config: ExpertConfig = DEFAULT_CONFIG) -> int:
    """Steps CLOSE ramps over for `spec`: the config's override, else the mass rule.

    The one place the two sources are reconciled, so a driver reporting the
    ramp it used and the FSM issuing it cannot disagree.
    """
    if config.close_ramp is not None:
        return config.close_ramp
    if spec is None:
        return objects.CLOSE_RAMP_REFERENCE_STEPS
    return spec.close_ramp_steps


def seat_stroke(config: ExpertConfig = DEFAULT_CONFIG) -> float:
    """How far the SEAT state moves the tool, metres.

    The clearance the approach deliberately kept, less the gap the seat is aimed
    at: :data:`JAW_CLEARANCE` minus :attr:`~ExpertConfig.seat_gap`, 2.0 mm at
    the shipped values. Read through the module global rather than captured, so
    ``scripts/demo_expert.py --jaw-clearance`` moves the seat with the approach
    it belongs to.
    """
    return max(0.0, JAW_CLEARANCE - config.seat_gap)


def seat_ramp_steps(config: ExpertConfig = DEFAULT_CONFIG) -> int:
    """Steps SEAT creeps its stroke over: :func:`seat_stroke` at the creep rate.

    40 at the shipped values (2.0 mm at 0.05 mm/step). Nominal, not exact: the
    ramp interpolates from the pose the approach actually *reached*, so an
    approach that stopped short adds its own residual to the stroke and the real
    tool speed is a little higher than the rate. That is the right way round --
    the residual is the part that most needs bringing in.
    """
    return max(1, math.ceil(seat_stroke(config) / max(1e-9, config.seat_creep_rate)))


def settle_steps(
    state: str, spec: ObjectSpec | None, config: ExpertConfig = DEFAULT_CONFIG
) -> int:
    """Steps `state` must hold its waypoint after the ramp before it may exit.

    Four cases, and the last two are new with SEAT:

    * :attr:`~ExpertConfig.side_settle_steps` for the two states of a **side**
      grasp that have to arrive accurately -- PREGRASP and ADVANCE;
    * :attr:`~ExpertConfig.seat_settle_steps` for SEAT itself, in either mode;
    * :attr:`~ExpertConfig.seat_approach_settle` for the *approach* of an object
      that seats, so the seat is aimed from a pose that has stopped moving
      (:data:`SEAT_APPROACH_SETTLE_STEPS`);
    * zero for everything else, which leaves the top-down FSM's exit rule (and
      so every dataset generated with it) bit-for-bit what it was.

    See :data:`SIDE_SETTLE_STEPS` for what the dwell is for.
    """
    if state == SEAT:
        return config.seat_settle_steps
    if spec is None:
        return 0
    if is_side_grasp(spec) and state in (PREGRASP, ADVANCE):
        return config.side_settle_steps
    if seats(spec) and state == approach_state(spec):
        return config.seat_approach_settle
    return 0


CLOSE_CREEP_LEAD_RAD: float = 0.080
"""How far above its contact angle a creeping CLOSE drops to the creep rate, radians.

5.8 mm of jaw gap, and every millimetre of it is spoken for. The creep is only
worth anything if the jaw is *already* creeping the first time it can touch the
object, so the hand-over has to clear three things at once (measured; the first
take of the creep used 0.030 rad, cleared none of them by enough, and the fast
leg's last step landed on the cylinder):

* **the mesh's own lead over the width formula**, 0.021 rad for the side-grasped
  cylinder and 0.023 for the puck. :func:`~manus.objects.contact_angle_for_width`
  is anchored on the cube's engagement band, and an object gripped elsewhere on
  the pads meets them a little early.
* **the clearance the object is shoved across**, ``JAW_CLEARANCE /
  JAW_WIDTH_PER_RAD`` = 0.028 rad, because contact starts at the *near* face and
  the whole point is to be creeping through the shove.
* **one step of the fast ramp**, 0.017-0.024 rad, or the jaw steps straight over
  the hand-over and into the object at ramp speed.

That is 0.073 rad worst case; 0.080 leaves a step in hand. What it costs is
CLOSE's length -- :func:`close_steps` comes out at 198 steps for the cylinder
and 218 for the puck, against a tuned ramp of 60 and 85 -- which is why
:func:`state_budget` has to know about the creep.
"""

CLOSE_CREEP_RATE_RAD: float = 0.0015
"""Jaw speed through the contact band on a creeping CLOSE, radians per step.

0.11 mm of gap per control step, 16x slower than the tuned ramp's 1.76 mm. The
number is set by the object, not by the jaw: what has to be avoided is shoving
the object across :data:`JAW_CLEARANCE` faster than it can get out of the way,
and "get out of the way" has a natural period -- for the 60 mm cylinder,
``2*pi*sqrt(I_edge / m g r)`` = 0.63 s, or 19 control steps. At the tuned ramp
the 2 mm gap is crossed in 1.1 steps, which is an impact; at this rate it takes
18, which is a push. See :func:`close_command` for what that is worth in energy.
"""


def close_command(
    entry: float,
    target: float,
    step: int,
    spec: ObjectSpec | None = None,
    config: ExpertConfig = DEFAULT_CONFIG,
) -> float:
    """Jaw target at `step` of CLOSE: the ramp, with an optional creep at the end.

    The default is the tuned linear ramp over
    :meth:`~ExpertConfig.ramp_steps` -- what every top-down catalogue object
    closes with, unchanged.

    An object with :attr:`~manus.objects.ObjectSpec.close_creep` set gets a
    **two-rate** close instead: the same linear ramp down to
    :data:`CLOSE_CREEP_LEAD_RAD` above its own contact angle, then
    :data:`CLOSE_CREEP_RATE_RAD` per step through the contact band. What it is
    aimed at is measured; **what it buys is partial**, and the takes that say so
    are named at the end of this docstring:

    The jaws are a position servo, and the object is *not* against the static
    pad when they arrive -- it stands :data:`JAW_CLEARANCE` clear of it, by
    design, so the descent (or the advance) can pass the static finger. So the
    closing jaw has to shove the object across that gap with the static pad
    still 2 mm away, and while it does, the commanded jaw runs on into the
    object: the position error, and with it the force, grows with every
    millimetre of blocked travel. By virtual work through the jaw's own rate
    (:data:`~manus.objects.JAW_WIDTH_PER_RAD`), a blocked gap of ``g`` stands
    the pads on ``kp * g / rate^2`` newtons, so crossing the whole clearance
    stores ``0.5 * kp * gap^2 / rate^2`` = **6.7 mJ** of spring in the servo --
    which is then dumped into the object at capture.

    6.7 mJ is 2.4x what it takes to topple the 60 mm cylinder about its base
    edge (2.8 mJ), and it is 0.67 m/s on a 30 g puck. That is what the filmed
    takes show: the cylinder falls onto the closing finger within two control
    steps of first contact, and the puck leaves the hand at ~0.9 m/s and rides
    it 50 mm up. Creeping does not reduce the *distance* the object is shoved;
    it reduces the error the servo builds while shoving it, because the object
    now has time to move at the jaw's own speed. At the creep rate the standing
    error is one step's travel, 0.11 mm of gap, and the stored spring is
    0.02 mJ -- 340x less.

    **What it actually bought, in sim** (``runs/fix_takes`` on the rented box,
    one attempt each, the same draw as the filmed previews):

    * the 40 mm puck went from riding the hand with an empty jaw to *briefly
      held*: 6 consecutive steps of the full success predicate, against 0
      before. Still short of the 30 the predicate wants, and still
      ``not_in_hand``.
    * the cylinder's failure changed shape rather than going away. It no longer
      topples at the first touch; it leans and slides out of the pads over
      ~20 mm instead. That is the part the energy model does not cover: a
      one-sided push at a 40 mm contact height needs 0.29 N to tip a 30 mm
      cylinder and 0.64 N to slide it, so the tip is what happens *at any
      speed*, and a cylinder tilted even 3 deg is wider than the gap and gets
      levered out as the jaws keep closing. The creep buys time, not stability.

    So the creep stays because it is the right shape for the mechanism it was
    measured against and it is scoped to the two objects that need it -- but
    the thing that would actually seat these two is a closure with **no shove
    at all** (the object against the static pad before the jaws move), which is
    a plan change, not a ramp change. That plan change is the :data:`SEAT`
    state, and the two objects that creep are the two that seat: with the gap
    already closed the creep is no longer crossing anything, it is simply the
    gentlest available arrival at a surface that is already there.

    Args:
        entry: Jaw angle CLOSE was entered at, radians.
        target: The object's close target, radians.
        step: Steps completed in CLOSE (1 on the first command).
        spec: Object being grasped; supplies the contact angle and whether to
            creep at all. None means the plain ramp.
        config: Tunables; the CLOSE ramp length is read from it.

    Returns:
        The ``gripper`` joint target for this step, radians.
    """
    span = max(1, config.ramp_steps(CLOSE, spec))
    if spec is None or not spec.close_creep:
        return entry + min(1.0, step / span) * (target - entry)
    hand_over = spec.contact_angle_rad + CLOSE_CREEP_LEAD_RAD
    if hand_over <= target:  # nothing to creep through
        return entry + min(1.0, step / span) * (target - entry)
    fast_steps = max(1, math.ceil(span * (entry - hand_over) / (entry - target)))
    if step <= fast_steps:
        return entry + (step / fast_steps) * (hand_over - entry)
    crept = hand_over - CLOSE_CREEP_RATE_RAD * (step - fast_steps)
    return max(target, crept)


def close_steps(spec: ObjectSpec | None, config: ExpertConfig = DEFAULT_CONFIG) -> int:
    """Steps :func:`close_command` needs to reach the close target for `spec`.

    The ramp length for an ordinary object; the fast leg plus the creep for one
    that creeps (:data:`CLOSE_CREEP_RATE_RAD`). This is what CLOSE's own budget
    has to cover, so it is worth reading rather than assuming.
    """
    span = max(1, config.ramp_steps(CLOSE, spec))
    if spec is None or not spec.close_creep:
        return span
    entry = config.gripper_open
    hand_over = spec.contact_angle_rad + CLOSE_CREEP_LEAD_RAD
    if hand_over <= spec.close_target_rad:
        return span
    fast_steps = max(1, math.ceil(span * (entry - hand_over) / (entry - spec.close_target_rad)))
    return fast_steps + math.ceil((hand_over - spec.close_target_rad) / CLOSE_CREEP_RATE_RAD)


def state_budget(
    state: str, spec: ObjectSpec | None = None, config: ExpertConfig = DEFAULT_CONFIG
) -> int:
    """Step ceiling for `state`: :attr:`~ExpertConfig.state_budget`, or the creep.

    The budget exists so a wedged attempt cannot hang the generator, so it has
    to be longer than the state's own commanded length -- and a creeping CLOSE
    (:func:`close_steps`) is longer than the tuned ramp the 240 was written for:
    the puck's is 218 steps, and a jaw stalled on the object still needs its
    stall window on top. Only a creeping object's CLOSE is extended; every other
    state of every other object answers the configured number unchanged, so a
    test or a sweep that pins the budget down still gets exactly what it asked
    for.
    """
    if state == CLOSE and spec is not None and spec.close_creep:
        return max(
            config.state_budget,
            close_steps(spec, config) + 2 * config.gripper_stall_window,
        )
    return config.state_budget


def converge_tol(
    spec: ObjectSpec | None,
    config: ExpertConfig = DEFAULT_CONFIG,
    *,
    state: str | None = None,
) -> float:
    """Arm convergence tolerance for `spec`, radians.

    **SEAT answers :attr:`~ExpertConfig.seat_converge_tol` in either mode**, and
    is the only state that takes a bar of its own -- see
    :data:`SEAT_CONVERGE_TOL` for why it is tighter than both approach bars.
    Pass `state` to get it; the default (None) is the approach bar, which is
    what every caller that predates SEAT wants.

    A **side** grasp answers :attr:`~ExpertConfig.side_converge_tol` flat: its
    residual is spent out of the hand's table clearance rather than off the
    pads' centreline, so the bar is derived from the clearance and does not
    scale with the object's width (:data:`SIDE_CONVERGE_TOL`). Everything below
    is the top-down rule.

    :attr:`~ExpertConfig.converge_tol` scaled by the object's grasp width
    against the width it was tuned at
    (:data:`~manus.objects.REFERENCE_WIDTH_M`), and never *loosened* past it::

        tol(spec) = config.converge_tol * min(1, grasp_width / 0.030)

    The reason is measured, not stylistic. An arm state exits the first step
    its ramp is done *and* the joints are inside this bar, so the bar is what
    decides how much of the approach is still in flight when CLOSE freezes the
    arm: on the three filmed previews the exit fired at 13-15 mrad and left the
    TCP 5.3-5.9 mm off the waypoint (see :data:`CONVERGE_TOL`). That error is a
    fixed property of the *arm*, so what matters is its size relative to the
    object -- 20% of the cube's grasp width, 37% of the die's. Scaling the bar
    holds that ratio, which in practice means a narrow object's DESCEND has to
    keep stepping until the droop integrator has actually cancelled the error
    instead of exiting mid-travel with one integrator update behind it.

    The cube is the reference, so its tolerance is :data:`CONVERGE_TOL` exactly
    and its behaviour is unchanged; so is every object at least as wide (the
    cylinder, the duplo, the ball, the puck). Only the die (0.0107) and the
    domino (0.0133) are tightened.

    Args:
        spec: Object being grasped; None means the reference width.
        config: Tunables; :attr:`~ExpertConfig.converge_tol` is read.
        state: The state being exited, or None for the mode's approach bar.

    Returns:
        The tolerance, radians.
    """
    if state == SEAT:
        return config.seat_converge_tol
    if spec is None:
        return config.converge_tol
    if is_side_grasp(spec):
        return config.side_converge_tol
    return config.converge_tol * min(1.0, spec.grasp_width_m / objects.REFERENCE_WIDTH_M)


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

MOVING_JAW_DEEPEST_Z: float = 0.00806
"""Deepest the *moving* jaw reaches below the TCP, over its whole sweep, metres.

Measured off the meshes (``tests/test_expert_logic.py`` re-derives it): the
finger swings down as it closes and back up as it goes past, peaking 1.8 mm
below the static fingertips at 0.14 rad. Only reached with the jaws well shut,
which is why it is a CLOSE-time clearance rather than an approach one -- see
:func:`jaw_depth`.
"""

MOVING_JAW_CLEAR_RAD: float = 0.347
"""Jaw angle above which the moving finger is shallower than the static tips, radians.

**Measured**, same sweep as :data:`MOVING_JAW_DEEPEST_Z`. Above it the static
fingertips are the lowest thing on the hand; by :data:`~manus.control.GRIPPER_OPEN`
the moving finger has swung 57 mm *behind* the TCP and is not in the approach
corridor at all.
"""


JAW_CLEARANCE: float = 0.002
"""Gap left between the object and the static jaw on the way down, metres.

Pure descent margin. The static jaw cannot move out of the way, so the object
has to be placed clear of it -- and the descent carries the full positioning
error of the arm, ~2 mm at the convergence tolerance. The moving jaw then
pushes the object back across this gap as it closes, so more clearance is a
safer approach and a bigger shove at contact.

That trade is exactly what the :data:`SEAT` state refuses to make: an object
with :attr:`~manus.objects.ObjectSpec.seat_close` keeps the full clearance for
the whole approach and then gives it back at 1.5 mm/s with the arm, so the
approach is as safe as the clearance is wide and the shove is nothing at all.
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
grip creeps towards the puck's top edge.

It is a *default*, overridable per object through
:attr:`~manus.objects.ObjectSpec.tip_clearance_m` (see :func:`tip_clearance`),
which is how the 3-7 mm band is swept without a code change. Sweeping it does
not rescue the puck -- the closing finger meets its rim tilted and descending
whatever the height, which is why the spec is marked experimental -- but the
band is where any future short object's grasp gets tuned.
"""

JAW_PARALLEL_REACH: float = 0.020
"""How far above the TCP the closing jaw still meets an object at its pads, metres.

**Measured off the meshes** (``tests/test_expert_logic.py`` re-derives it): the
moving finger's inner face is not parallel to the static one, it *leans in* as
it goes up, so the higher an object stands above the TCP the earlier in the
closing sweep the jaw catches it -- and it catches it up there, on one side,
with the pads still clear of the body. Sweeping a 30 mm column of increasing
height past the jaw, the lead over the pads' own contact angle is 13 mrad while
the column's top is within 11 mm of the TCP (the 30 mm cube's case, which
grasps), 17 mrad out to 20 mm, and then steps to 22, 26 and 45 mrad as the
finger's upper lobe comes into play. 20 mm is that last flat shelf.

Only a *tall* object can feel it, and it is the constraint the 60 mm cylinder
was violating: standing 26 mm above its mid-height grasp, it was struck 26 mm
above its own centre of mass, 2.0 mm before the pads reached its 30 mm body,
and toppled (the CLOSE trace in ``runs/object_previews/cylinder_3cm_demo.json``
and its mp4: z 30.0 -> 43.9 mm during CLOSE, peaking at 56.9). Every catalogue
object that grasps stands 16 mm or less above its TCP -- the ball is the
tallest, and its "top" is a tangent point rather than a wall -- so the shelf is
above all of them and :func:`grasp_height` raises the cylinder alone.

See :func:`grasp_height`; this is the CLOSE-time sibling of the approach-time
clearance :func:`pregrasp_height` keeps with :data:`JAW_TIP_Z`.
"""

SIDE_JAW_DEPTH: float = 0.0278
"""How far below the tool axis the hand's lowest material hangs at a level side
grasp, metres. **Measured off the meshes** (``tests/test_expert_logic.py``
re-derives it): the top-down :func:`jaw_depth`'s counterpart, and the thing that
sets how low a side grasp can be taken.

With the tool horizontal and the jaws level, "down" is the tool frame's -Y (at
:data:`SIDE_GRASP_ROLL`, where +Y points world *up*), so what decides the table
clearance is the hand's *half-width*, not its fingertips. And the fingers are
not the widest part of it: across the whole closing sweep the fingers themselves
stay inside +/-11.7 mm of the tool axis (:data:`SIDE_PAD_HALF_REACH`), while the
wrist_roll follower's housing -- 50 to 100 mm back along the approach, i.e.
between the object and the base -- reaches 27.8 mm on one side and 24.2 mm on
the other. (The gripper servo body is the next widest at 20.5 mm and never
decides it.)

This is the **27.8 mm side**, because :data:`SIDE_GRASP_ROLL` is now chosen by
the camera rather than by the table: the roll that puts the shallower 24.2 mm
side down is the roll that buries the wrist camera under the table. The 3.6 mm
that costs is bought back three times over by :func:`grasp_height`, which takes
a side grasp at the cup height rather than at the object's mid-height."""

SIDE_JAW_DEPTH_SHALLOW: float = 0.0242
"""The hand's *other* half-width about the tool axis, metres. **Measured.**

Kept because it is the number the roll choice was originally made on, and
because it is what the discarded branch would have been worth: 3.6 mm more table
clearance at the same grasp height, in exchange for an upside-down wrist camera
25 mm below the table (see :data:`SIDE_GRASP_ROLL`)."""

SIDE_PAD_HALF_REACH: float = 0.0117
"""Half-height of the hand's finger envelope about the tool axis, metres.
**Measured off the meshes**: everything forward of 50 mm behind the TCP stays
inside this of the tool axis over the whole closing sweep.

Turned on its side this is a *vertical* extent, and it is what stops
:func:`grasp_height` taking a side grasp arbitrarily high up a standing object:
the pads have to stay on the body, so the grasp cannot go closer than this to
the top of it."""

SIDE_GRASP_ROLL: float = 0.0
"""Tool roll a side grasp is planned at, radians -- see
:func:`manus.kinematics.tool_roll_of`.

Level, so the jaws close horizontally across the object the way a hand closes on
a cup. Both level rolls (0 and pi) are the same physical grasp with the fingers
swapped, and both are inside wrist_roll's travel at every pan, so the choice is
free of the arm -- but it is **not** free of the camera, which is bolted to the
gripper link 55 mm off the tool axis and rides the roll with everything else:

* at ``0`` the camera stands 55 mm **above** the tool axis (95 mm above the
  table at the cup grasp), looks 33 deg **down** at the object, and its image up
  axis has world-z ``+0.842`` -- upright,
* at ``pi`` the same camera hangs 55 mm **below** the tool axis, which at any
  side grasp height the arm can reach puts it *underneath the table*: at the
  filmed 30 mm grasp it sat 25 mm below the ground plane, looking 33 deg **up**,
  image up axis world-z ``-0.842``. The recorded POV was the underside of the
  ground plane with the object showing through it (measured, and visible in
  ``runs/object_previews/cylinder_3cm_0000.mp4``).

So the roll is spent on the camera. What it costs is table clearance -- pi puts
the hand's shallower side down (:data:`SIDE_JAW_DEPTH_SHALLOW`), 0 puts the
deeper one down, a 3.6 mm difference -- and :func:`grasp_height` pays it by
taking the grasp at the cup height instead of the object's mid-height, which is
worth 10 mm on the catalogue's cylinder. At 0 the tool's +Y points world **up**."""

SIDE_AZIMUTH_PASSES: int = 3
"""Fixed-point passes closing a side plan's approach azimuth (:func:`plan_grasp`).

A side grasp has to stand the tool so the object lands between the pads, which
needs the tool's *rotation* -- and the approach azimuth half of that is not
something :func:`~manus.kinematics.ik_solve` takes as a request. The arm has
five joints and the side grasp spends them all on position, pitch sum and roll;
the azimuth is then whatever the reach direction makes it (see
:meth:`~manus.kinematics.KinematicChain.approach_azimuth`), and it misses the
object's own azimuth by up to ~2.7 deg because the tool stands 17 mm off to one
side.

So the plan solves, reads the azimuth back off the solution, and re-aims. It is
a sharp contraction -- re-aiming by an angle moves the TCP by only the 17.5 mm
pad offset, which at a 0.36-0.42 m radius feeds back about a hundredth of it --
so the naive aim's 0.44 mm miss becomes 4 um after one pass and 0.15 um after
two, which is the URDF's own noise floor. Three is that plus a pass in hand.
"""

YAW_MATCH_TOL: float = math.radians(2.0)
"""How far the solved tool yaw may sit from the requested one, radians.

Wide enough to swallow ``ik_solve``'s own 1 deg yaw tolerance, far narrower
than the pi flip it is there to catch.
"""


def jaw_depth(gripper: float) -> float:
    """How far below the TCP the lowest point of either jaw sits, metres.

    Two-valued rather than a profile, because only the two ends of the sweep
    are load-bearing and both are measured: with the jaws open (the whole
    approach) the static fingertips lead, and anywhere the moving finger might
    lead instead the answer is its deepest point over the sweep, which is
    conservative by at most 3.7 mm. Used by :func:`pregrasp_height` to hold the
    hand above an object rather than through it.

    Args:
        gripper: Jaw joint angle, radians.

    Returns:
        Depth of the lowest jaw material below the TCP, metres.
    """
    return JAW_TIP_Z if gripper >= MOVING_JAW_CLEAR_RAD else MOVING_JAW_DEEPEST_Z


def pad_lateral_offset(spec: ObjectSpec, clearance: float | None = None) -> float:
    """Where the object's centre should sit along TCP-frame x, metres.

    Negative: the object is offset *away* from the static jaw, by its own half
    width plus `clearance`, so that the descent passes the static finger and the
    closing moving finger seats the object against it.

    Args:
        spec: Object being grasped; supplies the grasp width.
        clearance: Gap to leave to the static pad, metres. None -- the default
            and every approach's answer -- reads :data:`JAW_CLEARANCE` at call
            time, which is what makes ``--jaw-clearance`` an override rather
            than a no-op. The SEAT state passes
            :attr:`~ExpertConfig.seat_gap` instead: same object, same yaw, a
            tool standing 2 mm closer (:func:`seat_stroke`).
    """
    gap = JAW_CLEARANCE if clearance is None else clearance
    return JAW_FIXED_FACE_X - 0.5 * spec.grasp_width_m - gap


def tip_clearance(spec: ObjectSpec) -> float:
    """Gap to leave between the fingertips and the table for `spec`, metres.

    :data:`MIN_TIP_CLEARANCE` unless the spec overrides it
    (:attr:`~manus.objects.ObjectSpec.tip_clearance_m`), which only an object
    short enough for the clearance to bind can feel at all.
    """
    return MIN_TIP_CLEARANCE if spec.tip_clearance_m is None else spec.tip_clearance_m


def is_side_grasp(spec: ObjectSpec) -> bool:
    """Whether `spec` is grasped from the side rather than from above."""
    return spec.grasp_mode == "side"


def seats(spec: ObjectSpec) -> bool:
    """Whether `spec` closes the last :data:`JAW_CLEARANCE` with the arm first.

    :attr:`~manus.objects.ObjectSpec.seat_close`, i.e. whether the FSM walks a
    :data:`SEAT` between the approach and CLOSE. Off for every object the tuned
    closure already grasps -- their gate anchors are pinned and a state they do
    not walk cannot move them.
    """
    return spec.seat_close


def state_sequence(spec: ObjectSpec) -> tuple[str, ...]:
    """The FSM's state order for `spec`: the mode's, plus a SEAT if it asks.

    :data:`STATE_SEQUENCE` or :data:`SIDE_STATE_SEQUENCE` by
    :func:`is_side_grasp`, with :data:`SEAT` spliced in before CLOSE when
    :func:`seats` -- so the two shove-failing objects walk seven states and
    every other object walks the six it always did.
    """
    base = SIDE_STATE_SEQUENCE if is_side_grasp(spec) else STATE_SEQUENCE
    if not seats(spec):
        return base
    index = base.index(CLOSE)
    return base[:index] + (SEAT,) + base[index:]


def approach_state(spec: ObjectSpec) -> str:
    """The state that moves onto the grasp pose: ``DESCEND`` or ``ADVANCE``."""
    return ADVANCE if is_side_grasp(spec) else DESCEND


def side_body_behind_tcp(spec: ObjectSpec) -> float:
    """How far a side-grasped object reaches back past the TCP, metres.

    The side grasp's version of "how far the object stands above the TCP" --
    the quantity :data:`JAW_PARALLEL_REACH` bounds. Rotating the hand onto its
    side rotates that question with it: what used to be the object's height
    above the pads is now its own *radius* behind them, since the pads sit
    :data:`~manus.kinematics.TCP_TO_PAD_CENTRE` along the approach and the
    object's near face is a half-width short of its centre.

    For the 30 mm cylinder that is 11 mm against the 20 mm shelf -- and unlike
    the top-down bar it is a fixed property of the object, not something
    :func:`grasp_height` can trade height for, because moving the tool along the
    approach axis moves the pads off the object's centre line. An object round
    enough to be side-grasped at all is therefore either inside the shelf or not
    side-graspable; the catalogue's one is inside it by 9 mm.
    """
    return 0.5 * spec.grasp_width_m - TCP_TO_PAD_CENTRE


SIDE_GRASP_HEIGHT_FRACTION: float = 2.0 / 3.0
"""Fraction of a standing object's height a side grasp aims for.

The cup grasp, and it is above the middle for three reasons that all point the
same way -- the same reasons a hand takes a cup of water above its waist:

* **the table.** The hand hangs :data:`SIDE_JAW_DEPTH` below the tool axis, so
  every millimetre of grasp height is a millimetre of table clearance, and the
  clearance is what the whole side approach is spent out of (:data:`CONVERGE_TOL`
  and the droop the arm carries into ADVANCE both cash out here).
* **the lift.** Gripped above its centre of mass the object hangs from the pads
  and the retraction's ~30 deg of tilt swings it; gripped below, the same tilt
  levers it out.
* **the closing jaw.** A horizontal push lands a topple moment about the
  object's base contact; taken high the object is already held on both sides
  before that matters, and the pads' own band (:data:`SIDE_PAD_HALF_REACH`) is
  what has to stay on the body.

Two thirds is the highest simple fraction that keeps the whole pad band on a
60 mm cylinder with room to spare, and it lands the catalogue's side grasp at
40 mm: 12.2 mm of table clearance where the old mid-height grasp had 5.8 mm and
lost the hand into the table on 7.4 mm of droop."""


def side_table_clearance(spec: ObjectSpec) -> float:
    """Gap between the hand's lowest material and the table at `spec`'s side grasp.

    ``grasp_height - SIDE_JAW_DEPTH``, metres: the budget every vertical error in
    a side approach is spent out of -- the droop the arm has not yet cancelled,
    and the convergence bar that decides how much of it is allowed to survive
    into CLOSE (:data:`SIDE_CONVERGE_TOL`).
    """
    return grasp_height(spec) - SIDE_JAW_DEPTH


def grasp_height(spec: ObjectSpec) -> float:
    """Height above the table the jaw pads centre on for `spec`, metres.

    **Side grasps** are taken at the *cup height*,
    :data:`SIDE_GRASP_HEIGHT_FRACTION` of the way up the standing object -- the
    pads centre on the tool axis, which is horizontal, so there is no
    :data:`~manus.kinematics.TCP_TO_PAD_CENTRE` in the vertical any more and the
    TCP simply sits at this height. Two bars bracket it:

    * **the table**, which raises: the hand's own housing hangs
      :data:`SIDE_JAW_DEPTH` below the tool axis and has to clear the table by
      :func:`tip_clearance`, so nothing can be side-grasped below 32.8 mm.
    * **the object's top**, which lowers: the fingers reach
      :data:`SIDE_PAD_HALF_REACH` above the tool axis, so a grasp closer than
      that to the top of the object hangs half the pad band off it.

    The 60 mm cylinder's cup height is 40 mm, which clears the first by 7.2 mm
    and the second by 8.3 mm. Its old mid-height grasp cleared the table by
    0.8 mm of *plan* and by nothing at all in the sim: 7.4 mm of un-cancelled
    droop at ADVANCE put the housing on the table with the TCP 22.6 mm up
    instead of 30, and the fingertip then swept the cylinder's base inward
    through CLOSE (``runs/object_previews/cylinder_3cm_0000.mp4``, and the
    ADVANCE report in its ``_demo.json``).

    **Top-down grasps** take the object at its own mid-height, which puts the
    pads across the widest part of it -- raised, whichever of two bars binds,
    and both of them are raises:

    * **too short**: until the fingertips clear the table by
      :func:`tip_clearance`. The 10 mm puck is the only object short enough to
      feel the 5 mm default, and it is raised 2.3 mm -- but the clearance is
      per-object, so this is also the bar an object can be *given* to move its
      grasp deliberately: the 20 mm puck asks for 11.76 mm of it, which raises
      it 4.1 mm off its own centre and out of the band where the closing
      finger's deepest sweep reaches under its centre of mass.
    * **too tall**: until the object's top is no further above the TCP than
      :data:`JAW_PARALLEL_REACH`, so the closing jaw meets the object at its
      *pads* rather than leaning into its upper body first. Only the 60 mm
      cylinder is tall enough to feel it, and it is raised 6.0 mm (30 -> 36 mm,
      which drops the jaw's lead over the pads from 2.0 mm to 0.7 mm -- below
      the cube's own 1.0 mm -- and its contact from 26 mm above the cylinder's
      centre of mass to 18 mm).

    Neither bar can be met by lowering the grasp, so the two never fight: the
    answer is the highest of the three.
    """
    if is_side_grasp(spec):
        floor = SIDE_JAW_DEPTH + tip_clearance(spec)
        ceiling = spec.top_z - SIDE_PAD_HALF_REACH
        return max(floor, min(SIDE_GRASP_HEIGHT_FRACTION * spec.top_z, ceiling))
    lowest_centre = JAW_TIP_Z - TCP_TO_PAD_CENTRE + tip_clearance(spec)
    closing_centre = spec.top_z - TCP_TO_PAD_CENTRE - JAW_PARALLEL_REACH
    return max(spec.spawn_z, lowest_centre, closing_centre)


def pregrasp_height(spec: ObjectSpec, config: ExpertConfig | None = None) -> float:
    """TCP height PREGRASP waits at, and the approach starts from, metres.

    **A side grasp's PREGRASP is at the grasp's own height** -- it stands off
    along the table rather than above it, by :attr:`~ExpertConfig.side_retract`,
    so ADVANCE is a pure radial push and nothing about the approach is vertical.
    There is no hover bar to compute: the object is *beside* the hand at the
    stand-off, not under it.

    For a top-down grasp, the higher of two bars:

    * :attr:`~ExpertConfig.hover_height` above the grasp pose, the fixed
      stand-off the descent was tuned as, and
    * enough for the lowest jaw material to clear the *top of the object* by
      :attr:`~ExpertConfig.hover_margin`.

    The second bar exists because the first one is measured from the grasp
    pose, which sits at the object's mid-height: on anything taller than
    ``2 * (hover_height - jaw_depth - margin)`` the "hover" is inside the
    object. The 60 mm cylinder is the catalogue's case -- the fixed rule put the
    fingertips 2.3 mm *below* its top, and the tool swept it over on the way
    into the waypoint -- and the 40 mm ball misses by 0.3 mm. Both are the same
    arithmetic, so the fix is the arithmetic rather than a taller constant:
    a taller fixed hover would raise every grasp to suit the worst object and
    cost reach at the top of the region.

    Args:
        spec: Object being grasped.
        config: Tunables; :attr:`~ExpertConfig.hover_height`,
            :attr:`~ExpertConfig.hover_margin` and
            :attr:`~ExpertConfig.gripper_open` are read.

    Returns:
        TCP height above the table, metres.
    """
    config = DEFAULT_CONFIG if config is None else config
    if is_side_grasp(spec):
        return grasp_height(spec)
    stand_off = grasp_height(spec) + TCP_TO_PAD_CENTRE + config.hover_height
    clearing = spec.top_z + jaw_depth(config.gripper_open) + config.hover_margin
    return max(stand_off, clearing)


def tcp_target(
    object_xy: tuple[float, float],
    object_z: float,
    grasp_yaw: float,
    spec: ObjectSpec,
    clearance: float | None = None,
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

    `clearance` is the gap to the static pad the tool is stood off by; None is
    :data:`JAW_CLEARANCE`, the approach's, and the SEAT waypoint asks for
    :attr:`~ExpertConfig.seat_gap` instead (:func:`pad_lateral_offset`).
    """
    lateral = pad_lateral_offset(spec, clearance)
    return np.array(
        [
            object_xy[0] - lateral * math.cos(grasp_yaw),
            object_xy[1] - lateral * math.sin(grasp_yaw),
            object_z + TCP_TO_PAD_CENTRE,
        ]
    )


def side_tcp_target(
    object_xy: tuple[float, float],
    object_z: float,
    approach_azimuth: float,
    spec: ObjectSpec,
    tool_roll: float = SIDE_GRASP_ROLL,
    clearance: float | None = None,
) -> np.ndarray:
    """TCP position (3,) placing the jaws around `spec` for a **side** grasp.

    The same inversion :func:`tcp_target` does, written with the rotation matrix
    instead of by hand because the side grasp's tool frame is not axis-aligned::

        tcp = object - R(approach_azimuth, tool_roll) @ (lateral, 0, pad)

    -- and the same object placement in the tool frame, ``(lateral, 0,
    TCP_TO_PAD_CENTRE)``, because *the hand does not know it has been rotated*.
    What changes is where those two offsets point in the world: the lateral one
    is now **tangential** (it swings the tool sideways around the object rather
    than across it) and the pad one is now **radial** (the TCP stands 4 mm short
    of the object's centre along the approach instead of 4 mm above it). The
    grasp height, which was the tool's height plus the pad offset, is now the
    tool's height exactly.

    Args:
        object_xy: Object centre (x, y) in base_link coordinates, metres.
        object_z: Height the pads should centre on, metres
            (:func:`grasp_height`).
        approach_azimuth: World azimuth the hand comes in along, radians. Not a
            free request -- see :data:`SIDE_AZIMUTH_PASSES`.
        spec: Object being grasped; supplies the grasp width.
        tool_roll: Jaw tilt off level, radians; the default is the level branch
            the whole side pipeline is planned at.
        clearance: Gap to the static pad, metres; None is :data:`JAW_CLEARANCE`.
            **This is the direction a side grasp seats along**, and it is not
            the approach direction: the lateral offset is tangential here (see
            above), so SEAT slides the flat hand sideways across the object's
            face rather than pushing further in along the approach. Pushing
            along the approach would drive the pads off the object's centre
            line instead of closing the gap to the static pad.

    Returns:
        The (3,) TCP position, metres.
    """
    rotation = kinematics.horizontal_tool_rotation(approach_azimuth, tool_roll)
    offset = np.array([pad_lateral_offset(spec, clearance), 0.0, TCP_TO_PAD_CENTRE])
    return np.array([*object_xy, object_z]) - rotation @ offset


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
        grasp_yaw: Tool angle the grasp is planned at: a tool *yaw* for a
            top-down grasp, a tool *roll* off level for a side one -- whichever
            the mode's :class:`~manus.kinematics.ToolFamily` measures, and what
            was handed to :func:`~manus.kinematics.ik_solve`.
        q_pregrasp: Arm pose standing off from the object.
        q_grasp: Arm pose at the grasp.
        q_lift: Arm pose after the retraction.
        tcp_pregrasp: TCP position the pregrasp pose was solved for.
        tcp_grasp: TCP position the grasp pose was solved for.
        lateral_offset: TCP-frame x the object centre was aimed at, metres
            (:func:`pad_lateral_offset`).
        q_seat: Arm pose with the static pad against the object, or None for an
            object that does not seat (:func:`seats`). The same grasp with
            :attr:`~ExpertConfig.seat_gap` in place of :data:`JAW_CLEARANCE`,
            which moves the tool :func:`seat_stroke` along its own -x.
        tcp_seat: TCP position :attr:`q_seat` was solved for, or None.
        seat_offset: TCP-frame x the object centre is aimed at *at the seat*,
            metres, or None -- the seat's counterpart of
            :attr:`lateral_offset`, and the difference between the two is the
            stroke.
        lift_rise: TCP height gain from :attr:`q_grasp` to :attr:`q_lift`.
        close_target: Jaw target used by CLOSE.
        ik_converged: Whether both IK solves met their tolerances.
        reason: ``""`` when :attr:`ok`, else why the plan is not trustworthy.
        grasp_mode: ``"top"`` or ``"side"`` -- which of the two plans this is.
        approach_azimuth: World azimuth the hand comes in along for a side
            grasp, radians, as the solved pose actually delivers it (see
            :data:`SIDE_AZIMUTH_PASSES`); None for a top-down grasp, whose
            approach is straight down.
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
    grasp_mode: str = "top"
    approach_azimuth: float | None = None
    q_seat: np.ndarray | None = None
    tcp_seat: np.ndarray | None = None
    seat_offset: float | None = None

    @property
    def ok(self) -> bool:
        """Whether the plan is feasible: IK converged and the lift clears."""
        return self.reason == ""

    def waypoint(self, state: str) -> np.ndarray:
        """Arm waypoint of an arm state (PREGRASP, DESCEND/ADVANCE, SEAT, LIFT).

        Raises:
            KeyError: `state` does not move the arm.
            ValueError: SEAT was asked for on a plan that has no seat pose,
                which means the FSM and the plan were built from different
                specs -- worth failing loudly rather than seating on the
                approach pose and calling the gap closed.
        """
        if state == SEAT:
            if self.q_seat is None:
                raise ValueError("plan has no seat pose; spec.seat_close is not set")
            return self.q_seat
        return {
            PREGRASP: self.q_pregrasp,
            DESCEND: self.q_grasp,
            ADVANCE: self.q_grasp,
            LIFT: self.q_lift,
        }[state]


def _plan_side_grasp(
    spec: ObjectSpec,
    x: float,
    y: float,
    config: ExpertConfig,
) -> GraspPlan:
    """Solve the three waypoints for a **side** grasp of `spec` at ``(x, y)``.

    No yaw branches: the object is round (the catalogue refuses a side grasp on
    anything else) and the approach azimuth is not a choice, so the only branch
    a top-down plan would search over -- which way to face -- does not exist.
    What replaces it is the azimuth fixed point (:data:`SIDE_AZIMUTH_PASSES`):
    stand the tool where the object's *own* azimuth says, solve, and re-aim at
    the azimuth the arm actually delivered, until the object really is between
    the pads.

    PREGRASP then backs the solved grasp pose straight out along its own
    approach axis by :attr:`~ExpertConfig.side_retract`, so ADVANCE is a pure
    radial push and the hand never passes over the object.
    """
    region = kinematics.SIDE_GRASP_REGION
    object_z = grasp_height(spec)
    azimuth = region.polar(x, y)[1]
    tcp_grasp = side_tcp_target((x, y), object_z, azimuth, spec)
    q_grasp, grasp_ok = ik_solve(
        tcp_grasp, SIDE_GRASP_ROLL, family=kinematics.TOOL_HORIZONTAL
    )
    for _ in range(SIDE_AZIMUTH_PASSES):
        azimuth = _CHAIN.approach_azimuth(q_grasp)
        tcp_grasp = side_tcp_target((x, y), object_z, azimuth, spec)
        q_grasp, grasp_ok = ik_solve(
            tcp_grasp, SIDE_GRASP_ROLL, family=kinematics.TOOL_HORIZONTAL
        )

    approach = _CHAIN.fk_tcp(q_grasp)[1][:, 2]
    tcp_pregrasp = tcp_grasp - config.side_retract * approach
    q_pregrasp, pregrasp_ok = ik_solve(
        tcp_pregrasp, SIDE_GRASP_ROLL, family=kinematics.TOOL_HORIZONTAL
    )
    q_lift, lift_rise = plan_lift(q_grasp, config.lift_rise)

    # The seat is the same grasp with the tool stood 2 mm further along its own
    # -x, which for a side grasp is *tangential* -- solved rather than nudged so
    # the azimuth fixed point above still holds at the seated pose.
    q_seat: np.ndarray | None = None
    tcp_seat: np.ndarray | None = None
    seat_ok = True
    if seats(spec):
        tcp_seat = side_tcp_target(
            (x, y), object_z, azimuth, spec, clearance=config.seat_gap
        )
        q_seat, seat_ok = ik_solve(
            tcp_seat, SIDE_GRASP_ROLL, family=kinematics.TOOL_HORIZONTAL
        )

    rolled = max(
        abs(_wrap(_CHAIN.tool_roll(q_pregrasp) - SIDE_GRASP_ROLL)),
        abs(_wrap(_CHAIN.tool_roll(q_grasp) - SIDE_GRASP_ROLL)),
    ) > YAW_MATCH_TOL
    reason = ""
    if not (pregrasp_ok and grasp_ok):
        reason = f"ik_{'pregrasp' if not pregrasp_ok else 'grasp'}_unreachable"
    elif not seat_ok:
        reason = "ik_seat_unreachable"
    elif rolled:
        reason = "ik_solved_the_flipped_roll"
    elif lift_rise < config.min_lift_rise:
        reason = f"lift_rise_{lift_rise * 1e3:.0f}mm_below_minimum"
    return GraspPlan(
        grasp_yaw=SIDE_GRASP_ROLL,
        q_pregrasp=q_pregrasp,
        q_grasp=q_grasp,
        q_lift=q_lift,
        tcp_pregrasp=tcp_pregrasp,
        tcp_grasp=tcp_grasp,
        lateral_offset=pad_lateral_offset(spec),
        lift_rise=lift_rise,
        close_target=spec.close_target_rad,
        ik_converged=bool(pregrasp_ok and grasp_ok and seat_ok),
        reason=reason,
        grasp_mode="side",
        approach_azimuth=float(_CHAIN.approach_azimuth(q_grasp)),
        q_seat=q_seat,
        tcp_seat=tcp_seat,
        seat_offset=pad_lateral_offset(spec, config.seat_gap) if seats(spec) else None,
    )


def plan_grasp(
    spec: ObjectSpec,
    placement: EpisodeDraw | Sequence[float],
    q_current: np.ndarray | None = None,
    config: ExpertConfig = DEFAULT_CONFIG,
) -> GraspPlan:
    """Solve the three waypoints for grasping `spec` at `placement`.

    A side-mode object (:func:`is_side_grasp`) goes to :func:`_plan_side_grasp`,
    which is a different tool family over a different region and has no yaw
    branches to search; everything below is the top-down plan.

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
        config: Tunables; :func:`pregrasp_height`'s inputs,
            :attr:`~ExpertConfig.lift_rise` and
            :attr:`~ExpertConfig.min_lift_rise` are read.

    Returns:
        The :class:`GraspPlan`. An infeasible request still comes back as a
        plan -- with ``ok`` False and ``reason`` set, and waypoints holding
        ``ik_solve``'s best in-limit effort -- so a caller can run the attempt
        anyway and report an honest failure rather than a skipped sample.
    """
    x, y, object_yaw = _placement(placement)
    if is_side_grasp(spec):
        return _plan_side_grasp(spec, x, y, config)
    start = (
        np.zeros(kinematics.NUM_ARM_JOINTS)
        if q_current is None
        else np.asarray(q_current, dtype=float).reshape(-1)[: kinematics.NUM_ARM_JOINTS]
    )
    best: GraspPlan | None = None
    for grasp_yaw in grasp_yaw_candidates(spec, object_yaw, _CHAIN.tool_yaw(start)):
        tcp_grasp = tcp_target((x, y), grasp_height(spec), grasp_yaw, spec)
        tcp_pregrasp = np.array([tcp_grasp[0], tcp_grasp[1], pregrasp_height(spec, config)])
        q_pregrasp, pregrasp_ok = ik_solve(tcp_pregrasp, grasp_yaw)
        q_grasp, grasp_ok = ik_solve(tcp_grasp, grasp_yaw)
        q_lift, lift_rise = plan_lift(q_grasp, config.lift_rise)
        # Same grasp, tool stood 2 mm along its own -x: a purely horizontal
        # slide at the grasp height for a top-down grasp, so it costs the
        # fingertips none of their table clearance.
        q_seat: np.ndarray | None = None
        tcp_seat: np.ndarray | None = None
        seat_ok = True
        if seats(spec):
            tcp_seat = tcp_target(
                (x, y), grasp_height(spec), grasp_yaw, spec, clearance=config.seat_gap
            )
            q_seat, seat_ok = ik_solve(tcp_seat, grasp_yaw)
        flipped = max(
            abs(_wrap(_CHAIN.tool_yaw(q_pregrasp) - grasp_yaw)),
            abs(_wrap(_CHAIN.tool_yaw(q_grasp) - grasp_yaw)),
        ) > YAW_MATCH_TOL
        reason = ""
        if not (pregrasp_ok and grasp_ok):
            missed = "pregrasp" if not pregrasp_ok else "grasp"
            reason = f"ik_{missed}_unreachable"
        elif not seat_ok:
            reason = "ik_seat_unreachable"
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
            ik_converged=bool(pregrasp_ok and grasp_ok and seat_ok),
            reason=reason,
            q_seat=q_seat,
            tcp_seat=tcp_seat,
            seat_offset=pad_lateral_offset(spec, config.seat_gap) if seats(spec) else None,
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
        self._sequence: tuple[str, ...] = state_sequence(spec)
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
        self._seat_reference: np.ndarray | None = None
        self._seat_contact_steps = 0
        self._seat_excess = 0.0
        self._seat_exit: tuple[str, float] | None = None
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
        self._seat_reference = None
        self._seat_contact_steps = 0
        self._seat_excess = 0.0
        self._seat_exit = None
        self._reports = []
        self._timeouts = []
        self._started = True
        assert self._plan is not None
        return self._plan

    # -- read-only surface -----------------------------------------------------

    @property
    def state(self) -> str:
        """Current state, one of :attr:`sequence`."""
        return self._state

    @property
    def sequence(self) -> tuple[str, ...]:
        """The state order this expert walks: :func:`state_sequence` of its spec."""
        return self._sequence

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
        self._seat_reference = None
        self._seat_contact_steps = 0
        self._seat_excess = 0.0

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
        if self._state == SEAT:
            self._seat_exit = (exit_reason, self._seat_excess)
        if self._state in FREEZE_STATES:
            # CLOSE holds the arm exactly where the approach (or the seat) left
            # it -- including the droop bias -- so the jaws close on the pose
            # that was reached, not on a recomputed one that would nudge the arm
            # mid-grasp. On a seating object the latch happens twice and the
            # second one wins, which is the point: what CLOSE must hold is the
            # *seated* pose, pad against the object.
            self._frozen_arm = self._last_arm_command.copy()
        self._enter(self._sequence[self._sequence.index(self._state) + 1], arm, gripper)

    def _waypoint(self) -> np.ndarray | None:
        """Arm waypoint of the current state, or None if it does not move the arm."""
        if self._state in ARM_STATES:
            return self.plan.waypoint(self._state)
        return None

    def _ramp(self) -> float:
        """Fraction of the current state's ramp already commanded, in [0, 1]."""
        span = max(1, self.config.ramp_steps(self._state, self.spec))
        return min(1.0, self._state_step / span)

    def _may_exit(self) -> bool:
        """Whether the current state has finished commanding *and* settling.

        The dwell is zero everywhere except a side grasp's approach states and
        SEAT, and the commanded length is the state's ramp everywhere except a
        creeping CLOSE (:func:`close_steps`) -- so for a top-down catalogue
        object this is exactly ``self._ramp() >= 1.0``, the condition the FSM
        has always used. CLOSE has to wait out its whole creep before a stall may end it,
        or the squeeze the object is held with would be whatever the creep had
        reached rather than the commanded target.
        """
        if self._state == CLOSE:
            return self._state_step >= close_steps(self.spec, self.config)
        span = max(1, self.config.ramp_steps(self._state, self.spec))
        return self._state_step >= span + settle_steps(self._state, self.spec, self.config)

    def _exit_reason(self, arm: np.ndarray) -> str | None:
        """Why the current state should end now, or None to stay in it."""
        if self._state == DONE:
            return None
        budget_spent = self._state_step >= state_budget(self._state, self.spec, self.config)
        if self._state == HOLD:
            return "elapsed" if self._state_step >= self.config.hold_steps else None
        if self._state == SEAT and self._seat_contact(arm):
            # The pad has found the object: stop pushing *now*, mid-ramp if need
            # be. Everything after this is the moving jaw's job and it has a
            # backstop to work against, which is the whole point of the state.
            return "seated"
        if self._state in ARM_STATES:
            waypoint = self.plan.waypoint(self._state)
            converged = float(np.abs(arm - waypoint).max()) < converge_tol(
                self.spec, self.config, state=self._state
            )
            if self._may_exit() and converged:
                return "converged"
            return "timeout" if budget_spent else None
        # CLOSE: the jaws stop moving once they are squeezing the object (or
        # each other) -- a position proxy for effort saturation, since the
        # implicit actuator's applied torque is not part of the contract here.
        if self._may_exit() and self._gripper_stalled():
            return "stalled"
        return "timeout" if budget_spent else None

    def _seat_contact_bar(self) -> float:
        """Excess tracking error this seat calls contact, radians.

        :attr:`~ExpertConfig.seat_contact_tol` plus the lag a free arm is
        entitled to at this seat's own commanded joint rate -- see
        :data:`SEAT_CONTACT_LAG_STEPS`, which is where the per-object part of
        the bar comes from.
        """
        assert self._entry_arm is not None
        span = max(1, self.config.ramp_steps(SEAT, self.spec))
        rate = float(np.abs(self.plan.waypoint(SEAT) - self._entry_arm).max()) / span
        return self.config.seat_contact_tol + SEAT_CONTACT_LAG_STEPS * rate

    def _seat_contact(self, arm: np.ndarray) -> bool:
        """Whether the seating pad has run into the object, from tracking error alone.

        The arm has no force sensor, so contact is read the same way CLOSE reads
        it at the jaw: the servo is commanded somewhere it cannot go, and the
        error stands. What makes it readable at 0.68 mm is that the *baseline*
        is removed -- a drooping arm always tracks a few milliradians behind its
        command, so what is watched is the growth since the seat started moving
        (:data:`SEAT_CONTACT_REF_STEP`), not the error itself.

        Deliberately conservative in both directions:

        * the bar it is compared against carries an explicit allowance for the
          lag the commanded creep itself produces, scaled by this seat's own
          joint rate (:meth:`_seat_contact_bar`);
        * it wants :data:`SEAT_CONTACT_STEPS` consecutive samples, so one noisy
          step cannot end the seat two millimetres early.

        Args:
            arm: The (5,) measured arm pose this step, radians.

        Returns:
            Whether contact has been confirmed.
        """
        tracking = self._last_arm_command - arm
        if self._seat_reference is None:
            if self._state_step >= SEAT_CONTACT_REF_STEP:
                self._seat_reference = tracking.copy()
            return False
        self._seat_excess = float(np.abs(tracking - self._seat_reference).max())
        if self._seat_excess > self._seat_contact_bar():
            self._seat_contact_steps += 1
        else:
            self._seat_contact_steps = 0
        return self._seat_contact_steps >= SEAT_CONTACT_STEPS

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
        alpha = min(
            1.0, (self._state_step + 1) / max(1, self.config.ramp_steps(self._state, self.spec))
        )
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
        elif self._state in JAWS_OPEN_STATES:
            gripper_command = config.gripper_open
        elif self._state == CLOSE:
            gripper_command = close_command(
                self._entry_gripper,
                self.plan.close_target,
                self._state_step + 1,
                self.spec,
                config,
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
            "grasp_mode": plan.grasp_mode,
            "grasp_yaw": plan.grasp_yaw,
            "approach_azimuth": plan.approach_azimuth,
            "ik_converged": plan.ik_converged,
            "plan_ok": plan.ok,
            "plan_reason": plan.reason,
            "lift_rise": plan.lift_rise,
            "close_target": plan.close_target,
            "approach_ramp": self.config.ramp_steps(approach_state(self.spec), self.spec),
            "approach_settle": settle_steps(approach_state(self.spec), self.spec, self.config),
            "close_ramp": self.config.ramp_steps(CLOSE, self.spec),
            "close_steps": close_steps(self.spec, self.config),
            "converge_tol": converge_tol(self.spec, self.config),
            "grasp_height": grasp_height(self.spec),
            "seats": seats(self.spec),
            "seat_gap": self.config.seat_gap if seats(self.spec) else None,
            "seat_stroke": seat_stroke(self.config) if seats(self.spec) else None,
            "seat_ramp": seat_ramp_steps(self.config) if seats(self.spec) else None,
            "seat_exit": None if self._seat_exit is None else self._seat_exit[0],
            "seat_excess": None if self._seat_exit is None else self._seat_exit[1],
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

IN_HAND_RADIUS_M: float = 0.060
"""How far the object's centre may sit from the TCP and still be *in* the hand, metres.

A seated object sits at the pad centre, which is
``hypot(pad_lateral_offset, TCP_TO_PAD_CENTRE)`` from the TCP: 17.5 mm for the
cube, 22.4 mm for the widest catalogue grasp. This bar is more than twice that,
so it costs a real grasp nothing even with the arm drooping and the tool tilted
through the lift -- what it rejects is an object that is *riding* the hand
rather than being held by it, which starts at the wrist (the jaws are ~50 mm
long) and gets further away from there.
"""

STALL_SLACK_RAD: float = 0.10
"""How far a measured CLOSE stall may sit from the object's contact angle, radians.

The discriminating number, and it works because it is smaller than
:data:`~manus.objects.SQUEEZE_RAD` (0.139): jaws that meet the object stop
*at* its contact angle, jaws that meet nothing run all the way to the commanded
target a full squeeze below it, and 0.10 rad separates the two. Measured against
the seven filmed catalogue grasps, a held object stalls between 94 mrad below
its contact angle (the ping-pong ball, whose pads sink into a tangent point) and
2 mrad above it (cube, domino, duplo, die), while the two empty closures land
140 and 364 mrad below. The ball is the object with the least margin -- 6 mrad
-- so this is the constant to loosen first if a real grasp is ever called a
failure, and the constructor takes it per attempt for exactly that reason.
"""

STALL_TARGET_MARGIN_RAD: float = 0.02
"""How far above its commanded target the jaws must stall to count as loaded, radians.

The second half of the stall clause, and the exact one: the CLOSE target is
commanded *past* contact, so jaws that actually arrive at it were stopped by
nothing. 0.02 rad is 1.5 mm of jaw gap -- enough that an object seated a
millimetre thinner than declared still reads as held. It matters because
:data:`STALL_SLACK_RAD` alone stops discriminating on an object squeezed by
less than the slack (the die, at :data:`~manus.objects.LIGHT_SQUEEZE_RAD`,
would otherwise accept the empty closure that lands exactly on its target).
"""


class GraspSuccessMonitor:
    """The success predicate: object lifted 5 cm, held 30 steps, *in the hand*.

    Four clauses, all of which have to hold on the same step for it to count
    towards the sustain, because the interesting failure is the one that
    satisfies some of them:

    1. the object is :attr:`lift` above where it rested,
    2. the jaws are closed (below :attr:`gripper_max`),
    3. the object's centre is within :attr:`in_hand_radius` of the TCP, by FK
       from the measured joints, and
    4. the jaws stalled where an object of this width would stop them:
       within :attr:`stall_slack` below its contact angle *and* clear of the
       commanded target by :data:`STALL_TARGET_MARGIN_RAD`, and no further above
       contact than the squeeze plus that slack.

    Clauses 3 and 4 are the Step 21 addition, and they exist because height
    alone is not evidence of a grasp: an object knocked onto the forearm, or
    wedged against the outside of a jaw, *rises with the robot*. Two of the
    seven filmed catalogue previews passed the height-only predicate with
    provably empty jaws -- the puck (stalled at 0.187 rad against a 0.327
    contact angle, i.e. exactly on the commanded target) and the cylinder (rode
    the jaw down to the -0.1745 rad hard stop). Clause 4 rejects both from the
    gripper trace the monitor already sees; clause 3 rejects the geometry.

    Stateful because "sustained" means *consecutive* steps: any step that fails
    any clause restarts the count. Once :attr:`success` latches True it stays
    True -- a grasp that succeeded and was then thrown away by a later state
    still succeeded at the moment the predicate was met, and the episode is cut
    there by the driver anyway.

    :attr:`height_only` latches the *old* predicate alongside the new one, so a
    run can report how many of its successes were the height bar alone.

    Args:
        spec: Object being grasped; supplies the rest height, the contact angle
            and the squeeze.
        lift: Rise required above the object's rest height, metres.
        sustain: Consecutive steps required.
        gripper_max: Jaw angle at or below which the gripper counts as closed.
        in_hand_radius: TCP-to-object distance allowed, metres.
        stall_slack: Tolerance on the stall angle, radians.
        target_margin: Clearance the stall must keep above the commanded close
            target, radians.
    """

    def __init__(
        self,
        spec: ObjectSpec,
        *,
        lift: float = SUCCESS_LIFT_M,
        sustain: int = SUCCESS_SUSTAIN_STEPS,
        gripper_max: float = GRIPPER_HELD_MAX_RAD,
        in_hand_radius: float = IN_HAND_RADIUS_M,
        stall_slack: float = STALL_SLACK_RAD,
        target_margin: float = STALL_TARGET_MARGIN_RAD,
    ) -> None:
        self.spec = spec
        self.spawn_z = float(spec.spawn_z)
        self.lift = float(lift)
        self.threshold_z = self.spawn_z + self.lift
        self.sustain = int(sustain)
        self.gripper_max = float(gripper_max)
        self.in_hand_radius = float(in_hand_radius)
        self.stall_slack = float(stall_slack)
        self.target_margin = float(target_margin)
        self.stall_band = (
            max(
                spec.contact_angle_rad - self.stall_slack,
                spec.close_target_rad + self.target_margin,
            ),
            spec.contact_angle_rad + spec.squeeze_rad + self.stall_slack,
        )
        self.streak = 0
        self.best_streak = 0
        self.peak_z = -np.inf
        self.success = False
        self.height_streak = 0
        self.height_only = False
        self.tcp_distance: float | None = None
        self.gripper: float | None = None

    def update(
        self,
        object_pos: Sequence[float] | np.ndarray,
        joint_pos: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> bool:
        """Fold in one step's privileged state; returns :attr:`success`.

        Args:
            object_pos: Object centre ``(x, y, z)`` in the robot's own frame
                (i.e. with the environment origin already subtracted), metres.
            joint_pos: All six measured joint positions, in the form
                :func:`joint_vector` accepts, radians. The TCP comes from these
                by FK rather than from the commanded pose, so a drooping arm is
                measured where it actually is.

        Returns:
            :attr:`success`.
        """
        position = np.asarray(object_pos, dtype=float).reshape(-1)
        if position.shape != (3,):
            raise ValueError(
                f"object_pos must be the object centre (x, y, z), got shape {position.shape}"
            )
        measured = joint_vector(joint_pos)
        gripper = float(measured[GRIPPER_INDEX])
        tcp = _CHAIN.fk_tcp(measured[: kinematics.NUM_ARM_JOINTS])[0]

        height = float(position[2])
        self.peak_z = max(self.peak_z, height)
        self.tcp_distance = float(np.linalg.norm(position - tcp))
        self.gripper = gripper

        lifted = height >= self.threshold_z and gripper <= self.gripper_max
        in_hand = self.tcp_distance <= self.in_hand_radius
        seated = self.stall_band[0] <= gripper <= self.stall_band[1]

        self.height_streak = self.height_streak + 1 if lifted else 0
        if self.height_streak >= self.sustain:
            self.height_only = True
        if lifted and in_hand and seated:
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
            "height_only": self.height_only,
            "in_hand_radius": self.in_hand_radius,
            "stall_band": [self.stall_band[0], self.stall_band[1]],
            "tcp_distance": self.tcp_distance,
            "gripper": self.gripper,
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
    ``not_in_hand``
        The object was up, and stayed up, but the hand was not holding it --
        it rode the arm, or the jaws stalled nowhere near this object's width.
        The old height-only predicate scored these as successes; they are the
        ones that would poison a dataset, so they get their own name rather
        than being filed under ``slipped``.
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
    if monitor.height_only:
        return "not_in_hand"
    peak = monitor.peak_z
    if peak >= monitor.threshold_z:
        return "slipped"
    if peak >= monitor.spawn_z + nudge:
        return "short_lift"
    if expert.timeouts:
        return "timeout"
    return "no_grasp"
