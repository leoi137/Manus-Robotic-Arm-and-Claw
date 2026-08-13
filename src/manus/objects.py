"""Graspable object catalogue for the synthetic grasping task.

Every length is metres, every mass kilograms, every angle radians. Objects rest
on the ground plane (the work surface), so ``spawn_z`` is the height of the
body-frame origin when the object sits at rest.

Sim-free: the dataclass and the :data:`OBJECTS` table import without Isaac Sim.
Only :meth:`ObjectSpec.make_spawn_cfg` needs ``isaaclab``, and it imports it
lazily inside the function body so dataset tooling and unit tests can read the
catalogue on the CPU-only side of the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from manus.specs import JOINT_LIMITS, STS3215_EFFORT_LIMIT, STS3215_KP

if TYPE_CHECKING:
    import isaaclab.sim as sim_utils

Shape = Literal["cuboid", "cylinder", "sphere"]
"""Primitive an object is spawned as. Cylinders and spheres use :attr:`ObjectSpec.radius`."""

YawSymmetry = Literal["quarter", "half", "free"]
"""How much yaw freedom the grasp has; see :attr:`ObjectSpec.yaw_symmetry`."""

GraspMode = Literal["top", "side"]
"""How the hand comes at an object; see :attr:`ObjectSpec.grasp_mode`."""

DEFAULT_COLOR: tuple[float, float, float] = (0.75, 0.25, 0.2)
"""Linear-RGB diffuse colour an object spawns with.

Cosmetic only: :func:`manus.randomize.draw_episode` redraws the colour every
episode and :func:`manus.task_scene.write_object_color` overwrites the shader,
so colour is deliberately not part of an object's identity.
"""

NOMINAL_FRICTION = 1.0
"""Static/dynamic friction an object spawns with.

Midpoint of :data:`manus.randomize.FRICTION_RANGE`; the per-episode draw
overwrites it through the PhysX material tensor API at reset.
"""

REFERENCE_WIDTH_M = 0.030
"""Grasp width the close target was tuned at, metres: the cube and the cylinder."""

CLOSE_TARGET_30MM_RAD = 0.05
"""``gripper`` joint target for the CLOSE phase on a 30 mm object, in radians.

The gripper joint opens with increasing angle (see
``manus.control.GRIPPER_OPEN``/``GRIPPER_CLOSED``), so commanding *below* the
angle at which the jaws first touch makes the servo squeeze against its effort
limit. **TUNED at Step 7**, then re-measured at Step 20: under the convex-hull
colliders it was tuned against, a held 30 mm object stalled the jaws at
0.27-0.35 rad; with the jaws switched to SDF colliders the same grasp stalls at
:data:`MEASURED_STALL_30MM_RAD`, i.e. where the visual meshes always said it
should. The *target* did not need re-tuning — 0.05 rad still commands
:data:`SQUEEZE_RAD` past contact, which at kp 17.8 asks the servo for 2.5 N·m of
its 3.35 N·m limit and so holds the object hard. Backing off to 0.18 rad was
tried under the hull and is worse: it dropped an object the full squeeze held
(16-attempt probe over the lowest-friction gate draws).
"""

MEASURED_STALL_30MM_RAD = 0.189
"""Jaw angle a held 30 mm object stalls the fingers at, radians. **Measured.**

Step 20, from the filmed grasp that verified the SDF collider fix (both pads
visibly touching the cube). The visual meshes predict 0.195 for the same grasp,
so the anchor and the mesh geometry agree to 6 mrad — which is what licenses
:func:`close_target_for_width` to carry this one measurement to other widths
along the mesh-measured slope.
"""

SQUEEZE_RAD = MEASURED_STALL_30MM_RAD - CLOSE_TARGET_30MM_RAD
"""How far past contact the CLOSE target is commanded, radians (0.139).

The quantity actually being held constant across the catalogue: an object's
close target is its own contact angle minus this. At kp 17.8 it asks for 2.5 N·m
against a 3.35 N·m limit, so the servo squeezes below saturation with margin for
a jaw that has to travel a little further than the geometry predicted.
"""

MIN_SQUEEZE_RAD = 0.5 * STS3215_EFFORT_LIMIT / STS3215_KP
"""Smallest squeeze worth commanding, radians (0.094).

The floor the catalogue's own test asserts: at ``kp`` 17.8 a squeeze of this
much asks the servo for half its 3.35 N·m effort limit, and anything less is a
grip that a lift can shake loose. It is the bound :data:`LIGHT_SQUEEZE_RAD`
sits just above.
"""

LIGHT_SQUEEZE_RAD = 0.10
"""Squeeze used for an object too small to absorb the full one, radians.

:data:`SQUEEZE_RAD` was tuned on a 60 g, 30 mm cube, where 0.139 rad past
contact is 10 mm of jaw travel the object simply refuses to give. On the 16 mm
die the *same* 0.139 rad is 63% of the object's whole width and leaves the close
target 0.032 rad off the jaw's -0.1745 rad hard stop: a die that yields even
2 mm (the filmed grasp stalls 1.7 mm inside nominal contact) drives the jaws
into the stop, where the servo has no position error left to squeeze with and
the grip goes slack. 0.10 rad is a hair above :data:`MIN_SQUEEZE_RAD` -- still
1.78 N·m, still 7 mm of over-travel -- and doubles the margin to the stop.
"""

JAW_WIDTH_PER_RAD = 0.0727
"""Jaw opening gained per radian of ``gripper`` angle, metres. **Measured.**

Off the vendored jaw STLs (``tests/test_expert_logic.py`` re-derives it, so a
re-exported model cannot drift away from it): the narrowest opening between the
pads over the band that engages a resting object is linear in the joint angle to
better than 0.5 mm over the whole useful range,
``gap ~ 15.5 mm + 72.7 mm/rad * angle``. Inverting that fit reproduces the
mesh-measured contact angle of every catalogue width (16-40 mm) to within
0.3 mrad, which is far inside the 6 mrad the mesh and the sim disagree by.
"""


def contact_angle_for_width(width: float) -> float:
    """Jaw angle at which the pads first touch an object `width` metres across.

    One sim measurement carried along the mesh-measured jaw slope::

        contact(width) = MEASURED_STALL_30MM_RAD + (width - 0.030) / JAW_WIDTH_PER_RAD

    This is the angle a *held* object stops the jaws at, so it is both the
    anchor of :func:`close_target_for_width` and the reference the success
    predicate checks a measured stall against
    (:class:`manus.expert.GraspSuccessMonitor`): jaws that ran past it by more
    than a tolerance closed on nothing.
    """
    return MEASURED_STALL_30MM_RAD + (width - REFERENCE_WIDTH_M) / JAW_WIDTH_PER_RAD


def close_target_for_width(width: float, squeeze: float = SQUEEZE_RAD) -> float:
    """CLOSE target for an object `width` metres across, in radians.

    ``contact_angle_for_width(width) - squeeze`` — which, since both terms
    shift together, is just the tuned 30 mm target shifted by the width
    difference. Every catalogue object therefore closes with the *same*
    :data:`SQUEEZE_RAD` past its own contact angle — the thing that was actually
    tuned — rather than the same absolute angle. The one exception is an object
    small enough that the full squeeze is a hazard rather than a grip; see
    :data:`LIGHT_SQUEEZE_RAD`.

    Args:
        width: Distance the jaws must span, metres.
        squeeze: How far past contact to command, radians.

    Returns:
        The ``gripper`` joint target for CLOSE, radians.

    Raises:
        ValueError: The target falls outside the jaw joint's travel, so the
            articulation would clamp it and the squeeze would be silently
            weaker than commanded (or absent).
    """
    target = contact_angle_for_width(width) - squeeze
    lower, upper = JOINT_LIMITS["gripper"]
    if not lower <= target <= upper:
        raise ValueError(
            f"a {width * 1e3:.1f} mm object needs a close target of {target:.3f} rad, "
            f"outside the gripper's travel [{lower:.3f}, {upper:.3f}]"
        )
    return target


REFERENCE_MASS_KG = 0.06
"""Object mass the CLOSE ramp was tuned at, kilograms: the cube."""

CLOSE_RAMP_REFERENCE_STEPS = 60
"""CLOSE ramp of a :data:`REFERENCE_MASS_KG` object, in control steps. **Tuned.**

Step 7's number, and the one the 200-attempt Step 8 gate is anchored on: 60
steps is 0.7 rad/s at the jaw, slow enough not to flick the 60 g cube out of the
hand and firm enough not to slip. See :attr:`manus.expert.ExpertConfig.close_ramp`.
"""

CLOSE_RAMP_MAX_STEPS = 150
"""Slowest CLOSE ramp the rule will ask for, in control steps. **Measured.**

What 150 buys is measured and stands: at 60 steps the closing jaw punts the 5 g
die out of the hand at first touch, at 150 the die does not move at all through
contact (Step 21 preview, ``runs/object_previews/die_16mm.mp4``: its centre
holds 8.0-8.3 mm from the jaws' first graze at 0.08 rad right through their
nominal 16 mm contact at -0.004).

What 150 does *not* buy has been retracted. The old "at 150 it seats" reading
came from an attempt the height-only predicate scored as a success; under the
Step 21 predicate the same attempt is a ``no_grasp``, because the die is
squeezed back *out* 1.4 mm into the squeeze -- 8.0 -> 19.0 mm in six control
steps, ~80 mm/s, an order of magnitude faster than the 7 mm/s the pads are
closing at -- and the jaws then run to their commanded target on an empty hand.
That is a squeeze failure, not an approach failure, so the ramp is not the knob
that fixes it and a longer one is not indicated: the die survives the part of
CLOSE the ramp governs.

150 remains the clamp because 5 s of closing at 30 Hz is the slowest close
anyone has verified, and past it the only certain effect is a longer episode.
"""


def close_ramp_for_mass(mass_kg: float) -> int:
    """Steps the CLOSE ramp should take for an object of `mass_kg`.

    Two measured anchors -- 60 g closes in 60 steps, 5 g needs 150 -- joined by
    the simplest monotone curve through them::

        steps = clamp(60 * sqrt(0.060 / mass), 60, 150)

    Why light objects need a slower jaw: a position-servo jaw that overshoots
    an object's surface by one control step's travel delivers an impulse
    proportional to its speed, and the velocity that impulse imparts goes as
    ``1 / mass``. Taken literally that model would ask for ``steps ~ 1 / mass``
    (720 steps for the die), which is far more than the die actually needed --
    friction and the opposing pad absorb most of it -- so the exponent is the
    geometric middle of "no mass dependence" and the impulse model, calibrated
    to land on the two anchors rather than derived from first principles alone.

    Args:
        mass_kg: Object mass, kilograms.

    Returns:
        Ramp length in control steps, in ``[60, 150]``.

    Raises:
        ValueError: `mass_kg` is not positive.
    """
    if mass_kg <= 0.0:
        raise ValueError(f"mass must be positive, got {mass_kg}")
    steps = CLOSE_RAMP_REFERENCE_STEPS * math.sqrt(REFERENCE_MASS_KG / mass_kg)
    return int(round(min(max(steps, CLOSE_RAMP_REFERENCE_STEPS), CLOSE_RAMP_MAX_STEPS)))


@dataclass(frozen=True)
class ObjectSpec:
    """A graspable rigid body: geometry, inertia and grasp parameters.

    **The jaws close along the object's local x.** That convention is not free
    to choose — :func:`manus.expert.tcp_target` stands the tool so the object
    centre sits along the tool's own +x, so ``grasp_width_m`` has to be the
    local-x extent and nothing else. It is checked here rather than trusted,
    because for a square-section object (all this pipeline carried until the
    catalogue grew) getting it wrong is invisible, while for a domino it grasps
    the long way round and the jaws never reach.

    Attributes:
        name: Catalogue key; also the object identity recorded in manifests.
        shape: ``"cuboid"`` (uses :attr:`half_extents`), ``"cylinder"`` (uses
            :attr:`radius` and :attr:`height`, axis along local +Z) or
            ``"sphere"`` (uses :attr:`radius`).
        mass_kg: Body mass in kilograms.
        grasp_width_m: Width the jaws must span, in metres — the object's size
            along its local x, the axis the gripper closes along.
        spawn_z: Height of the body origin above the ground plane at rest, in
            metres, i.e. half :attr:`extent_z`.
        close_target_rad: ``gripper`` joint target that squeezes this object,
            in radians (see :func:`close_target_for_width`).
        close_ramp: Steps CLOSE spends ramping the jaws shut, overriding the
            mass rule (:func:`close_ramp_for_mass`). None uses the rule, which
            is what every catalogue object does — the field exists so a
            re-tuning run can pin one object without touching the rule.
        tip_clearance_m: Gap to leave between the fingertips and the table at
            the grasp, metres, overriding :data:`manus.expert.MIN_TIP_CLEARANCE`.
            Only bites on an object too short to centre the pads on, i.e. the
            puck; None uses the default.
        close_creep: Whether CLOSE creeps through this object's contact band
            instead of ramping straight through it
            (:func:`manus.expert.close_command`). Off by default, and off for
            every object the tuned ramp already grasps -- it is for the two the
            *shove* destabilises: an object the closing jaw can topple (the
            standing cylinder) or squirt out of the hand (the 40 mm puck).
        seat_close: Whether the FSM walks a :data:`manus.expert.SEAT` between
            the approach and CLOSE -- the arm closing the last
            :data:`manus.expert.JAW_CLEARANCE` onto the static pad at a creep,
            so the moving jaw finds the object already against its stop and
            has no gap to shove it across. The other half of the same two
            objects' problem, and the half ``close_creep`` could not reach:
            creeping makes the shove slow, seating removes it. Off by default,
            and off for every object the tuned closure already grasps -- their
            gate anchors are pinned, and a state they do not walk cannot move
            them.
        experimental: Whether this object is known not to grasp reliably. It
            stays in :data:`OBJECTS` and can be run by name, but it is left out
            of :data:`DEFAULT_OBJECTS`, so a sweep over "every object" does not
            quietly bake its failures into a dataset.
        yaw_symmetry: Which grasp yaws line the jaws up with the object:
            ``"quarter"`` (square section — every 90°), ``"half"`` (rectangular
            section — every 180°, the only two that span the short axis) or
            ``"free"`` (round — any yaw at all). Declared per object and
            checked against the geometry, since it is the one property the
            expert cannot recover from :attr:`grasp_width_m` alone.
        grasp_mode: ``"top"`` — the hand comes straight down, the default and
            what every object but one is grasped with — or ``"side"``, the hand
            laid flat and driven in radially so the jaws close across the
            object horizontally, the way a hand takes a cup. Side mode is a
            different tool family, a different placement region and a different
            approach state in the FSM; see :data:`manus.kinematics.TOOL_HORIZONTAL`
            and :data:`manus.expert.ADVANCE`.
        half_extents: Half-sizes (x, y, z) in metres. Cuboids only.
        radius: Cylinder or sphere radius in metres.
        height: Cylinder height in metres. Cylinders only.
    """

    name: str
    shape: Shape
    mass_kg: float
    grasp_width_m: float
    spawn_z: float
    close_target_rad: float
    yaw_symmetry: YawSymmetry
    half_extents: tuple[float, float, float] | None = None
    radius: float | None = None
    height: float | None = None
    close_ramp: int | None = None
    tip_clearance_m: float | None = None
    close_creep: bool = False
    seat_close: bool = False
    experimental: bool = False
    grasp_mode: GraspMode = "top"

    def __post_init__(self) -> None:
        if self.shape == "cuboid":
            if self.half_extents is None:
                raise ValueError(f"{self.name}: a cuboid needs half_extents")
        elif self.shape == "cylinder":
            if self.radius is None or self.height is None:
                raise ValueError(f"{self.name}: a cylinder needs radius and height")
        elif self.shape == "sphere":
            if self.radius is None:
                raise ValueError(f"{self.name}: a sphere needs radius")
        else:
            raise ValueError(f"{self.name}: unknown shape {self.shape!r}")
        if abs(self.grasp_width_m - self.local_x_extent) > 1e-9:
            raise ValueError(
                f"{self.name}: grasp_width_m {self.grasp_width_m} is not the local-x extent "
                f"{self.local_x_extent} — the jaws close along local x, so it has to be"
            )
        if self.yaw_symmetry != self.geometric_yaw_symmetry:
            raise ValueError(
                f"{self.name}: declared yaw_symmetry {self.yaw_symmetry!r}, but the geometry "
                f"is {self.geometric_yaw_symmetry!r}"
            )
        if self.close_ramp is not None and self.close_ramp < 1:
            raise ValueError(f"{self.name}: close_ramp must be at least one step")
        if self.tip_clearance_m is not None and self.tip_clearance_m < 0.0:
            raise ValueError(f"{self.name}: tip_clearance_m cannot be negative")
        if self.grasp_mode not in ("top", "side"):
            raise ValueError(f"{self.name}: unknown grasp_mode {self.grasp_mode!r}")
        if self.grasp_mode == "side" and self.yaw_symmetry != "free":
            # A side grasp spends the arm's fifth freedom on the closing plane's
            # roll, which leaves the *approach azimuth* pinned by the reach
            # direction (manus.kinematics.TOOL_HORIZONTAL). So the jaws arrive
            # along whatever radius the object happens to sit on and the object
            # has to look the same from all of them.
            raise ValueError(
                f"{self.name}: a side grasp cannot choose its approach azimuth, so the "
                f"object must be round about its own z ('free'), not {self.yaw_symmetry!r}"
            )

    @property
    def local_x_extent(self) -> float:
        """Size along local x — the axis the jaws close along — in metres."""
        if self.shape == "cuboid":
            assert self.half_extents is not None  # guaranteed by __post_init__
            return 2.0 * self.half_extents[0]
        assert self.radius is not None  # guaranteed by __post_init__
        return 2.0 * self.radius

    @property
    def extent_z(self) -> float:
        """Full height of the object as it rests on the ground, in metres."""
        if self.shape == "cuboid":
            assert self.half_extents is not None  # guaranteed by __post_init__
            return 2.0 * self.half_extents[2]
        assert self.radius is not None  # guaranteed by __post_init__
        return self.height if self.shape == "cylinder" else 2.0 * self.radius

    @property
    def top_z(self) -> float:
        """Height of the object's highest point above the ground at rest, metres."""
        return self.spawn_z + 0.5 * self.extent_z

    @property
    def contact_angle_rad(self) -> float:
        """Jaw angle this object stops the closing pads at, radians.

        :func:`contact_angle_for_width` of its own grasp width: what
        :attr:`close_target_rad` is measured *from*, and what a measured CLOSE
        stall is compared against to tell a grasp from an empty closure.
        """
        return contact_angle_for_width(self.grasp_width_m)

    @property
    def squeeze_rad(self) -> float:
        """How far past contact this object's CLOSE target is commanded, radians.

        :data:`SQUEEZE_RAD` for everything the tuned squeeze suits, and the
        target's own value for anything given a lighter one — read back from
        :attr:`close_target_rad` rather than declared, so an overridden target
        and its squeeze cannot disagree.
        """
        return self.contact_angle_rad - self.close_target_rad

    @property
    def close_ramp_steps(self) -> int:
        """Steps CLOSE ramps the jaws shut over: the override, else the mass rule."""
        return self.close_ramp if self.close_ramp is not None else close_ramp_for_mass(self.mass_kg)

    @property
    def geometric_yaw_symmetry(self) -> YawSymmetry:
        """The symmetry class this object's own geometry implies."""
        if self.shape != "cuboid":
            return "free"  # round about local z: a cylinder on its end, a ball
        assert self.half_extents is not None  # guaranteed by __post_init__
        half_x, half_y, _ = self.half_extents
        return "quarter" if abs(half_x - half_y) <= 1e-9 else "half"

    def make_spawn_cfg(
        self, color: tuple[float, float, float] = DEFAULT_COLOR
    ) -> sim_utils.ShapeCfg | sim_utils.MeshCfg:
        """Build this object's Isaac Lab shape spawner.

        The returned cfg spawns ``<prim>`` as an Xform carrying the rigid-body
        and mass APIs, with the collider and its materials under
        ``<prim>/geometry`` — the layout both
        :func:`isaaclab.sim.spawners.shapes.shapes._spawn_geom_from_prim_type`
        (cuboids, spheres) and
        :func:`isaaclab.sim.spawners.meshes.meshes._spawn_mesh_geom_from_mesh`
        (cylinders) impose, and the reason the per-episode colour write targets
        ``<prim>/geometry/material/Shader``.

        .. warning::
            Requires a running Isaac Sim app: ``isaaclab`` is imported here,
            not at module scope, to keep this module sim-free.

        Args:
            color: Linear-RGB diffuse colour to spawn with.

        Returns:
            A :class:`~isaaclab.sim.CuboidCfg`, :class:`~isaaclab.sim.SphereCfg`
            or :class:`~isaaclab.sim.MeshCylinderCfg`.
        """
        import isaaclab.sim as sim_utils

        shared = {
            "mass_props": sim_utils.MassPropertiesCfg(mass=self.mass_kg),
            # Small, light objects squeezed between two jaws need more position
            # iterations than the PhysX default to stop the grasp jittering.
            "rigid_props": sim_utils.PhysxRigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,  # m/s
            ),
            "collision_props": sim_utils.PhysxCollisionPropertiesCfg(collision_enabled=True),
            # RigidBodyMaterialCfg, not PhysxRigidBodyMaterialCfg: identical config
            # (same spawn func, fields and defaults; it subclasses the Physx one),
            # but the mesh spawner's material type check accepts only this name.
            "physics_material": sim_utils.RigidBodyMaterialCfg(
                static_friction=NOMINAL_FRICTION,
                dynamic_friction=NOMINAL_FRICTION,
                restitution=0.0,
            ),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
        }
        if self.shape == "cuboid":
            assert self.half_extents is not None  # guaranteed by __post_init__
            return sim_utils.CuboidCfg(
                size=tuple(2.0 * half for half in self.half_extents), **shared
            )
        if self.shape == "sphere":
            return sim_utils.SphereCfg(radius=self.radius, **shared)
        # A triangle-mesh cylinder with a convex-hull collider, NOT the analytic
        # ``CylinderCfg``. PhysX has no native cylinder shape: an analytic cylinder
        # prim becomes "custom geometry", and its narrowphase against the jaws' SDF
        # colliders generates no contacts at all -- measured, not surmised
        # (``runs/contact_probe/probe.json``): the seated cylinder rested 2 mm from
        # the fixed pad through all of SEAT and then slid 40 mm *through* the pad
        # plane over 140 CLOSE steps with zero contacts reported, before being
        # caught 4.2 mm deep by the jaw's upper structure at 54.9 N. The hull of
        # the 32-section mesh is within 0.08 mm of the true surface, and
        # convex-vs-SDF is the pair every working cuboid grasp already exercises.
        return sim_utils.MeshCylinderCfg(
            radius=self.radius, height=self.height, axis="Z", **shared
        )


OBJECTS: dict[str, ObjectSpec] = {
    # ~2200 kg/m^3: a dense resin/plastic block, heavy enough not to be flicked
    # away by contact impulses but well inside the STS3215 payload.
    "cube_3cm": ObjectSpec(
        name="cube_3cm",
        shape="cuboid",
        half_extents=(0.015, 0.015, 0.015),
        mass_kg=0.06,
        grasp_width_m=0.03,
        spawn_z=0.015,
        close_target_rad=close_target_for_width(0.03),
        yaw_symmetry="quarter",
    ),
    # d = 3 cm, h = 6 cm, standing on one flat end (~1900 kg/m^3). Grasped
    # across the diameter, so the jaw span matches the cube's.
    #
    # THE CATALOGUE'S SIDE GRASP -- taken like a cup of water, hand flat, jaws
    # closing horizontally across its waist at its own mid-height. That is the
    # answer to the failure top-down kept having with it: 60 mm standing tall
    # is 26 mm above a mid-height top-down grasp, deep into the part of the
    # closing sweep where the moving finger's face leans inward
    # (manus.expert.JAW_PARALLEL_REACH), so the jaw met its upper body 26 mm
    # above its centre of mass and 2.0 mm before the pads reached its 30 mm
    # body, and pushed it over. Raising the top-down grasp to 36 mm shrank that
    # lead but kept the same hand-over-the-lid geometry; coming in from the side
    # removes it. Horizontally the object is 30 mm across at every height, so
    # nothing about the object is "above the TCP" any more -- only its own
    # radius, 11 mm of it, is, well inside the parallel band.
    #
    # STILL NOT GRASPED, as of the Step 24 takes -- read this before sweeping it
    # into a dataset. What Step 24 fixed is the *approach* and the *camera*: the
    # hand now arrives 0.97 mm from its waypoint instead of 7.9 mm, keeps 11 mm
    # of table clearance where it used to stand on the table, and the wrist
    # camera is above the table looking down at the object instead of buried
    # under the ground plane (runs/object_previews/cylinder_3cm_fixed*.png).
    # CLOSE still loses it: a one-sided push at a 40 mm contact height tips a
    # 30 mm cylinder before it slides (0.29 N against 0.64 N), and a cylinder
    # tilted even 3 deg is wider than the closing gap, so the jaws lever it out.
    # See manus.expert.close_command.
    #
    # Its height is what makes it the side case rather than the puck: the hand's
    # own housing hangs 27.8 mm below the tool axis (manus.expert.SIDE_JAW_DEPTH),
    # so a side grasp is only defined on something tall enough to be taken well
    # above that. This one is taken at its *cup* height, 40 mm -- two thirds of
    # the way up, above its centre of mass, 12.2 mm of table clearance -- which
    # is what the first filmed attempt showed was needed: taken at its 30 mm
    # mid-height the housing had 5.8 mm of clearance and 7.4 mm of un-cancelled
    # gravity droop put it on the table.
    "cylinder_3cm": ObjectSpec(
        name="cylinder_3cm",
        shape="cylinder",
        radius=0.015,
        height=0.06,
        mass_kg=0.08,
        grasp_width_m=0.03,
        spawn_z=0.03,
        close_target_rad=close_target_for_width(0.03),
        yaw_symmetry="free",
        grasp_mode="side",
        close_creep=True,
        seat_close=True,
        experimental=True,  # fails at CLOSE: seating-shove energy dump (see fix-crew notes)
    ),
    # A 16 mm acrylic die (~1200 kg/m^3): the smallest thing in the catalogue,
    # and the object the arm's own convergence residual is worst for -- 5.9 mm
    # of TCP error left over at the end of DESCEND is 37% of its grasp width
    # against the cube's 20%, which is why manus.expert.converge_tol scales the
    # bar by the object and the die gets 0.0107 rad rather than 0.02. See
    # CLOSE_RAMP_MAX_STEPS for the part of its Step 21 failure the ramp does
    # *not* explain.
    # and at 5 g the lightest thing with corners. Both of its deviations from
    # the catalogue defaults are about the same fact -- a 5 g, 16 mm object has
    # nothing to absorb the hand with. The jaws come in over
    # close_ramp_for_mass(0.005) = 150 steps rather than the cube's 60, and they
    # squeeze LIGHT_SQUEEZE_RAD past contact rather than the full SQUEEZE_RAD,
    # which moves the close target from -0.143 rad (0.032 off the jaw's hard
    # stop, where a die that yields at all lets the servo run out of squeeze) to
    # -0.104 (0.071 off it).
    "die_16mm": ObjectSpec(
        name="die_16mm",
        shape="cuboid",
        half_extents=(0.008, 0.008, 0.008),
        mass_kg=0.005,
        grasp_width_m=0.016,
        spawn_z=0.008,
        close_target_rad=close_target_for_width(0.016, squeeze=LIGHT_SQUEEZE_RAD),
        yaw_symmetry="quarter",
    ),
    # A double-six domino lying face up: 20 x 40 x 15 mm, ~1700 kg/m^3 (urea
    # resin). Rectangular, so only two of the four quarter turns put the jaws
    # across the 20 mm width -- the first object in the catalogue for which the
    # local-x convention is load-bearing rather than decorative.
    "domino_20x40": ObjectSpec(
        name="domino_20x40",
        shape="cuboid",
        half_extents=(0.010, 0.020, 0.0075),
        mass_kg=0.02,
        grasp_width_m=0.020,
        spawn_z=0.0075,
        close_target_rad=close_target_for_width(0.020),
        yaw_symmetry="half",
    ),
    # A 40 x 10 mm plastic puck lying flat (~1200 kg/m^3), gripped across its
    # full diameter. Two firsts: the widest grasp in the catalogue (0.187 rad,
    # still 1.56 rad off the jaw's open stop), and the only object short enough
    # that the pads cannot centre on it -- see manus.expert.grasp_height, which
    # raises the grasp to keep the fingertips off the table and so grips the
    # puck's top half.
    #
    # EXPERIMENTAL: the closing jaw punts it off the table. The moving finger
    # meets a 10 mm rim tilted and still descending (16 mm of tip drop per
    # radian at the puck's 0.33 rad contact angle), so it touches the top edge
    # first and drags it down and in. That is a property of the jaw's sweep, not
    # of the grasp height: over the whole feasible 3-7 mm tip-clearance band the
    # first contact stays within 1.4 mm of the puck's top face while the pads'
    # purchase on the rim falls from 7 mm to 3 mm (measured off the meshes in
    # tests/test_expert_logic.py). Left in the catalogue, and tip_clearance_m is
    # exposed so the band can be swept in sim, but out of DEFAULT_OBJECTS.
    "puck_d40x10": ObjectSpec(
        name="puck_d40x10",
        shape="cylinder",
        radius=0.020,
        height=0.010,
        mass_kg=0.015,
        grasp_width_m=0.040,
        spawn_z=0.005,
        close_target_rad=close_target_for_width(0.040),
        yaw_symmetry="free",
        experimental=True,
    ),
    # The 10 mm puck's respec: same 40 mm disc, twice as thick (~1200 kg/m^3,
    # 30 g). The width, and so the close target and the contact angle, are
    # identical -- what changes is the one thing the thin puck's failure was
    # about, how much rim the pads get.
    #
    # A 20 mm rim is tall enough for grasp_height to centre the pads on it
    # (the fingertips end up 7.7 mm off the table, past MIN_TIP_CLEARANCE with
    # 2.7 mm to spare), so the pads cover 12.3 mm of the rim. The 10 mm puck
    # cannot be centred at all: raised until the tips clear, its pads reach only
    # the top 5.0 mm, and the closing finger -- still descending 16 mm per
    # radian when it arrives -- meets the top edge first and drags the whole
    # thing down and in.
    #
    # STILL NOT GRASPED, as of the Step 24 takes -- read this before sweeping it
    # into a dataset. Four filmed attempts on the rented box (runs/fix_takes)
    # all end ``not_in_hand``: the puck leaves the pads during CLOSE and rides
    # the hand up. The two changes below are measured improvements to that, not
    # a fix -- the best of them holds the full success predicate for 6
    # consecutive steps against the 30 it needs, where the Step 23 preview held
    # 0 (``runs/object_previews/puck_d40x20_fixed.mp4``).
    #
    # THE GRASP IS RAISED 4.1 mm OFF ITS OWN CENTRE, and tip_clearance_m is how,
    # because the bar is this object's alone. Centred, the Step 23 preview was
    # the same ``not_in_hand``: the puck climbed the closing finger and rode it
    # 50 mm up with the arm frozen (``runs/object_previews/puck_d40x20_0000.mp4``,
    # CLOSE steps 76-86; the jaws then ran to their commanded 0.188 rad on an
    # empty hand). What lets it climb is where the hand pinches a 40 mm wall: at
    # the 0.327 rad contact angle the moving finger's face is 18.7 deg off the
    # static pad, so the two meet in a *ridge* 3 mm above the TCP rather than
    # over a face, and the moving finger's own deepest sweep (8.06 mm below the
    # TCP, i.e. 4.1 mm below the puck's centre of mass at the centred grasp)
    # reaches under that centre and levers the rim up.
    #
    # 11.76 mm of tip clearance is the height at which that deepest sweep lands
    # exactly *at* the 10 mm centre of mass -- grasp 14.1 mm, TCP 18.1 mm --
    # and it is bounded on the other side by the pads: at that height they still
    # hold 8.2 mm of rim, past the 8 mm bar the 20 mm respec was chosen for.
    # Nothing else in the catalogue can feel this knob (see
    # manus.expert.tip_clearance). Filmed against the centred grasp it is worth
    # 6 held steps to 4, which is the whole of its evidence.
    #
    # close_creep is the other half: see manus.expert.close_command for the
    # 6.7 mJ the servo stores while shoving this puck across JAW_CLEARANCE, and
    # for what creeping through that shove did and did not buy.
    "puck_d40x20": ObjectSpec(
        name="puck_d40x20",
        shape="cylinder",
        radius=0.020,
        height=0.020,
        mass_kg=0.030,
        grasp_width_m=0.040,
        spawn_z=0.010,
        close_target_rad=close_target_for_width(0.040),
        yaw_symmetry="free",
        tip_clearance_m=0.01176,
        close_creep=True,
        seat_close=True,
        experimental=True,  # fails at CLOSE: seating-shove energy dump (see fix-crew notes)
    ),
    # A regulation ping-pong ball: 40 mm, 2.7 g, hollow (~80 kg/m^3). Kept at
    # its real mass deliberately -- it is the catalogue's slip-and-roll case,
    # two orders of magnitude lighter than the cube, and the only object whose
    # contact with a flat pad is a point rather than a face.
    "pingpong_40mm": ObjectSpec(
        name="pingpong_40mm",
        shape="sphere",
        radius=0.020,
        mass_kg=0.0027,
        grasp_width_m=0.040,
        spawn_z=0.020,
        close_target_rad=close_target_for_width(0.040),
        yaw_symmetry="free",
    ),
    # A 2x4 Duplo brick, studs up: 31.8 x 63.8 x 24 mm and hollow, so ~410
    # kg/m^3. Rectangular like the domino, but two-thirds again as wide and
    # long enough to overhang both pads.
    "duplo_32x64": ObjectSpec(
        name="duplo_32x64",
        shape="cuboid",
        half_extents=(0.0159, 0.0319, 0.012),
        mass_kg=0.02,
        grasp_width_m=0.0318,
        spawn_z=0.012,
        close_target_rad=close_target_for_width(0.0318),
        yaw_symmetry="half",
    ),
}
"""Graspable objects by name. ``cube_3cm`` is the primary task object."""

DEFAULT_OBJECTS: tuple[str, ...] = tuple(
    name for name, spec in OBJECTS.items() if not spec.experimental
)
"""Objects a sweep over "the catalogue" should cover, in catalogue order.

:data:`OBJECTS` minus the :attr:`~ObjectSpec.experimental` ones. Anything left
out is still spawnable and still runnable by name — it is excluded from the
*default*, not from the catalogue.
"""
