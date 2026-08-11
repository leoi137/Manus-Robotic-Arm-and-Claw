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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from manus.specs import JOINT_LIMITS

if TYPE_CHECKING:
    import isaaclab.sim as sim_utils

Shape = Literal["cuboid", "cylinder", "sphere"]
"""Primitive an object is spawned as. Cylinders and spheres use :attr:`ObjectSpec.radius`."""

YawSymmetry = Literal["quarter", "half", "free"]
"""How much yaw freedom the grasp has; see :attr:`ObjectSpec.yaw_symmetry`."""

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


def close_target_for_width(width: float) -> float:
    """CLOSE target for an object `width` metres across, in radians.

    One measurement carried along the jaw geometry::

        contact(width) = MEASURED_STALL_30MM_RAD + (width - 0.030) / JAW_WIDTH_PER_RAD
        close_target    = contact(width) - SQUEEZE_RAD

    which, since both terms shift together, is just the tuned 30 mm target
    shifted by the width difference. Every catalogue object therefore closes
    with the *same* :data:`SQUEEZE_RAD` past its own contact angle — the thing
    that was actually tuned — rather than the same absolute angle.

    Args:
        width: Distance the jaws must span, metres.

    Returns:
        The ``gripper`` joint target for CLOSE, radians.

    Raises:
        ValueError: The target falls outside the jaw joint's travel, so the
            articulation would clamp it and the squeeze would be silently
            weaker than commanded (or absent).
    """
    target = CLOSE_TARGET_30MM_RAD + (width - REFERENCE_WIDTH_M) / JAW_WIDTH_PER_RAD
    lower, upper = JOINT_LIMITS["gripper"]
    if not lower <= target <= upper:
        raise ValueError(
            f"a {width * 1e3:.1f} mm object needs a close target of {target:.3f} rad, "
            f"outside the gripper's travel [{lower:.3f}, {upper:.3f}]"
        )
    return target


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
        yaw_symmetry: Which grasp yaws line the jaws up with the object:
            ``"quarter"`` (square section — every 90°), ``"half"`` (rectangular
            section — every 180°, the only two that span the short axis) or
            ``"free"`` (round — any yaw at all). Declared per object and
            checked against the geometry, since it is the one property the
            expert cannot recover from :attr:`grasp_width_m` alone.
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
    def geometric_yaw_symmetry(self) -> YawSymmetry:
        """The symmetry class this object's own geometry implies."""
        if self.shape != "cuboid":
            return "free"  # round about local z: a cylinder on its end, a ball
        assert self.half_extents is not None  # guaranteed by __post_init__
        half_x, half_y, _ = self.half_extents
        return "quarter" if abs(half_x - half_y) <= 1e-9 else "half"

    def make_spawn_cfg(
        self, color: tuple[float, float, float] = DEFAULT_COLOR
    ) -> sim_utils.ShapeCfg:
        """Build this object's Isaac Lab shape spawner.

        The returned cfg spawns ``<prim>`` as an Xform carrying the rigid-body
        and mass APIs, with the collider and its materials under
        ``<prim>/geometry`` — the layout
        :func:`isaaclab.sim.spawners.shapes.shapes._spawn_geom_from_prim_type`
        imposes, and the reason the per-episode colour write targets
        ``<prim>/geometry/material/Shader``.

        .. warning::
            Requires a running Isaac Sim app: ``isaaclab`` is imported here,
            not at module scope, to keep this module sim-free.

        Args:
            color: Linear-RGB diffuse colour to spawn with.

        Returns:
            A :class:`~isaaclab.sim.CuboidCfg`, :class:`~isaaclab.sim.CylinderCfg`
            or :class:`~isaaclab.sim.SphereCfg`.
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
            "physics_material": sim_utils.PhysxRigidBodyMaterialCfg(
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
        return sim_utils.CylinderCfg(radius=self.radius, height=self.height, axis="Z", **shared)


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
    ),
    # A 16 mm acrylic die (~1200 kg/m^3): the smallest thing in the catalogue,
    # and the one whose close target sits deepest into the jaw joint's negative
    # travel (-0.142 rad, 0.03 rad off the stop).
    "die_16mm": ObjectSpec(
        name="die_16mm",
        shape="cuboid",
        half_extents=(0.008, 0.008, 0.008),
        mass_kg=0.005,
        grasp_width_m=0.016,
        spawn_z=0.008,
        close_target_rad=close_target_for_width(0.016),
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
