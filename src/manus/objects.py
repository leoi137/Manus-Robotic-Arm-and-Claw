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

if TYPE_CHECKING:
    import isaaclab.sim as sim_utils

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

CLOSE_TARGET_30MM_RAD = 0.05
"""``gripper`` joint target for the CLOSE phase on a 30 mm object, in radians.

The gripper joint opens with increasing angle (see
``manus.control.GRIPPER_OPEN``/``GRIPPER_CLOSED``), so commanding *below* the
angle at which the jaws first touch makes the servo squeeze against its effort
limit. **TUNED at Step 7** (was 0.2, a placeholder). Measured in sim over the
Step 8 gate, a held 30 mm object stalls the jaws at 0.27-0.35 rad — wider than
the 0.16 rad the visual meshes predict, because PhysX collides the convex
approximations of the concave fingers. 0.05 rad therefore commands 0.22-0.30 rad
past contact, which at kp 17.8 asks for more than the servo's 3.35 N·m limit and
so holds the object at full effort. Backing off to 0.18 rad was tried and is
worse: it dropped an object the full squeeze held (16-attempt probe over the
lowest-friction gate draws). Both catalogue objects are 3 cm across and share
the value.
"""


@dataclass(frozen=True)
class ObjectSpec:
    """A graspable rigid body: geometry, inertia and grasp parameters.

    Attributes:
        name: Catalogue key; also the object identity recorded in manifests.
        shape: ``"cuboid"`` (uses :attr:`half_extents`) or ``"cylinder"``
            (uses :attr:`radius` and :attr:`height`, axis along local +Z).
        mass_kg: Body mass in kilograms.
        grasp_width_m: Width the jaws must span, in metres — the object's size
            across the axis the gripper closes along.
        spawn_z: Height of the body origin above the ground plane at rest, in
            metres.
        close_target_rad: ``gripper`` joint target that squeezes this object,
            in radians (see :data:`CLOSE_TARGET_30MM_RAD`).
        half_extents: Half-sizes (x, y, z) in metres. Cuboids only.
        radius: Cylinder radius in metres. Cylinders only.
        height: Cylinder height in metres. Cylinders only.
    """

    name: str
    shape: Literal["cuboid", "cylinder"]
    mass_kg: float
    grasp_width_m: float
    spawn_z: float
    close_target_rad: float
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
        else:
            raise ValueError(f"{self.name}: unknown shape {self.shape!r}")

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
            A :class:`~isaaclab.sim.CuboidCfg` or :class:`~isaaclab.sim.CylinderCfg`.
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
        close_target_rad=CLOSE_TARGET_30MM_RAD,
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
        close_target_rad=CLOSE_TARGET_30MM_RAD,
    ),
}
"""Graspable objects by name. ``cube_3cm`` is the primary task object."""
