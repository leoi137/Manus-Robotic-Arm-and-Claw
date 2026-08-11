"""Grasp task scene: the SO-101 arm, a graspable object, and a key light.

.. warning::
    Requires a running Isaac Sim app; import only after ``AppLauncher(...)``.
    For the sim-free half of the task setup (object catalogue, per-episode
    draws) import :mod:`manus.objects` and :mod:`manus.randomize` instead.

:class:`GraspSceneCfg` extends :class:`manus.scene.SoArmSceneCfg` with the
object under grasp and a distant light, so per-episode light *direction* has
something to act on. :func:`apply_randomization` stamps one
:class:`~manus.randomize.EpisodeDraw` onto a live scene; it delegates to one
small function per randomized quantity so a GPU failure bisects to a single
write.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import warp as wp

from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from manus import randomize
from manus.objects import OBJECTS, ObjectSpec
from manus.randomize import EpisodeDraw
from manus.scene import SoArmSceneCfg

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

DEFAULT_OBJECT = "cube_3cm"
"""Catalogue key of the object :class:`GraspSceneCfg` spawns by default."""

DEFAULT_OBJECT_POS: tuple[float, float, float] = (
    randomize.PAN_AXIS_XY[0] + 0.5 * (randomize.REGION_R[0] + randomize.REGION_R[1]),
    randomize.PAN_AXIS_XY[1],
    OBJECTS[DEFAULT_OBJECT].spawn_z,
)
"""Spawn position (m): mid-radius of the grasp region, straight ahead of the base.

Only the pose the object *loads* at — :func:`write_object_pose` moves it to the
episode's drawn placement before the first step.
"""

DEFAULT_SUN_AZ_DEG = 45.0
"""Distant-light azimuth the scene loads with, in degrees about world +x."""

DEFAULT_SUN_EL_DEG = 55.0
"""Distant-light elevation the scene loads with, in degrees above the horizon."""


def sun_orientation_xyzw(azimuth: float, elevation: float) -> tuple[float, float, float, float]:
    """Orientation (x, y, z, w) aiming a distant light down its own ray.

    A USD ``DistantLight`` emits along its local -Z, so its local +Z must point
    back *towards* the light source: at azimuth ``a`` and elevation ``e``
    (radians) that direction is ``(cos e cos a, cos e sin a, sin e)``.
    """
    return randomize.quat_from_z_axis_xyzw((
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ))


@configclass
class GraspSceneCfg(SoArmSceneCfg):
    """SO-101 scene with a graspable object and a directional key light.

    The inherited entity order (ground, robot, wrist camera, dome light) is
    preserved and the two new entities are appended. That is safe: Isaac Lab
    spawns every non-sensor entity before any sensor regardless of declaration
    order (``InteractiveScene._add_entities_from_cfg``), so the wrist camera
    still finds its parent link.
    """

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=OBJECTS[DEFAULT_OBJECT].make_spawn_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=DEFAULT_OBJECT_POS,
            rot=(0.0, 0.0, 0.0, 1.0),  # (x, y, z, w)
        ),
    )

    # The dome light is ambient by construction, so randomizing a light
    # *direction* needs a directional source. Kept global (not env-scoped) to
    # match the dome light and the ground plane.
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 0.98, 0.95),
            intensity=1500.0,
            angle=1.5,  # degrees of angular size; softens the shadow edges
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=sun_orientation_xyzw(
                math.radians(DEFAULT_SUN_AZ_DEG), math.radians(DEFAULT_SUN_EL_DEG)
            )
        ),
    )


def apply_randomization(
    scene: InteractiveScene,
    draw: EpisodeDraw,
    spec: ObjectSpec = OBJECTS[DEFAULT_OBJECT],
) -> None:
    """Stamp one episode's draw onto a live scene.

    Call after ``sim.reset()`` — the asset views these writes go through only
    exist once the scene is initialised — and before the episode's first
    rendered step, since the camera and material writes are not picked up
    retroactively.

    Args:
        scene: An initialised scene built from :class:`GraspSceneCfg`.
        draw: The attempt's sampled randomization.
        spec: Spec of the object actually spawned; supplies its resting height.
    """
    write_object_pose(scene, draw, spec)
    write_object_color(scene, draw)
    write_object_friction(scene, draw)
    write_ground_albedo(scene, draw)
    write_light_state(scene, draw)
    write_wrist_camera_jitter(scene, draw)


def write_object_pose(scene: InteractiveScene, draw: EpisodeDraw, spec: ObjectSpec) -> None:
    """Teleport the object to the drawn (x, y, yaw) at its resting height.

    Velocities are zeroed too, so momentum from a previous attempt cannot leak
    across the reset. Poses are written in the simulation frame, hence the
    environment origin.
    """
    obj = scene["object"]
    device = obj.device
    count = obj.num_instances
    pos = torch.tensor(
        [[draw.object_x, draw.object_y, spec.spawn_z]], dtype=torch.float32, device=device
    ).repeat(count, 1) + scene.env_origins.to(device=device, dtype=torch.float32)
    quat = torch.tensor(
        [randomize.quat_from_rpy_xyzw(0.0, 0.0, draw.object_yaw)],
        dtype=torch.float32,
        device=device,
    ).repeat(count, 1)

    obj.write_root_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1))
    obj.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((count, 6), dtype=torch.float32, device=device)
    )


def write_object_color(scene: InteractiveScene, draw: EpisodeDraw) -> None:
    """Recolour the object's preview surface.

    The shape spawner puts the visual material under ``<object>/geometry/material``
    and the shader that actually carries ``inputs:diffuseColor`` one level below
    that, so the attribute path is fixed by the spawner's layout, not chosen here.
    """
    color = Gf.Vec3f(*draw.object_color)
    for prim_path in sim_utils.find_matching_prim_paths(scene["object"].cfg.prim_path):
        sim_utils.change_prim_property(
            prop_path=f"{prim_path}/geometry/material/Shader.inputs:diffuseColor",
            value=color,
            type_to_create_if_not_exist=Sdf.ValueTypeNames.Color3f,
        )


def write_object_friction(scene: InteractiveScene, draw: EpisodeDraw) -> None:
    """Set the object's surface friction from the draw.

    PhysX exposes shape materials as ``(static friction, dynamic friction,
    restitution)`` rows on the rigid-body view, indexed
    ``(num_instances, num_shapes, 3)``; restitution is left as spawned. This is
    surface friction, distinct from the joint friction in :mod:`manus.specs`.
    The buffer is CPU-side, so this is a reset-time write, never a per-step one.
    """
    view = scene["object"].root_view
    materials = wp.to_torch(view.get_material_properties())
    materials[..., 0] = draw.object_static_friction
    materials[..., 1] = draw.object_dynamic_friction
    env_ids = torch.arange(materials.shape[0], dtype=torch.int32, device=materials.device)
    view.set_material_properties(
        wp.from_torch(materials, dtype=wp.float32), wp.from_torch(env_ids, dtype=wp.int32)
    )


def write_ground_albedo(scene: InteractiveScene, draw: EpisodeDraw) -> None:
    """Grey the ground plane to the drawn albedo.

    The stock grid asset exposes its base colour as the MDL input
    ``diffuse_tint`` on ``Looks/theGrid/Shader`` — the same attribute
    ``GroundPlaneCfg.color`` writes at spawn time.
    """
    grey = Gf.Vec3f(draw.ground_albedo, draw.ground_albedo, draw.ground_albedo)
    sim_utils.change_prim_property(
        prop_path=f"{scene.cfg.ground.prim_path}/Looks/theGrid/Shader.inputs:diffuse_tint",
        value=grey,
        type_to_create_if_not_exist=Sdf.ValueTypeNames.Color3f,
    )


def write_light_state(scene: InteractiveScene, draw: EpisodeDraw) -> None:
    """Set both light intensities and swing the distant light to the drawn sun angle."""
    sim_utils.change_prim_property(
        prop_path=f"{scene.cfg.dome_light.prim_path}.inputs:intensity",
        value=draw.dome_intensity,
    )
    sim_utils.change_prim_property(
        prop_path=f"{scene.cfg.distant_light.prim_path}.inputs:intensity",
        value=draw.distant_intensity,
    )

    view = scene["distant_light"]
    # set_local_poses' docstring claims (w, x, y, z), but the USD implementation
    # feeds the rows straight into Vt.QuatdArray.FromNumpy, which reads
    # (x, y, z, real) -- verified empirically (a wxyz row comes back with
    # real part 0). Pass xyzw, matching what the code actually consumes.
    quat_xyzw = sun_orientation_xyzw(draw.distant_azimuth, draw.distant_elevation)
    view.set_local_poses(
        orientations=torch.tensor([quat_xyzw], dtype=torch.float32).repeat(view.count, 1)
    )


def write_wrist_camera_jitter(scene: InteractiveScene, draw: EpisodeDraw) -> None:
    """Perturb the wrist camera's mounting pose by the drawn offset.

    Isaac Lab's :class:`Camera` has no local-pose setter, and its cached
    ``data.pos_w`` is stale unless ``update_latest_camera_pose`` is on, so the
    jitter goes straight through the camera's frame view. That view's poses are
    parent-relative — the gripper-link frame the camera is mounted in. Its
    ``set_local_poses`` docstring claims scalar-first (w, x, y, z) quaternions,
    but the USD implementation hands the rows to ``Vt.QuatdArray.FromNumpy``,
    which reads (x, y, z, real) — verified empirically. Pass (x, y, z, w).

    The jitter composes onto the *configured* mount rather than the current
    pose, so repeated episodes cannot accumulate drift.
    """
    cam = scene["wrist_cam"]
    offset = cam.cfg.offset
    # Camera.__init__ converts the cfg rotation into the OpenGL convention
    # before spawning; the USD prim therefore carries the converted value, and
    # composing onto anything else would double-apply the convention.
    base_rot = convert_camera_frame_orientation_convention(
        torch.tensor([offset.rot], dtype=torch.float32),
        origin=offset.convention,
        target="opengl",
    )[0]
    quat_xyzw = randomize.quat_mul_xyzw(tuple(base_rot.tolist()), draw.cam_dquat_xyzw)
    pos = [base + delta for base, delta in zip(offset.pos, draw.cam_dpos, strict=True)]

    # Private view: the plan's grounded escape hatch, since CameraCfg.OffsetCfg
    # is read once at spawn and cannot be re-applied on a live sensor.
    view = cam._view
    view.set_local_poses(
        translations=torch.tensor([pos], dtype=torch.float32).repeat(view.count, 1),
        orientations=torch.tensor([quat_xyzw], dtype=torch.float32).repeat(view.count, 1),
    )
