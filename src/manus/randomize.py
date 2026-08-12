"""Seeded per-episode domain randomization for the grasping data factory.

Sim-free and deterministic: :func:`draw_episode` maps
``(dataset_name, attempt_index)`` to a fully specified :class:`EpisodeDraw` via
a stable SHA-256 seed, so any attempt can be re-derived from its manifest on
any machine. The *draw*, not the seed, is the replay contract — GPU PhysX is
not bit-reproducible across drivers, so every episode records the sampled
values themselves (:meth:`EpisodeDraw.to_dict`).

Units: metres, radians, kilograms. Colours are linear RGB in [0, 1]; light
intensities are USD ``inputs:intensity`` values.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

# The grasp region lives in manus.kinematics, next to the IK that has to cover
# it. These names are re-exported rather than merely used: the region is part of
# this module's surface -- callers sample placements from it and then test them.
from manus.kinematics import (
    BASE_KEEPOUT_ABS_Y,
    BASE_KEEPOUT_X,
    GRASP_REGION,
    GRASP_REGIONS,
    PAN_AXIS_XY,
    REGION_AZ_DEG,
    REGION_R,
    SIDE_GRASP_REGION,
    GraspRegion,
    in_base_keepout,
    in_grasp_region,
)
from manus.objects import ObjectSpec

__all__ = [
    "BASE_KEEPOUT_ABS_Y",
    "BASE_KEEPOUT_X",
    "GRASP_REGION",
    "PAN_AXIS_XY",
    "REGION_AZ_DEG",
    "REGION_R",
    "SIDE_GRASP_REGION",
    "EpisodeDraw",
    "draw_episode",
    "in_base_keepout",
    "in_grasp_region",
    "placement_region",
    "quat_from_rpy_xyzw",
    "quat_from_z_axis_xyzw",
    "quat_mul_xyzw",
    "stable_hash64",
    "xyzw_to_wxyz",
]

# --- Sampling ranges ---------------------------------------------------------
COLOR_CHANNEL_RANGE: tuple[float, float] = (0.05, 0.95)
"""Per-channel linear-RGB range for the object's diffuse colour."""

DOME_INTENSITY_RANGE: tuple[float, float] = (400.0, 1100.0)
"""Dome-light intensity range (the scene default is 750)."""

DISTANT_INTENSITY_RANGE: tuple[float, float] = (500.0, 2500.0)
"""Distant ("sun") light intensity range."""

DISTANT_AZ_RANGE_DEG: tuple[float, float] = (0.0, 360.0)
"""Distant-light azimuth range about world +x, in degrees."""

DISTANT_EL_RANGE_DEG: tuple[float, float] = (25.0, 80.0)
"""Distant-light elevation range above the horizon, in degrees.

Floored well above 0 so the key light never grazes the work surface and washes
the wrist view out with shadow.
"""

CAM_DPOS_M = 0.003
"""Wrist-camera mount jitter, per axis, in metres (uniform in +/- this)."""

CAM_DROT_DEG = 2.0
"""Wrist-camera mount jitter, per roll/pitch/yaw axis, in degrees."""

GROUND_ALBEDO_RANGE: tuple[float, float] = (0.15, 0.6)
"""Grey level of the ground plane, from dark asphalt to light desk."""

FRICTION_RANGE: tuple[float, float] = (0.8, 1.2)
"""Object surface friction range (static and dynamic)."""

_MAX_PLACEMENT_TRIES = 1000
"""Rejection-sampling budget for one object placement."""


# --- Pure quaternion helpers -------------------------------------------------
# These live here, not in manus.task_scene, because task_scene imports isaaclab
# at module scope and so cannot be exercised by the sim-free test suite. All
# quaternions are (x, y, z, w) -- the order Isaac Lab cfgs use -- except where
# a name says otherwise.


def quat_from_rpy_xyzw(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """Intrinsic Z-Y-X Euler angles (radians) to a unit quaternion (x, y, z, w)."""
    sr, cr = math.sin(0.5 * roll), math.cos(0.5 * roll)
    sp, cp = math.sin(0.5 * pitch), math.cos(0.5 * pitch)
    sy, cy = math.sin(0.5 * yaw), math.cos(0.5 * yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_mul_xyzw(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton product ``a * b`` of two (x, y, z, w) quaternions.

    Composes ``b`` in ``a``'s own frame: rotating by the result is rotating by
    ``a`` then by ``b`` expressed in the rotated frame.
    """
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_from_z_axis_xyzw(
    direction: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Shortest-arc rotation carrying world +Z onto ``direction`` as (x, y, z, w).

    ``direction`` need not be normalised. The rotation about the resulting axis
    is unspecified (any is valid for an axially symmetric target such as a
    distant light), which is what makes the shortest arc the natural choice.
    """
    x, y, z = direction
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    x, y, z = x / norm, y / norm, z / norm
    # Axis = (0, 0, 1) x direction; w = 1 + cos(angle) before normalisation.
    w = 1.0 + z
    if w < 1e-12:  # antiparallel: any axis in the xy-plane gives the 180 deg flip
        return (1.0, 0.0, 0.0, 0.0)
    half = math.sqrt(2.0 * w)
    return (-y / half, x / half, 0.0, 0.5 * half)


def xyzw_to_wxyz(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Reorder an (x, y, z, w) quaternion to USD's scalar-first (w, x, y, z).

    Isaac Lab cfgs and this module speak (x, y, z, w); USD xform ops and the
    frame views that write them (``FrameView.set_local_poses``) speak
    (w, x, y, z). Getting this backwards silently mis-aims the wrist camera.
    """
    x, y, z, w = quat
    return (w, x, y, z)


# --- The draw ----------------------------------------------------------------

_TUPLE_FIELDS = frozenset({"object_color", "cam_dpos", "cam_dquat_xyzw"})
"""Fields that serialise as JSON arrays rather than bare numbers."""


@dataclass(frozen=True)
class EpisodeDraw:
    """Everything randomized for one attempt, fully serializable.

    Attributes:
        object_x: Object position along world x, in metres.
        object_y: Object position along world y, in metres.
        object_yaw: Object yaw about world +z, in radians.
        object_color: Linear-RGB diffuse colour of the object.
        object_static_friction: Object surface static friction coefficient.
        object_dynamic_friction: Object surface dynamic friction coefficient
            (never above the static one).
        dome_intensity: Dome-light ``inputs:intensity``.
        distant_intensity: Distant-light ``inputs:intensity``.
        distant_azimuth: Distant-light azimuth about world +x, in radians.
        distant_elevation: Distant-light elevation above the horizon, in radians.
        cam_dpos: Wrist-camera mount offset (x, y, z) in the gripper-link
            frame, in metres.
        cam_dquat_xyzw: Wrist-camera mount rotation jitter, applied in the
            camera's own frame, as (x, y, z, w).
        ground_albedo: Grey level of the ground plane in [0, 1].
    """

    object_x: float
    object_y: float
    object_yaw: float
    object_color: tuple[float, float, float]
    object_static_friction: float
    object_dynamic_friction: float
    dome_intensity: float
    distant_intensity: float
    distant_azimuth: float
    distant_elevation: float
    cam_dpos: tuple[float, float, float]
    cam_dquat_xyzw: tuple[float, float, float, float]
    ground_albedo: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping: tuples become lists, scalars become plain floats."""
        return {
            field.name: (
                [float(v) for v in getattr(self, field.name)]
                if field.name in _TUPLE_FIELDS
                else float(getattr(self, field.name))
            )
            for field in fields(self)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EpisodeDraw:
        """Rebuild a draw from :meth:`to_dict` output. Raises KeyError if incomplete."""
        return cls(
            **{
                field.name: (
                    tuple(float(v) for v in data[field.name])
                    if field.name in _TUPLE_FIELDS
                    else float(data[field.name])
                )
                for field in fields(cls)
            }
        )


def stable_hash64(dataset_name: str, attempt_index: int) -> int:
    """Reproducible 64-bit seed for one attempt of one dataset.

    SHA-256 rather than :func:`hash`, which is salted per interpreter process
    (``PYTHONHASHSEED``) and would make a dataset unreproducible across runs.
    """
    digest = hashlib.sha256(f"{dataset_name}\x00{attempt_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def placement_region(spec: ObjectSpec | None = None) -> GraspRegion:
    """The patch of table `spec` should be placed on.

    :data:`~manus.kinematics.GRASP_REGION` for a top-down object and for
    ``None``, :data:`~manus.kinematics.SIDE_GRASP_REGION` for a side-grasped
    one. The two do not overlap -- a side grasp stands the whole hand out along
    the table, which is worth ~160 mm of reach -- so this is not a refinement of
    one region but a choice between two.
    """
    return GRASP_REGIONS["top" if spec is None else spec.grasp_mode]


def _sample_object_xy(
    rng: np.random.Generator, region: GraspRegion = GRASP_REGION
) -> tuple[float, float]:
    """Draw a placement uniformly by area over `region`, outside the keep-out.

    Radius is drawn as ``sqrt(U(r_min^2, r_max^2))`` so samples spread evenly
    over the annulus sector instead of piling up at its inner edge.

    Two rng draws per try whichever region is passed, and a second try is only
    spent when the keep-out rejects the first. The side region never reaches the
    keep-out at all, so a side placement always costs exactly two draws -- the
    same as the overwhelming majority of top-down ones -- and the rest of the
    episode's randomization lands on the same numbers either way.
    """
    r_min, r_max = region.radius
    az_max = math.radians(region.azimuth_max_deg)
    pan_x, pan_y = region.pan_axis_xy
    for _ in range(_MAX_PLACEMENT_TRIES):
        radius = math.sqrt(rng.uniform(r_min**2, r_max**2))
        azimuth = rng.uniform(-az_max, az_max)
        x = pan_x + radius * math.cos(azimuth)
        y = pan_y + radius * math.sin(azimuth)
        if not region.in_keepout(x, y):
            return float(x), float(y)
    raise RuntimeError(
        f"no placement outside the base keep-out in {_MAX_PLACEMENT_TRIES} tries; "
        "the keep-out rectangle probably swallowed the region"
    )


def draw_episode(
    dataset_name: str, attempt_index: int, spec: ObjectSpec | None = None
) -> EpisodeDraw:
    """Sample the full randomization for one attempt.

    Deterministic in its arguments: the same pair always yields an identical
    draw, and different attempt indices are independent (the seed is a hash,
    not a counter, so neighbouring attempts share no low-order structure).

    Args:
        dataset_name: Dataset the attempt belongs to, e.g. ``"grasp_cube_v1"``.
        attempt_index: Attempt counter within that dataset. Held-out evaluation
            placements live at ``>= 10_000_000`` by convention.
        spec: Object the attempt will grasp, which decides *which region* the
            placement is drawn from (:func:`placement_region`) -- a side grasp
            reaches a different annulus entirely. None means the top-down
            region, so every pre-existing call is unchanged, and the seed and
            the number of rng draws do not depend on this argument either way:
            the same ``(dataset_name, attempt_index)`` gives the same lighting,
            colour, friction and camera jitter whatever is being grasped.

    Returns:
        The sampled :class:`EpisodeDraw`.
    """
    rng = np.random.default_rng(stable_hash64(dataset_name, attempt_index))

    object_x, object_y = _sample_object_xy(rng, placement_region(spec))
    static_friction = float(rng.uniform(*FRICTION_RANGE))
    # PhysX expects dynamic <= static. Clamping rather than re-drawing keeps the
    # number of rng calls independent of the values drawn.
    dynamic_friction = min(static_friction, float(rng.uniform(*FRICTION_RANGE)))
    cam_drot = math.radians(CAM_DROT_DEG)
    cam_roll, cam_pitch, cam_yaw = rng.uniform(-cam_drot, cam_drot, size=3).tolist()

    return EpisodeDraw(
        object_x=object_x,
        object_y=object_y,
        object_yaw=float(rng.uniform(-math.pi, math.pi)),
        object_color=tuple(rng.uniform(*COLOR_CHANNEL_RANGE, size=3).tolist()),
        object_static_friction=static_friction,
        object_dynamic_friction=dynamic_friction,
        dome_intensity=float(rng.uniform(*DOME_INTENSITY_RANGE)),
        distant_intensity=float(rng.uniform(*DISTANT_INTENSITY_RANGE)),
        distant_azimuth=math.radians(float(rng.uniform(*DISTANT_AZ_RANGE_DEG))),
        distant_elevation=math.radians(float(rng.uniform(*DISTANT_EL_RANGE_DEG))),
        cam_dpos=tuple(rng.uniform(-CAM_DPOS_M, CAM_DPOS_M, size=3).tolist()),
        cam_dquat_xyzw=quat_from_rpy_xyzw(cam_roll, cam_pitch, cam_yaw),
        ground_albedo=float(rng.uniform(*GROUND_ALBEDO_RANGE)),
    )
