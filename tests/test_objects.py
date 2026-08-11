"""Tests for the sim-free half of the grasp task setup.

Covers the object catalogue (:mod:`manus.objects`) and the per-episode
randomization (:mod:`manus.randomize`), including the quaternion helpers
:mod:`manus.task_scene` uses to drive USD writes — those live in
``manus.randomize`` precisely so they can be tested without Isaac Sim.

Lengths are metres, angles radians.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from manus.objects import OBJECTS, ObjectSpec
from manus.randomize import (
    BASE_KEEPOUT_ABS_Y,
    BASE_KEEPOUT_X,
    CAM_DPOS_M,
    CAM_DROT_DEG,
    DOME_INTENSITY_RANGE,
    FRICTION_RANGE,
    GROUND_ALBEDO_RANGE,
    PAN_AXIS_XY,
    REGION_AZ_DEG,
    REGION_R,
    EpisodeDraw,
    draw_episode,
    in_base_keepout,
    in_grasp_region,
    quat_from_rpy_xyzw,
    quat_from_z_axis_xyzw,
    quat_mul_xyzw,
    stable_hash64,
    xyzw_to_wxyz,
)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

SAMPLE_ATTEMPTS = range(500)
"""Attempt indices swept by the statistical property tests."""


# --- Object catalogue --------------------------------------------------------


def test_catalogue_keys_match_spec_names():
    assert all(key == spec.name for key, spec in OBJECTS.items())
    assert set(OBJECTS) == {"cube_3cm", "cylinder_3cm"}


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_spec_rests_on_the_ground_plane(spec):
    # spawn_z is the body origin height at rest, i.e. half the object's height.
    half_height = spec.half_extents[2] if spec.shape == "cuboid" else spec.height / 2.0
    assert spec.spawn_z == pytest.approx(half_height)


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_spec_grasp_width_matches_geometry(spec):
    width = 2 * spec.half_extents[0] if spec.shape == "cuboid" else 2 * spec.radius
    assert spec.grasp_width_m == pytest.approx(width)


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_spec_is_physically_plausible(spec):
    assert 0.0 < spec.mass_kg < 0.5
    assert 0.0 < spec.grasp_width_m < 0.05


def test_cuboid_without_half_extents_is_rejected():
    with pytest.raises(ValueError, match="needs half_extents"):
        ObjectSpec(
            name="bad", shape="cuboid", mass_kg=0.1, grasp_width_m=0.03,
            spawn_z=0.015, close_target_rad=0.0,
        )


def test_cylinder_without_radius_is_rejected():
    with pytest.raises(ValueError, match="needs radius and height"):
        ObjectSpec(
            name="bad", shape="cylinder", mass_kg=0.1, grasp_width_m=0.03,
            spawn_z=0.03, close_target_rad=0.0, height=0.06,
        )


def test_unknown_shape_is_rejected():
    with pytest.raises(ValueError, match="unknown shape"):
        ObjectSpec(
            name="bad", shape="sphere", mass_kg=0.1, grasp_width_m=0.03,
            spawn_z=0.015, close_target_rad=0.0,
        )


def test_object_and_randomize_modules_import_without_isaac():
    """The dataset side of the pipeline runs with no simulator.

    Run in a subprocess so the assertion cannot be corrupted by whatever this
    session already imported. ``make_spawn_cfg`` is the only isaaclab consumer
    and it imports lazily, inside the call.
    """
    probe = (
        "import manus.objects, manus.randomize, sys;"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in {'isaacsim', 'isaaclab'});"
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=SRC_DIR, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --- Seeding -----------------------------------------------------------------


def test_stable_hash64_is_pinned():
    # Hard-coded: a change here silently invalidates every recorded dataset.
    assert stable_hash64("grasp_cube_v1", 0) == 4478714552628142013
    assert stable_hash64("grasp_cube_v1", 1) == 10179360694728744792


def test_stable_hash64_separates_datasets_and_attempts():
    assert stable_hash64("a", 0) != stable_hash64("b", 0)
    assert stable_hash64("a", 0) != stable_hash64("a", 1)
    # The separator must make the name/index split unambiguous.
    assert stable_hash64("a", 11) != stable_hash64("a1", 1)


# --- Draws -------------------------------------------------------------------


def test_draw_is_deterministic():
    assert draw_episode("grasp_cube_v1", 42) == draw_episode("grasp_cube_v1", 42)


def test_draws_differ_across_attempts_and_datasets():
    assert draw_episode("grasp_cube_v1", 42) != draw_episode("grasp_cube_v1", 43)
    assert draw_episode("grasp_cube_v1", 42) != draw_episode("grasp_cube_dev", 42)


def test_draw_round_trips_through_json():
    draw = draw_episode("grasp_cube_dev", 7)
    assert EpisodeDraw.from_dict(json.loads(json.dumps(draw.to_dict()))) == draw


def test_draw_dict_is_json_scalars_only():
    values = draw_episode("grasp_cube_dev", 7).to_dict().values()
    flat = [item for value in values for item in (value if isinstance(value, list) else [value])]
    assert all(type(item) is float for item in flat)


def test_from_dict_rejects_an_incomplete_draw():
    partial = draw_episode("grasp_cube_dev", 7).to_dict()
    del partial["ground_albedo"]
    with pytest.raises(KeyError):
        EpisodeDraw.from_dict(partial)


def test_placements_are_in_region_and_outside_the_keepout():
    for attempt in SAMPLE_ATTEMPTS:
        draw = draw_episode("grasp_cube_v1", attempt)
        assert in_grasp_region(draw.object_x, draw.object_y), attempt
        assert not in_base_keepout(draw.object_x, draw.object_y), attempt


def test_placements_span_the_region():
    """Guards against a sampler that collapses onto one radius or azimuth."""
    draws = [draw_episode("grasp_cube_v1", i) for i in SAMPLE_ATTEMPTS]
    radii = [math.hypot(d.object_x - PAN_AXIS_XY[0], d.object_y - PAN_AXIS_XY[1]) for d in draws]
    azimuths = [
        math.degrees(math.atan2(d.object_y - PAN_AXIS_XY[1], d.object_x - PAN_AXIS_XY[0]))
        for d in draws
    ]
    assert min(radii) < REGION_R[0] + 0.02 and max(radii) > REGION_R[1] - 0.02
    assert min(azimuths) < -REGION_AZ_DEG + 15 and max(azimuths) > REGION_AZ_DEG - 15


def test_draw_stays_inside_its_declared_ranges():
    for attempt in SAMPLE_ATTEMPTS:
        draw = draw_episode("grasp_cube_v1", attempt)
        assert DOME_INTENSITY_RANGE[0] <= draw.dome_intensity <= DOME_INTENSITY_RANGE[1]
        assert GROUND_ALBEDO_RANGE[0] <= draw.ground_albedo <= GROUND_ALBEDO_RANGE[1]
        assert FRICTION_RANGE[0] <= draw.object_static_friction <= FRICTION_RANGE[1]
        assert FRICTION_RANGE[0] <= draw.object_dynamic_friction <= FRICTION_RANGE[1]
        # PhysX requires dynamic friction not to exceed static friction.
        assert draw.object_dynamic_friction <= draw.object_static_friction
        assert -math.pi <= draw.object_yaw <= math.pi
        assert all(abs(d) <= CAM_DPOS_M for d in draw.cam_dpos)
        assert all(0.0 <= c <= 1.0 for c in draw.object_color)


def test_camera_jitter_is_a_small_unit_rotation():
    for attempt in SAMPLE_ATTEMPTS:
        quat = draw_episode("grasp_cube_v1", attempt).cam_dquat_xyzw
        assert math.sqrt(sum(c * c for c in quat)) == pytest.approx(1.0)
        # Three axis rotations of at most CAM_DROT_DEG compose to at most 3x that.
        angle_deg = math.degrees(2 * math.acos(min(1.0, abs(quat[3]))))
        assert angle_deg <= 3 * CAM_DROT_DEG


# --- Region predicates -------------------------------------------------------


def test_grasp_region_excludes_points_outside_the_annulus():
    pan_x, pan_y = PAN_AXIS_XY
    assert not in_grasp_region(pan_x + REGION_R[0] - 0.01, pan_y)
    assert not in_grasp_region(pan_x + REGION_R[1] + 0.01, pan_y)
    assert in_grasp_region(pan_x + sum(REGION_R) / 2, pan_y)


def test_grasp_region_excludes_points_behind_the_azimuth_limit():
    pan_x, pan_y = PAN_AXIS_XY
    radius = sum(REGION_R) / 2
    behind = math.radians(REGION_AZ_DEG + 10)
    assert not in_grasp_region(pan_x + radius * math.cos(behind), pan_y + radius * math.sin(behind))


def test_base_keepout_covers_its_rectangle():
    assert in_base_keepout(0.0, 0.0)
    assert in_base_keepout(BASE_KEEPOUT_X[0], BASE_KEEPOUT_ABS_Y)
    assert not in_base_keepout(BASE_KEEPOUT_X[1] + 0.01, 0.0)
    assert not in_base_keepout(0.0, BASE_KEEPOUT_ABS_Y + 0.01)


# --- Quaternion helpers ------------------------------------------------------


def _rotate(quat, vector):
    """Rotate `vector` by the (x, y, z, w) quaternion, via q * v * q^-1."""
    x, y, z, w = quat
    vx, vy, vz = vector
    rotated = quat_mul_xyzw(quat_mul_xyzw(quat, (vx, vy, vz, 0.0)), (-x, -y, -z, w))
    return rotated[:3]


def test_xyzw_to_wxyz_moves_the_scalar_first():
    assert xyzw_to_wxyz((1.0, 2.0, 3.0, 4.0)) == (4.0, 1.0, 2.0, 3.0)


def test_quat_from_rpy_identity_and_quarter_turn():
    assert quat_from_rpy_xyzw(0.0, 0.0, 0.0) == pytest.approx((0.0, 0.0, 0.0, 1.0))
    root_half = math.sqrt(0.5)
    quarter_turn = quat_from_rpy_xyzw(0.0, 0.0, math.pi / 2)
    assert quarter_turn == pytest.approx((0.0, 0.0, root_half, root_half))
    # A +90 deg yaw takes +x to +y.
    assert _rotate(quarter_turn, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)


def test_quat_mul_by_identity_is_a_no_op():
    quat = quat_from_rpy_xyzw(0.3, -0.2, 1.1)
    assert quat_mul_xyzw(quat, (0.0, 0.0, 0.0, 1.0)) == pytest.approx(quat)
    assert quat_mul_xyzw((0.0, 0.0, 0.0, 1.0), quat) == pytest.approx(quat)


def test_quat_mul_composes_yaw_additively():
    combined = quat_mul_xyzw(quat_from_rpy_xyzw(0, 0, 0.4), quat_from_rpy_xyzw(0, 0, 0.5))
    assert combined == pytest.approx(quat_from_rpy_xyzw(0, 0, 0.9))


@pytest.mark.parametrize(
    "direction",
    [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (1, 2, 3), (-0.5, 0.25, -0.8)],
)
def test_quat_from_z_axis_maps_z_onto_the_direction(direction):
    norm = math.sqrt(sum(c * c for c in direction))
    unit = tuple(c / norm for c in direction)
    assert _rotate(quat_from_z_axis_xyzw(direction), (0.0, 0.0, 1.0)) == pytest.approx(unit, abs=1e-9)


def test_quat_from_z_axis_rejects_the_zero_vector():
    with pytest.raises(ValueError, match="non-zero"):
        quat_from_z_axis_xyzw((0.0, 0.0, 0.0))
