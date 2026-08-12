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

from manus.objects import (
    CLOSE_RAMP_MAX_STEPS,
    CLOSE_RAMP_REFERENCE_STEPS,
    CLOSE_TARGET_30MM_RAD,
    DEFAULT_OBJECTS,
    JAW_WIDTH_PER_RAD,
    LIGHT_SQUEEZE_RAD,
    MEASURED_STALL_30MM_RAD,
    MIN_SQUEEZE_RAD,
    OBJECTS,
    REFERENCE_MASS_KG,
    REFERENCE_WIDTH_M,
    SQUEEZE_RAD,
    ObjectSpec,
    close_ramp_for_mass,
    close_target_for_width,
    contact_angle_for_width,
)
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
from manus.specs import JOINT_LIMITS

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

SAMPLE_ATTEMPTS = range(500)
"""Attempt indices swept by the statistical property tests."""


# --- Object catalogue --------------------------------------------------------


def test_catalogue_keys_match_spec_names():
    assert all(key == spec.name for key, spec in OBJECTS.items())
    assert set(OBJECTS) == {
        "cube_3cm",
        "cylinder_3cm",
        "die_16mm",
        "domino_20x40",
        "puck_d40x10",
        "puck_d40x20",
        "pingpong_40mm",
        "duplo_32x64",
    }


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_spec_rests_on_the_ground_plane(spec):
    # spawn_z is the body origin height at rest, i.e. half the object's height.
    assert spec.spawn_z == pytest.approx(0.5 * spec.extent_z)


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_spec_grasp_width_matches_geometry(spec):
    width = 2 * spec.half_extents[0] if spec.shape == "cuboid" else 2 * spec.radius
    assert spec.grasp_width_m == pytest.approx(width)


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_spec_is_physically_plausible(spec):
    assert 0.0 < spec.mass_kg < 0.5
    assert 0.0 < spec.grasp_width_m < 0.05
    # Densities: the ping-pong ball is hollow (80 kg/m^3) and the cube is a
    # dense resin block (2200). Anything outside that spread is a typo.
    volume = {
        "cuboid": lambda s: 8 * math.prod(s.half_extents),
        "cylinder": lambda s: math.pi * s.radius**2 * s.height,
        "sphere": lambda s: 4 / 3 * math.pi * s.radius**3,
    }[spec.shape](spec)
    assert 50.0 < spec.mass_kg / volume < 2500.0


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_declared_symmetry_matches_the_geometry(spec):
    assert spec.yaw_symmetry == spec.geometric_yaw_symmetry


def test_the_catalogue_covers_every_symmetry_class():
    """All three branches of the expert's yaw planner have an object exercising them."""
    assert {spec.yaw_symmetry for spec in OBJECTS.values()} == {"quarter", "half", "free"}


def test_a_rectangular_cuboid_cannot_be_declared_square():
    with pytest.raises(ValueError, match="declared yaw_symmetry 'quarter'"):
        ObjectSpec(
            name="bad", shape="cuboid", half_extents=(0.01, 0.02, 0.01), mass_kg=0.1,
            grasp_width_m=0.02, spawn_z=0.01, close_target_rad=0.0, yaw_symmetry="quarter",
        )


def test_a_grasp_width_off_the_local_x_axis_is_rejected():
    """The jaws close along local x; declaring the long side would grasp the wrong way."""
    with pytest.raises(ValueError, match="not the local-x extent"):
        ObjectSpec(
            name="bad", shape="cuboid", half_extents=(0.01, 0.02, 0.01), mass_kg=0.1,
            grasp_width_m=0.04, spawn_z=0.01, close_target_rad=0.0, yaw_symmetry="half",
        )


def test_cuboid_without_half_extents_is_rejected():
    with pytest.raises(ValueError, match="needs half_extents"):
        ObjectSpec(
            name="bad", shape="cuboid", mass_kg=0.1, grasp_width_m=0.03,
            spawn_z=0.015, close_target_rad=0.0, yaw_symmetry="quarter",
        )


def test_cylinder_without_radius_is_rejected():
    with pytest.raises(ValueError, match="needs radius and height"):
        ObjectSpec(
            name="bad", shape="cylinder", mass_kg=0.1, grasp_width_m=0.03,
            spawn_z=0.03, close_target_rad=0.0, yaw_symmetry="free", height=0.06,
        )


def test_sphere_without_radius_is_rejected():
    with pytest.raises(ValueError, match="a sphere needs radius"):
        ObjectSpec(
            name="bad", shape="sphere", mass_kg=0.1, grasp_width_m=0.03,
            spawn_z=0.015, close_target_rad=0.0, yaw_symmetry="free",
        )


def test_unknown_shape_is_rejected():
    with pytest.raises(ValueError, match="unknown shape"):
        ObjectSpec(
            name="bad", shape="pyramid", mass_kg=0.1, grasp_width_m=0.03,
            spawn_z=0.015, close_target_rad=0.0, yaw_symmetry="quarter",
        )


# --- Close targets -----------------------------------------------------------


def test_the_formula_reproduces_the_tuned_thirty_millimetre_target():
    """The anchor: the one target that was tuned in sim has to survive the formula."""
    assert close_target_for_width(REFERENCE_WIDTH_M) == pytest.approx(CLOSE_TARGET_30MM_RAD)
    assert MEASURED_STALL_30MM_RAD - CLOSE_TARGET_30MM_RAD == pytest.approx(SQUEEZE_RAD)


def test_close_targets_are_pinned():
    """Hard-coded: a formula change that moves a grasp has to be seen, not inferred."""
    assert {name: round(spec.close_target_rad, 4) for name, spec in OBJECTS.items()} == {
        "cube_3cm": 0.05,
        "cylinder_3cm": 0.05,
        "die_16mm": -0.1036,  # LIGHT_SQUEEZE_RAD, not the full squeeze
        "domino_20x40": -0.0876,
        # The two pucks are the same 40 mm disc, so the same target: the width
        # is what the jaws span, and the respec changed only the rim's height.
        "puck_d40x10": 0.1876,
        "puck_d40x20": 0.1876,
        "pingpong_40mm": 0.1876,
        "duplo_32x64": 0.0748,
    }


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_every_close_target_comes_from_the_formula(spec):
    assert spec.close_target_rad == pytest.approx(
        close_target_for_width(spec.grasp_width_m, spec.squeeze_rad)
    )


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_the_contact_angle_is_the_close_target_plus_the_squeeze(spec):
    """The two halves of the CLOSE geometry are one identity, per object."""
    assert spec.contact_angle_rad == pytest.approx(contact_angle_for_width(spec.grasp_width_m))
    assert spec.close_target_rad == pytest.approx(spec.contact_angle_rad - spec.squeeze_rad)


def test_the_die_is_the_one_object_squeezed_less_than_the_tuned_amount():
    """Pinned: the light squeeze is a deliberate, single exception."""
    lighter = [name for name, spec in OBJECTS.items() if spec.squeeze_rad < SQUEEZE_RAD - 1e-9]
    assert lighter == ["die_16mm"]
    assert OBJECTS["die_16mm"].squeeze_rad == pytest.approx(LIGHT_SQUEEZE_RAD)


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_every_squeeze_still_asks_the_servo_for_half_its_effort(spec):
    """The floor a lighter squeeze must not go through, at the catalogue level."""
    assert spec.squeeze_rad >= MIN_SQUEEZE_RAD


def test_the_die_keeps_clear_of_the_jaws_hard_stop():
    """Why the die's squeeze was cut: the stop is where the grip goes slack.

    The full squeeze put its target 0.032 rad off the -0.1745 rad stop, closer
    than any other object by a factor of four; the light one doubles that.
    """
    die = OBJECTS["die_16mm"]
    stop = JOINT_LIMITS["gripper"][0]
    assert die.close_target_rad - stop == pytest.approx(0.071, abs=1e-3)
    assert contact_angle_for_width(die.grasp_width_m) - SQUEEZE_RAD - stop < 0.035


# --- Close ramp ----------------------------------------------------------------


def test_the_close_ramp_rule_hits_both_measured_anchors():
    """The two numbers the rule exists to interpolate, and it must reproduce them.

    60 g in 60 steps is the Step 8 gate's anchor (200/200 attempts); 5 g in 150
    is what the die needed once 60 punted it.
    """
    assert close_ramp_for_mass(REFERENCE_MASS_KG) == CLOSE_RAMP_REFERENCE_STEPS
    assert close_ramp_for_mass(0.005) == 150


def test_the_close_ramp_is_monotone_and_bounded_in_mass():
    masses = [0.0005, 0.001, 0.0027, 0.005, 0.01, 0.015, 0.02, 0.06, 0.08, 0.5]
    ramps = [close_ramp_for_mass(mass) for mass in masses]
    assert ramps == sorted(ramps, reverse=True)
    assert all(CLOSE_RAMP_REFERENCE_STEPS <= ramp <= CLOSE_RAMP_MAX_STEPS for ramp in ramps)


def test_a_massless_object_is_rejected_rather_than_ramped_forever():
    with pytest.raises(ValueError, match="mass must be positive"):
        close_ramp_for_mass(0.0)


def test_close_ramps_are_pinned():
    """Hard-coded, like the close targets: a rule change has to be seen."""
    assert {name: spec.close_ramp_steps for name, spec in OBJECTS.items()} == {
        "cube_3cm": 60,  # the gate's anchor -- must not move
        "cylinder_3cm": 60,
        "die_16mm": 150,
        "domino_20x40": 104,
        "puck_d40x10": 120,
        "puck_d40x20": 85,  # twice the puck's mass, so the jaws may come in faster
        "pingpong_40mm": 150,
        "duplo_32x64": 104,
    }


def test_a_spec_can_override_the_mass_rule():
    spec = ObjectSpec(
        name="probe", shape="cuboid", half_extents=(0.015, 0.015, 0.015), mass_kg=0.06,
        grasp_width_m=0.03, spawn_z=0.015, close_target_rad=0.05, yaw_symmetry="quarter",
        close_ramp=33,
    )
    assert spec.close_ramp_steps == 33


def test_a_nonsense_override_is_rejected():
    common = dict(
        shape="cuboid", half_extents=(0.015, 0.015, 0.015), mass_kg=0.06,
        grasp_width_m=0.03, spawn_z=0.015, close_target_rad=0.05, yaw_symmetry="quarter",
    )
    with pytest.raises(ValueError, match="close_ramp must be at least one step"):
        ObjectSpec(name="bad", close_ramp=0, **common)
    with pytest.raises(ValueError, match="tip_clearance_m cannot be negative"):
        ObjectSpec(name="bad", tip_clearance_m=-0.001, **common)


# --- Experimental objects --------------------------------------------------------


def test_the_default_sweep_leaves_the_experimental_objects_out_but_keeps_them_runnable():
    """The puck fails by geometry (see manus.expert), so it must not ride into a dataset.

    It stays in OBJECTS -- spawnable, plannable, runnable by name -- and only
    the default "every object" list drops it.
    """
    assert set(DEFAULT_OBJECTS) < set(OBJECTS)
    assert [name for name in OBJECTS if name not in DEFAULT_OBJECTS] == [
        "cylinder_3cm",  # CLOSE-time seating-shove failure; side approach itself is proven
        "puck_d40x10",  # pads can only reach a ~3 mm sliver of rim
        "puck_d40x20",  # CLOSE-time seating-shove failure (same mechanism as the cylinder)
    ]
    assert DEFAULT_OBJECTS == tuple(
        name for name, spec in OBJECTS.items() if not spec.experimental
    )
    assert "cube_3cm" == DEFAULT_OBJECTS[0]


def test_the_close_target_tracks_the_width_at_the_measured_rate():
    """Every extra millimetre of object opens the jaws by 1/JAW_WIDTH_PER_RAD."""
    step = close_target_for_width(0.031) - close_target_for_width(0.030)
    assert step == pytest.approx(0.001 / JAW_WIDTH_PER_RAD)


@pytest.mark.parametrize("spec", OBJECTS.values(), ids=list(OBJECTS))
def test_every_close_target_is_inside_the_jaw_travel(spec):
    lower, upper = JOINT_LIMITS["gripper"]
    assert lower < spec.close_target_rad < upper


def test_a_width_the_jaws_cannot_squeeze_is_refused():
    """A target the articulation would clamp is a silently weak grasp: refuse it."""
    with pytest.raises(ValueError, match="outside the gripper's travel"):
        close_target_for_width(0.005)


def test_every_spec_builds_its_isaac_spawner():
    """The module's one Isaac API call, exercised without a running app.

    ``isaaclab.sim``'s spawner configs are plain dataclasses and import on their
    own, so the cfg each object hands the simulator can be built and read back
    here. That is as far as the sim-free side can check a spawner — the USD prim
    it produces still needs a live app — but it is the difference between
    "SphereCfg takes these fields" as a claim and as a fact.
    """
    sim_utils = pytest.importorskip("isaaclab.sim", reason="Isaac Lab not installed")
    expected = {
        "cuboid": sim_utils.CuboidCfg,
        "cylinder": sim_utils.CylinderCfg,
        "sphere": sim_utils.SphereCfg,
    }
    for spec in OBJECTS.values():
        cfg = spec.make_spawn_cfg()
        assert isinstance(cfg, expected[spec.shape]), spec.name
        assert cfg.mass_props.mass == spec.mass_kg
        if spec.shape == "cuboid":
            assert cfg.size == pytest.approx(tuple(2 * h for h in spec.half_extents))
        else:
            assert cfg.radius == spec.radius
        if spec.shape == "cylinder":
            assert cfg.height == spec.height
            assert cfg.axis == "Z"  # the resting axis every catalogue cylinder assumes


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


# --- Per-mode placement regions -----------------------------------------------------
#
# A side-grasped object is placed on a different annulus, so the draw has to
# know which object it is drawing for. What must *not* change is everything
# else about the draw -- and, for a top-down object, not even the placement.

GOLDEN_TOP_DOWN_DRAW = {
    "object_x": 0.1077012236965687,
    "object_y": -0.11191569290331138,
    "object_yaw": -0.9431381922702697,
    "object_color": [0.15418878646169656, 0.5613792582366869, 0.4586327750467186],
    "object_static_friction": 0.8235333022006006,
    "object_dynamic_friction": 0.8235333022006006,
    "dome_intensity": 447.7512484151919,
    "distant_intensity": 700.863433358707,
    "distant_azimuth": 5.87677178442419,
    "distant_elevation": 0.5343219004161894,
    "cam_dpos": [0.001622654240251633, 0.0017808265135320122, -0.0022005880641807773],
    "cam_dquat_xyzw": [
        0.003963767333973988,
        0.011387546686311784,
        -0.0069675547703871065,
        0.9999030280529762,
    ],
    "ground_albedo": 0.22187591912019738,
}
"""``draw_episode("grasp_cube_v2", 7)`` recorded before ``draw_episode`` learned
about grasp modes.

Every digit of it, because a draw *is* the replay contract: a dataset records
the values it was generated with and re-derives nothing, so a draw that moved
would silently make every committed episode unreproducible.
"""


def test_a_top_down_draw_is_byte_identical_to_the_one_before_grasp_modes():
    from manus.objects import OBJECTS
    from manus.randomize import draw_episode

    assert draw_episode("grasp_cube_v2", 7).to_dict() == GOLDEN_TOP_DOWN_DRAW
    # ... naming a top-down object, or none at all, is the same call.
    assert draw_episode("grasp_cube_v2", 7, OBJECTS["cube_3cm"]).to_dict() == (
        GOLDEN_TOP_DOWN_DRAW
    )
    assert draw_episode("grasp_cube_v2", 7, OBJECTS["puck_d40x20"]).to_dict() == (
        GOLDEN_TOP_DOWN_DRAW
    )


def test_a_side_object_is_drawn_from_the_side_region_and_nothing_else_moves():
    """The placement moves to the other annulus; the rest of the draw does not.

    Same seed, same rng, same number of draws -- only the region the first two
    numbers are mapped through. That is what keeps a side-mode dataset
    comparable with a top-down one: the lighting, friction, colour and camera
    jitter of attempt *n* are the same scene either way.
    """
    from manus.kinematics import GRASP_REGION, SIDE_GRASP_REGION
    from manus.objects import OBJECTS
    from manus.randomize import draw_episode, placement_region

    top = draw_episode("grasp_cube_v2", 7).to_dict()
    side = draw_episode("grasp_cube_v2", 7, OBJECTS["cylinder_3cm"]).to_dict()
    moved = {key for key in top if top[key] != side[key]}
    assert moved == {"object_x", "object_y"}
    assert SIDE_GRASP_REGION.contains(side["object_x"], side["object_y"])
    assert not GRASP_REGION.contains(side["object_x"], side["object_y"])
    assert placement_region(OBJECTS["cylinder_3cm"]) is SIDE_GRASP_REGION
    assert placement_region(OBJECTS["cube_3cm"]) is GRASP_REGION
    assert placement_region() is GRASP_REGION


@pytest.mark.parametrize("name", list(OBJECTS))
def test_every_object_is_drawn_inside_its_own_region(name):
    from manus.randomize import draw_episode, placement_region

    spec = OBJECTS[name]
    region = placement_region(spec)
    for attempt in range(200):
        draw = draw_episode("region_sweep", attempt, spec)
        assert region.contains(draw.object_x, draw.object_y), (name, attempt)


def test_the_two_regions_are_reachable_through_the_mode_table():
    from manus.kinematics import GRASP_REGION, GRASP_REGIONS, SIDE_GRASP_REGION

    assert GRASP_REGIONS == {"top": GRASP_REGION, "side": SIDE_GRASP_REGION}
    assert {spec.grasp_mode for spec in OBJECTS.values()} <= set(GRASP_REGIONS)
