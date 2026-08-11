"""The eval client's sim-free maths: Wilson intervals, temporal ensembling, cells.

``scripts/eval_policy.py`` keeps every Isaac import inside a function so the
parts that decide what the eval *reports* can be tested without a simulator —
which matters, because those are the parts a wrong answer would be hardest to
notice in.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_policy as ev  # noqa: E402

Z95 = 1.959963984540054


# --- Wilson interval ----------------------------------------------------------
# Pinned against published values first, then against the mathematical property
# that defines the interval. The pins alone would only prove the formula has not
# changed; the root property proves it is the right formula.


@pytest.mark.parametrize(
    ("successes", "total", "lower", "upper"),
    [
        (0, 10, 0.000000000, 0.277532800),
        (8, 10, 0.490162472, 0.943317849),
        (10, 10, 0.722467200, 1.000000000),
        (1, 1, 0.206549314, 1.000000000),
        (150, 200, 0.685659017, 0.804918320),
        (200, 200, 0.981154674, 1.000000000),
        (1, 2, 0.094531206, 0.905468794),
    ],
)
def test_wilson_matches_bisected_roots(successes, total, lower, upper):
    """Literals produced by bisecting the score equation, not by this formula."""
    low, high = ev.wilson_interval(successes, total)
    assert low == pytest.approx(lower, abs=1e-9)
    assert high == pytest.approx(upper, abs=1e-9)


@pytest.mark.parametrize(("successes", "total"), [(1, 7), (13, 40), (150, 200), (37, 61)])
def test_wilson_bounds_are_roots_of_the_score_equation(successes, total):
    """The defining property: at each bound, ``|p̂ - p| = z·sqrt(p(1-p)/n)``.

    Independent of how the bound is computed — a wrong closed form (Wald, or a
    sign slip in the discriminant) fails this even when it looks plausible.
    """
    proportion = successes / total
    for bound in ev.wilson_interval(successes, total):
        residual = abs(proportion - bound) - Z95 * math.sqrt(
            bound * (1.0 - bound) / total
        )
        assert residual == pytest.approx(0.0, abs=1e-12)


def test_wilson_lower_bound_never_leaves_the_unit_interval():
    """Where the Wald interval goes negative, Wilson must not."""
    for total in (1, 5, 10, 200):
        low, high = ev.wilson_interval(0, total)
        assert low == 0.0
        assert 0.0 < high < 1.0
        low, high = ev.wilson_interval(total, total)
        assert high == 1.0
        assert 0.0 < low < 1.0


def test_wilson_lower_bound_rises_with_more_evidence():
    """Same rate, more trials: the lower bound must tighten upwards."""
    bounds = [ev.wilson_interval(int(0.8 * n), n)[0] for n in (10, 50, 200, 1000)]
    assert bounds == sorted(bounds)
    assert bounds[-1] < 0.8


def test_wilson_of_no_trials_claims_nothing():
    assert ev.wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        ev.wilson_interval(11, 10)


def test_the_gate_number_the_plan_pre_registered():
    """Plan §Policy gate: LB95 >= 0.75 over >= 200 placements.

    163/200 (81.5%) is the smallest count that clears it — worth pinning,
    because it is the bar the whole pipeline is aimed at, and 162 does not.
    """
    assert ev.wilson_interval(162, 200)[0] < 0.75 <= ev.wilson_interval(163, 200)[0]


# --- Temporal ensembling ------------------------------------------------------


def _offline_ensemble(chunks: list[np.ndarray], coeff: float, tick: int) -> np.ndarray:
    """The ACT paper's weighted average, computed directly from the history.

    Predictions for `tick` come from every chunk issued at ticks
    ``max(0, tick-K+1) .. tick``; the chunk issued at ``s`` holds it at index
    ``tick - s``. Ordered oldest-first, the j-th gets weight ``exp(-coeff·j)``.
    """
    chunk_size = chunks[0].shape[0]
    first = max(0, tick - chunk_size + 1)
    weights, values = [], []
    for order, issued in enumerate(range(first, tick + 1)):
        weights.append(math.exp(-coeff * order))
        values.append(chunks[issued][tick - issued])
    total = sum(weights)
    return sum(weight * value for weight, value in zip(weights, values, strict=True)) / total


@pytest.mark.parametrize("coeff", [0.1, 0.01, 0.0, -0.05])
def test_online_ensemble_equals_the_offline_weighted_average(coeff):
    """The online update is an optimisation, not a different algorithm."""
    rng = np.random.default_rng(20260811)
    chunk_size, dim, ticks = 8, 6, 25
    chunks = [rng.normal(size=(chunk_size, dim)) for _ in range(ticks)]
    ensembler = ev.TemporalEnsembler(coeff, chunk_size)
    for tick in range(ticks):
        online = ensembler.update(chunks[tick])
        offline = _offline_ensemble(chunks, coeff, tick)
        assert online == pytest.approx(offline, abs=1e-12)


def test_first_tick_is_the_first_chunks_first_action():
    rng = np.random.default_rng(1)
    chunk = rng.normal(size=(5, 6))
    assert ev.TemporalEnsembler(0.1, 5).update(chunk) == pytest.approx(chunk[0])


def test_a_constant_policy_is_a_fixed_point():
    """If every chunk agrees, the ensemble must not invent motion."""
    chunk = np.tile(np.arange(6.0), (12, 1))
    ensembler = ev.TemporalEnsembler(0.1, 12)
    for _ in range(30):
        assert ensembler.update(chunk) == pytest.approx(np.arange(6.0))


def test_zero_coefficient_is_the_unweighted_mean():
    rng = np.random.default_rng(3)
    chunk_size = 4
    chunks = [rng.normal(size=(chunk_size, 2)) for _ in range(chunk_size)]
    ensembler = ev.TemporalEnsembler(0.0, chunk_size)
    for tick in range(chunk_size):
        action = ensembler.update(chunks[tick])
    expected = np.mean([chunks[s][chunk_size - 1 - s] for s in range(chunk_size)], axis=0)
    assert action == pytest.approx(expected)


def test_positive_coefficient_weighs_the_oldest_chunk_most():
    """The paper's direction: ``w₀`` is on the oldest prediction."""
    chunk_size = 4
    old = np.zeros((chunk_size, 1))
    new = np.ones((chunk_size, 1))
    ensembler = ev.TemporalEnsembler(0.5, chunk_size)
    ensembler.update(old)
    for _ in range(chunk_size - 2):
        ensembler.update(new)
    action = ensembler.update(new)
    # Three "new" votes against one "old" one, yet the old one holds the
    # largest single weight, so the mean sits below the naive 3/4.
    assert 0.5 < action.item() < 0.75


def test_reset_forgets_the_previous_episode():
    rng = np.random.default_rng(5)
    chunks = [rng.normal(size=(6, 3)) for _ in range(4)]
    ensembler = ev.TemporalEnsembler(0.1, 6)
    for chunk in chunks:
        ensembler.update(chunk)
    ensembler.reset()
    assert ensembler.update(chunks[0]) == pytest.approx(chunks[0][0])


def test_a_wrong_sized_chunk_is_refused():
    ensembler = ev.TemporalEnsembler(0.1, 8)
    with pytest.raises(ValueError, match="8-step chunk"):
        ensembler.update(np.zeros((7, 6)))


def test_ensembler_survives_an_episode_longer_than_the_chunk():
    """The online counts must stay inside the weight table for a long episode."""
    ensembler = ev.TemporalEnsembler(0.1, 5)
    for _ in range(200):
        ensembler.update(np.zeros((5, 6)))


# --- Region cells -------------------------------------------------------------


def test_cells_tile_the_region():
    from manus.kinematics import GRASP_REGION

    low, high = GRASP_REGION.radius
    span = GRASP_REGION.azimuth_max_deg
    seen = set()
    for radius_bin in range(ev.RADIUS_BINS):
        for azimuth_bin in range(ev.AZIMUTH_BINS):
            radius = low + (high - low) * (radius_bin + 0.5) / ev.RADIUS_BINS
            azimuth = math.radians(-span + 2 * span * (azimuth_bin + 0.5) / ev.AZIMUTH_BINS)
            x = GRASP_REGION.pan_axis_xy[0] + radius * math.cos(azimuth)
            y = GRASP_REGION.pan_axis_xy[1] + radius * math.sin(azimuth)
            seen.add(ev.cell_of(x, y))
    assert len(seen) == ev.RADIUS_BINS * ev.AZIMUTH_BINS


def test_cell_grid_matches_the_expert_gate_report():
    """The two breakdowns are only comparable if the grids are identical."""
    source = (REPO_ROOT / "scripts" / "gen_workspace_map.py").read_text(encoding="utf-8")
    assert f"RADIUS_BINS = {ev.RADIUS_BINS}" in source
    assert f"AZIMUTH_BINS = {ev.AZIMUTH_BINS}" in source


def test_cell_label_reads_back_the_bounds():
    label = ev.cell_label("r0_az0")
    assert "r 0.111-" in label
    assert "-105" in label


def test_cells_outside_the_region_clamp_rather_than_crash():
    from manus.kinematics import GRASP_REGION

    x, y = GRASP_REGION.pan_axis_xy
    assert ev.cell_of(x + 10.0, y) == f"r{ev.RADIUS_BINS - 1}_az{ev.AZIMUTH_BINS // 2}"


# --- Outcome taxonomy ---------------------------------------------------------


def test_classify_covers_the_height_trace():
    spawn, threshold = 0.015, 0.065
    assert ev.classify(True, 0.07, spawn, threshold) == "success"
    assert ev.classify(False, 0.07, spawn, threshold) == "slipped"
    assert ev.classify(False, 0.05, spawn, threshold) == "short_lift"
    assert ev.classify(False, 0.016, spawn, threshold) == "no_lift"


def test_classify_success_wins_over_every_other_clause():
    assert ev.classify(True, -1.0, 0.015, 0.065) == "success"


# --- Summaries ----------------------------------------------------------------


def _episode(success: bool, cell: str, path: float = 1.0, excursion: float = 0.5) -> dict:
    return {
        "success": success,
        "cell": cell,
        "outcome": "success" if success else "no_lift",
        "motion": {"total_path_rad": path, "max_excursion_rad": excursion},
    }


def test_summarise_counts_and_bins():
    episodes = [_episode(True, "r0_az0"), _episode(False, "r0_az0"), _episode(True, "r2_az5")]
    summary = ev.summarise(episodes)
    assert summary["episodes"] == 3
    assert summary["successes"] == 2
    assert summary["cells"]["r0_az0"] == {"n": 2, "successes": 1, "rate": 0.5}
    assert summary["cells"]["r2_az5"]["rate"] == 1.0
    assert summary["outcomes"] == {"success": 2, "no_lift": 1}
    assert summary["wilson95"]["lower"] == pytest.approx(ev.wilson_interval(2, 3)[0])


def test_motion_summary_flags_a_dead_policy():
    episodes = [_episode(False, "r0_az0", path=0.0, excursion=0.0) for _ in range(3)]
    motion = ev.motion_summary(episodes)
    assert motion["episodes_with_no_motion"] == 3
    assert motion["max_excursion_rad"]["max"] == 0.0


def test_motion_summary_of_a_live_policy():
    episodes = [_episode(True, "r1_az3", path=4.0, excursion=1.2) for _ in range(2)]
    motion = ev.motion_summary(episodes)
    assert motion["episodes_with_no_motion"] == 0
    assert motion["total_path_rad"]["mean"] == pytest.approx(4.0)


# --- Run naming and the held-out contract -------------------------------------


def test_run_name_follows_the_plan_pattern():
    name = ev.stamped_run_name("eval", "grasp_cube_dev", "d0e26f309350c7433bdf")
    kind, dataset, stamp, sha = name.split("__")
    assert (kind, dataset, sha) == ("eval", "grasp_cube_dev", "d0e26f30")
    assert len(stamp) == 13 and stamp[8] == "-"


def test_run_name_without_a_repository_still_names_the_run():
    assert ev.stamped_run_name("eval", "x", None).endswith("__nogit")


def test_held_out_floor_is_the_plans_number():
    assert ev.EVAL_SEED_BASE == 10_000_000


def test_eval_draws_cannot_collide_with_training_draws():
    """Different namespace *and* index range: two independent reasons."""
    from manus.randomize import draw_episode, stable_hash64

    training = {stable_hash64("grasp_cube_dev", index) for index in range(200)}
    evaluation = {
        stable_hash64("eval_dev", ev.EVAL_SEED_BASE + index) for index in range(200)
    }
    assert not training & evaluation
    first = draw_episode("eval_dev", ev.EVAL_SEED_BASE)
    assert draw_episode("eval_dev", ev.EVAL_SEED_BASE) == first
    assert draw_episode("grasp_cube_dev", ev.EVAL_SEED_BASE) != first
