"""Pins what the data factory leaves on disk, and which half of it git keeps.

``tests/test_gitignore_contract.py`` pins the raw-dataset and run negations.
This file covers what the Step 9-12 scripts add on top: the converted LeRobot
tree (regenerable, ignored *without* negations), the contact sheet that sits
next to the ledger it describes (ignored), and the small previews in ``media/``
that have to stay committable for ``DATASETS.md`` to link to them.

The rest of the file is the sim-free half of the generator: the chunk planner
and the failure-retention policy decide how much GPU time a run spends and what
evidence survives it, and both are pure functions of the ledger.
"""

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

IGNORED_PATHS = [
    "datasets/lerobot/grasp_cube_v1/data/chunk-000/file-000.parquet",
    "datasets/lerobot/grasp_cube_v1/videos/observation.images.wrist/chunk-000/file-000.mp4",
    "datasets/lerobot/grasp_cube_v1/meta/info.json",
    "datasets/lerobot/grasp_cube_v1/val_split.json",
    "datasets/raw/grasp_cube_v1/preview.mp4",
]
"""Regenerable bulk: the converted dataset and the QC contact sheet."""

TRACKED_PATHS = [
    "media/datasets/grasp_cube_v1.gif",
    "media/eval/act__grasp_cube_v1__20260811-1200__d0e26f3.gif",
]
"""Small previews the generated catalogues link to."""


def _is_ignored(path: str) -> bool:
    """Whether git's exclude rules would ignore `path` (which need not exist)."""
    result = subprocess.run(["git", "check-ignore", "-q", "--", path], cwd=REPO_ROOT)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore failed on {path!r}: rc={result.returncode}")
    return result.returncode == 0


@pytest.mark.parametrize("path", IGNORED_PATHS)
def test_regenerable_artifacts_are_ignored(path):
    assert _is_ignored(path)


@pytest.mark.parametrize("path", TRACKED_PATHS)
def test_tracked_previews_are_not_ignored(path):
    assert not _is_ignored(path)


# --- Chunk planning ----------------------------------------------------------
# gen_dataset imports isaaclab at module scope (for AppLauncher's CLI args) and
# parses argv on import, so it is loaded through a helper that supplies both.


def _import_script(name, argv):
    """Import ``scripts/<name>.py`` fresh with `argv` as its command line."""
    pytest.importorskip("isaaclab", reason=f"{name} needs isaaclab for AppLauncher's args")
    import importlib

    saved = sys.argv
    sys.argv = [f"{name}.py", *argv]
    try:
        module = importlib.import_module(name)
        return importlib.reload(module)
    finally:
        sys.argv = saved
        sys.modules.pop(name, None)


def _gen_dataset(argv):
    """Import ``scripts/gen_dataset.py`` with a parsed command line."""
    return _import_script("gen_dataset", argv)


def _ledger(outcomes, start=0):
    """A minimal ledger: one record per outcome, indices from `start`."""
    return [
        {"attempt_index": start + index, "outcome": outcome}
        for index, outcome in enumerate(outcomes)
    ]


@pytest.fixture
def generator():
    return _gen_dataset(["--dataset", "d", "--target-successes", "10", "--chunk", "4"])


def test_a_fresh_dataset_starts_at_attempt_zero(generator):
    assert generator.plan_chunk([]) == ([0, 1, 2, 3], "pending")


def test_the_next_index_is_past_the_highest_not_the_count(generator):
    # A killed chunk can leave a gap; re-issuing an index would overwrite an
    # episode file and re-use its seed.
    todo, _ = generator.plan_chunk(_ledger(["success", "success"], start=7))
    assert todo == [9, 10, 11, 12]


def test_the_target_counts_successes_already_on_disk(generator):
    todo, status = generator.plan_chunk(_ledger(["success"] * 10))
    assert (todo, status) == ([], "complete")


def test_failures_do_not_count_towards_the_target(generator):
    todo, status = generator.plan_chunk(_ledger(["no_grasp"] * 4))
    assert status == "pending" and todo == [4, 5, 6, 7]


def test_a_chunk_never_overshoots_the_target(generator):
    # Nine successes recorded, ten wanted, chunk of four: run one attempt.
    todo, _ = generator.plan_chunk(_ledger(["success"] * 9))
    assert todo == [9]


def test_the_attempt_budget_ends_the_run():
    generator = _gen_dataset(
        ["--dataset", "d", "--target-successes", "10", "--max-attempts", "12"]
    )
    todo, status = generator.plan_chunk(_ledger(["no_grasp"] * 12))
    assert (todo, status) == ([], "exhausted")


def test_the_last_chunk_stops_at_the_attempt_budget():
    generator = _gen_dataset(
        ["--dataset", "d", "--target-successes", "10", "--max-attempts", "12", "--chunk", "50"]
    )
    todo, _ = generator.plan_chunk(_ledger(["no_grasp"] * 10))
    assert todo == [10, 11]


def test_the_attempt_budget_defaults_to_three_times_the_target():
    generator = _gen_dataset(["--dataset", "d", "--target-successes", "10"])
    assert generator.plan_chunk(_ledger(["no_grasp"] * 30))[1] == "exhausted"
    assert generator.plan_chunk(_ledger(["no_grasp"] * 29))[1] == "pending"


# --- Failure retention -------------------------------------------------------


def test_failures_are_kept_until_the_per_mode_cap(generator):
    kept = Counter({"no_grasp": generator.FAILURE_KEEP_PER_MODE - 1})
    assert generator.should_keep_failure(kept, "no_grasp")
    kept["no_grasp"] += 1
    assert not generator.should_keep_failure(kept, "no_grasp")
    # A mode that has not filled its own quota still gets a slot: the point of
    # the cap is a sample that spans modes.
    assert generator.should_keep_failure(kept, "slipped")


def test_retention_stops_at_the_total_cap(generator):
    kept = Counter({f"mode_{index}": 5 for index in range(generator.FAILURE_KEEP_TOTAL // 5)})
    assert sum(kept.values()) == generator.FAILURE_KEEP_TOTAL
    assert not generator.should_keep_failure(kept, "fresh_mode")


def test_only_failures_with_a_kept_file_count_against_the_cap(generator):
    attempts = [
        {"attempt_index": 0, "outcome": "success", "episode_file": "episode_00000000.npz"},
        {"attempt_index": 1, "outcome": "no_grasp", "episode_file": "failures/e.npz"},
        {"attempt_index": 2, "outcome": "no_grasp", "episode_file": None},
    ]
    assert generator.retained_failures(attempts) == Counter({"no_grasp": 1})


# --- Which episodes the previews and the replay gate look at -----------------
# Both are sample selections a human then trusts as representative, so what
# they pick is worth pinning: a preview that always showed the same episode, or
# a replay gate that only ever re-drove episode 0, would look identical from
# the outside and prove much less.

import make_previews  # noqa: E402 - needs the sys.path line at the top


@pytest.fixture
def ledger_tree(tmp_path):
    """A dataset directory whose ledger points at (empty) episode files."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from manus import recorder

    dataset_dir = tmp_path / "datasets" / "raw" / "d"
    for index in range(10):
        success = index % 4 != 3
        name = f"episode_{index:08d}.npz"
        if success:
            (dataset_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (dataset_dir / name).write_bytes(b"")
        recorder.append_attempt(
            dataset_dir,
            recorder.AttemptRecord(
                attempt_index=index,
                seed=index,
                draw={},
                outcome=recorder.SUCCESS if success else "no_grasp",
                episode_file=name if success else None,
            ),
        )
    return dataset_dir


def test_previews_only_show_successful_episodes(ledger_tree):
    chosen = make_previews.success_episodes(ledger_tree)
    assert [path.name for path in chosen] == [
        f"episode_{index:08d}.npz" for index in (0, 1, 2, 4, 5, 6, 8, 9)
    ]


@pytest.mark.parametrize(
    ("count", "wanted", "expected"),
    [(8, 3, [0, 4, 7]), (3, 3, [0, 1, 2]), (2, 3, [0, 1]), (50, 16, None)],
)
def test_preview_sampling_spans_the_dataset(count, wanted, expected):
    picked = make_previews._evenly_spaced(count, wanted)
    if expected is not None:
        assert picked == expected
    else:
        # Evenly spaced, both ends included, no repeats.
        assert len(picked) == wanted
        assert picked[0] == 0 and picked[-1] == count - 1
        assert picked == sorted(set(picked))


def test_the_replay_gate_takes_the_first_middle_and_last_episode(ledger_tree):
    replay_check = _import_script("replay_check", ["--dataset", "d"])
    chosen = replay_check.chosen_episodes(ledger_tree, 3)
    # Eight successes, so: first, middle, last.
    assert [path.name for path in chosen] == [
        "episode_00000000.npz",
        "episode_00000005.npz",
        "episode_00000009.npz",
    ]
