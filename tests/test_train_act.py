"""The trainer's sim-free, torch-free bookkeeping.

``scripts/train_act.py`` runs in `.venv-lerobot`, but its checkpoint retention,
loss summary, validation batching and split validation are plain Python — and
they are the parts whose bugs are silent. Retention that drops the wrong
directory loses a twelve-hour run's best policy; a val split that quietly
disagrees with the manifest trains on the data it is scored against. Both are
pinned here, in the same sim-free suite as everything else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_act as ta  # noqa: E402


# --- Checkpoint retention -----------------------------------------------------


def test_retention_keeps_last_best_and_every_ten_thousand():
    steps = [2000, 4000, 6000, 8000, 10000, 12000]
    keep = ta.retention_keep(steps, last=12000, best=6000, every=10000)
    assert keep == {12000, 10000, 6000}


def test_retention_keeps_last_even_when_it_is_also_best():
    assert ta.retention_keep([2000, 4000], last=4000, best=4000, every=10000) == {4000}


def test_retention_without_a_validation_point_still_keeps_last():
    assert ta.retention_keep([2000, 4000], last=4000, best=None, every=10000) == {4000}


def test_retention_never_drops_a_ten_thousand_multiple():
    steps = list(range(2000, 62000, 2000))
    keep = ta.retention_keep(steps, last=60000, best=58000, every=10000)
    assert {10000, 20000, 30000, 40000, 50000, 60000} <= keep
    assert 58000 in keep
    assert 2000 not in keep


def test_retention_bounds_the_disk_for_a_long_run():
    """Six 10k marks plus last plus best is the worst case at 60k steps."""
    steps = list(range(500, 60500, 500))
    keep = ta.retention_keep(steps, last=60000, best=37500, every=10000)
    assert len(keep) == 7


def test_checkpoint_directory_names_sort_lexically(tmp_path):
    """Zero padding, so ``sorted()`` on names is ``sorted()`` on steps."""
    names = [ta.checkpoint_dir(tmp_path, step).name for step in (500, 2000, 10000, 60000)]
    assert names == sorted(names)


def test_checkpoint_steps_reads_back_what_was_written(tmp_path):
    for step in (500, 2000, 10000):
        ta.checkpoint_dir(tmp_path, step).mkdir(parents=True)
    (tmp_path / "checkpoints" / "last").symlink_to("00010000")
    assert ta.checkpoint_steps(tmp_path) == [500, 2000, 10000]


def test_checkpoint_steps_of_a_fresh_run_is_empty(tmp_path):
    assert ta.checkpoint_steps(tmp_path) == []


# --- Loss summary -------------------------------------------------------------


def _curve(values: list[float]) -> list[dict]:
    return [{"step": index + 1, "loss": value, "l1_loss": value / 10} for index, value in enumerate(values)]


def test_loss_summary_compares_disjoint_windows():
    summary = ta.loss_summary(_curve([10.0] * 100 + [1.0] * 100), window=100)
    assert summary["windows_disjoint"] is True
    assert summary["first_mean"] == pytest.approx(10.0)
    assert summary["last_mean"] == pytest.approx(1.0)
    assert summary["decreased"] is True


def test_loss_summary_refuses_to_claim_on_overlapping_windows():
    """A 20-step shake-out compares a window with itself; that is not evidence."""
    summary = ta.loss_summary(_curve([5.0] * 20), window=100)
    assert summary["windows_disjoint"] is False
    assert summary["decreased"] is None
    assert summary["first_mean"] == summary["last_mean"]


def test_loss_summary_reports_a_rise_as_a_rise():
    summary = ta.loss_summary(_curve([1.0] * 100 + [9.0] * 100), window=100)
    assert summary["decreased"] is False


def test_loss_summary_of_nothing_is_empty_not_zero():
    summary = ta.loss_summary([], window=100)
    assert summary["steps"] == 0
    assert summary["first_mean"] is None
    assert summary["decreased"] is None


def test_loss_summary_tracks_l1_separately():
    """Train loss carries ACT's KL term; L1 is the part comparable with val."""
    summary = ta.loss_summary(_curve([10.0] * 100 + [2.0] * 100), window=100)
    assert summary["l1_first_mean"] == pytest.approx(1.0)
    assert summary["l1_last_mean"] == pytest.approx(0.2)


# --- Validation batching ------------------------------------------------------


def test_val_batches_partition_every_frame():
    batches = ta.fixed_val_batches(size=25, batch_size=8, cap=0)
    assert [len(batch) for batch in batches] == [8, 8, 8, 1]
    assert [index for batch in batches for index in batch] == list(range(25))


def test_val_batches_are_deterministic():
    assert ta.fixed_val_batches(37, 8, 0) == ta.fixed_val_batches(37, 8, 0)


def test_val_batches_cap_takes_the_same_prefix_every_time():
    capped = ta.fixed_val_batches(1000, 8, cap=4)
    assert len(capped) == 4
    assert capped == ta.fixed_val_batches(1000, 8, cap=0)[:4]


def test_val_batches_of_an_empty_split():
    assert ta.fixed_val_batches(0, 8, 0) == []


# --- Validation split integrity ----------------------------------------------


def _split(tmp_path: Path, payload: dict) -> Path:
    (tmp_path / "val_split.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_split_must_exist(tmp_path):
    with pytest.raises(SystemExit, match="convert_dataset"):
        ta.read_val_split(tmp_path, {"dataset_id": "abc"})


def test_split_must_come_from_this_dataset(tmp_path):
    """A stale conversion would train on one dataset and claim another."""
    _split(tmp_path, {"dataset_id": "old", "train_episode_indices": [1]})
    with pytest.raises(SystemExit, match="re-convert"):
        ta.read_val_split(tmp_path, {"dataset_id": "new"})


def test_split_must_leave_something_to_train_on(tmp_path):
    _split(tmp_path, {"dataset_id": "same", "train_episode_indices": []})
    with pytest.raises(SystemExit, match="no training episodes"):
        ta.read_val_split(tmp_path, {"dataset_id": "same"})


def test_matching_split_is_returned(tmp_path):
    payload = {"dataset_id": "same", "train_episode_indices": [1, 2], "val_episode_indices": [0]}
    _split(tmp_path, payload)
    assert ta.read_val_split(tmp_path, {"dataset_id": "same"}) == payload


def test_the_real_dev_split_agrees_with_its_manifest():
    """The converted dev dataset is derived, but its split sidecar is the contract."""
    lerobot_dir = REPO_ROOT / "datasets" / "lerobot" / "grasp_cube_dev"
    manifest_path = REPO_ROOT / "datasets" / "raw" / "grasp_cube_dev" / "manifest.json"
    if not lerobot_dir.is_dir() or not manifest_path.is_file():
        pytest.skip("grasp_cube_dev is not converted in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = ta.read_val_split(lerobot_dir, manifest)
    assert split["val_attempt_indices"] == manifest["val_episode_indices"]
    assert not set(split["train_episode_indices"]) & set(split["val_episode_indices"])


# --- Resilient reads ----------------------------------------------------------


class _Flaky:
    """A dataset whose frame decode fails `failures` times before succeeding."""

    def __init__(self, failures: int, length: int = 10) -> None:
        self.failures = failures
        self.length = length
        self.calls = 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("decodeAVFrame: Could not push packet to decoder")
        return f"item{index}"


def test_a_transient_decode_failure_is_retried():
    inner = _Flaky(failures=1)
    assert ta.ResilientDataset(inner, retries=2)[3] == "item3"
    assert inner.calls == 2


def test_a_persistently_bad_frame_falls_through_to_its_neighbour():
    """One unreadable frame must not end a twelve-hour run."""
    inner = _Flaky(failures=3)
    assert ta.ResilientDataset(inner, retries=2)[3] == "item4"


def test_a_non_decode_error_is_not_swallowed():
    class Broken:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> None:
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        ta.ResilientDataset(Broken())[0]


def test_retries_are_counted_for_the_run_record():
    before = ta._DECODE_RETRIES.value
    ta.ResilientDataset(_Flaky(failures=2), retries=3)[0]
    assert ta._DECODE_RETRIES.value == before + 2


# --- Run naming ---------------------------------------------------------------


def test_run_name_follows_the_plan_pattern():
    name = ta.stamped_run_name("train", "grasp_cube_v1", "abcdef1234567890")
    kind, dataset, stamp, sha = name.split("__")
    assert (kind, dataset, sha) == ("train", "grasp_cube_v1", "abcdef12")
    assert len(stamp) == 13 and stamp[8] == "-"


# --- Ceilings -----------------------------------------------------------------


def test_the_vram_ceiling_is_the_plans_number():
    assert ta.VRAM_CEILING_MIB == 5500
    assert ta.MIN_FREE_VRAM_MIB == 6500


def test_the_chunk_default_is_justified_in_the_module_docstring():
    assert ta.DEFAULT_CHUNK == 50
    assert "50 steps is" in ta.__doc__ or "1.67 s" in ta.__doc__
