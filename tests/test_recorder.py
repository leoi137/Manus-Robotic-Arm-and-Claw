"""Tests for the raw episode format, the attempts ledger and the manifest.

The recorder is the narrowest point of the pipeline: everything downstream —
the converter, the replay gate, the trainer — reads what it wrote, in another
interpreter, months later. So the properties pinned here are the ones that are
expensive to discover late: frames come back in the slots they went into, an
interrupted write is invisible, a non-finite array never reaches disk, and the
same episodes always hash to the same ``dataset_id``.

Frames are a tiny 8x6 px so a whole dataset fits in a temp directory.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from manus import recorder

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

FRAME_SHAPE = (6, 8, 3)
"""(height, width, channels) of the synthetic wrist frames: 8x6 px."""


def make_frame(index: int) -> np.ndarray:
    """A smooth, index-specific test frame.

    Smooth so JPEG reproduces it closely, and keyed on `index` through the blue
    channel so two frames of the same episode are far apart in pixel space —
    which is what lets the round-trip test tell "frame 3 came back as frame 3"
    from "frame 3 came back as some other frame".
    """
    height, width, _ = FRAME_SHAPE
    rows = np.broadcast_to(np.linspace(0, 255, height)[:, None], (height, width))
    columns = np.broadcast_to(np.linspace(0, 255, width)[None, :], (height, width))
    blue = np.full((height, width), (index * 37) % 256)
    return np.stack([rows, columns, blue], axis=-1).astype(np.uint8)


def record_episode(attempt_index: int, steps: int) -> recorder.EpisodeRecorder:
    """An in-memory episode of `steps` steps with distinct, checkable values."""
    episode = recorder.EpisodeRecorder(attempt_index)
    for step in range(steps):
        episode.add_step(
            make_frame(step),
            np.full(recorder.NUM_JOINTS, 0.1 * step, dtype=np.float32),
            np.full(recorder.NUM_JOINTS, 0.1 * step + 0.05, dtype=np.float32),
            step / recorder.CONTROL_HZ,
        )
    return episode


def write_dataset(directory: Path, sizes: dict[int, int]) -> None:
    """Write one episode plus one ledger row per ``attempt_index: steps`` pair."""
    for attempt_index, steps in sizes.items():
        path = record_episode(attempt_index, steps).write(directory)
        recorder.append_attempt(
            directory,
            recorder.AttemptRecord(
                attempt_index, attempt_index, {"object_x": 0.2}, recorder.SUCCESS, path.name
            ),
        )


# --- Temporal contract -------------------------------------------------------


def test_the_temporal_contract_is_stated_verbatim_in_the_docstring():
    # The sentence travels with the data; the docstring is where it is written
    # down once. If the two ever diverge, one of them is lying.
    assert recorder.TEMPORAL_CONTRACT in recorder.__doc__


def test_control_rate_physics_rate_and_decimation_agree():
    assert recorder.CONTROL_HZ * recorder.DECIMATION * recorder.PHYSICS_DT == pytest.approx(1.0)


def test_meta_carries_the_temporal_contract(tmp_path):
    meta = recorder.load_episode(record_episode(0, 3).write(tmp_path)).meta
    assert meta["temporal_contract"] == recorder.TEMPORAL_CONTRACT
    assert (meta["control_hz"], meta["decimation"]) == (recorder.CONTROL_HZ, recorder.DECIMATION)
    assert meta["physics_dt"] == pytest.approx(recorder.PHYSICS_DT)


# --- Round trip --------------------------------------------------------------


def _expected_rows(steps: int, offset: float) -> np.ndarray:
    """The (steps, 6) float32 block :func:`record_episode` feeds the recorder."""
    return np.stack(
        [
            np.full(recorder.NUM_JOINTS, 0.1 * step + offset, dtype=np.float32)
            for step in range(steps)
        ]
    )


def test_round_trip_restores_every_array(tmp_path):
    sizes = {0: 3, 1: 5, 2: 4}
    write_dataset(tmp_path, sizes)

    for attempt_index, steps in sizes.items():
        episode = recorder.load_episode(recorder.episode_path(tmp_path, attempt_index))
        assert len(episode) == steps
        assert episode.attempt_index == attempt_index
        assert np.array_equal(episode.joint_pos, _expected_rows(steps, 0.0))
        assert np.array_equal(episode.actions, _expected_rows(steps, 0.05))
        assert np.array_equal(
            episode.timestamps, np.arange(steps, dtype=np.float64) / recorder.CONTROL_HZ
        )
        assert episode.joint_pos.dtype == np.float32
        assert episode.timestamps.dtype == np.float64
        assert episode.jpeg_offsets.dtype == np.int64


def test_round_trip_restores_frames_pixel_exactly(tmp_path):
    steps = 5
    record_episode(0, steps).write(tmp_path)
    episode = recorder.load_episode(recorder.episode_path(tmp_path, 0))

    for step in range(steps):
        # Pixel-exact against the codec, which is the only exactness a lossy
        # format can offer: same bytes in, same bytes out, at the same index.
        expected = recorder.decode_jpeg(recorder.encode_jpeg(make_frame(step)))
        assert np.array_equal(episode.frame(step), expected)


def test_each_frame_lands_in_its_own_slot(tmp_path):
    """Offsets, ordering and indexing, checked against the source pictures."""
    steps = 5
    record_episode(0, steps).write(tmp_path)
    episode = recorder.load_episode(recorder.episode_path(tmp_path, 0))

    for step in range(steps):
        decoded = episode.frame(step).astype(np.int16)
        distances = [
            float(np.abs(decoded - make_frame(other).astype(np.int16)).mean())
            for other in range(steps)
        ]
        # A decoded frame must sit far closer to its own source than to any
        # other frame's. The absolute distance is not zero and cannot be: at
        # 8x6 px, JPEG's 2x2 chroma subsampling alone costs a few levels.
        others = sorted(distances[:step] + distances[step + 1 :])
        assert distances[step] < others[0] / 2, distances


def test_frames_are_indexable_from_the_end(tmp_path):
    record_episode(0, 4).write(tmp_path)
    episode = recorder.load_episode(recorder.episode_path(tmp_path, 0))
    assert np.array_equal(episode.frame(-1), episode.frame(3))
    assert episode.jpeg(0).startswith(b"\xff\xd8")  # JPEG SOI marker


@pytest.mark.parametrize("index", [4, -5, 99])
def test_out_of_range_frames_raise(tmp_path, index):
    record_episode(0, 4).write(tmp_path)
    episode = recorder.load_episode(recorder.episode_path(tmp_path, 0))
    with pytest.raises(IndexError, match="out of range"):
        episode.frame(index)


def test_episodes_load_without_pickle(tmp_path):
    """The whole point of the format: no reader ever needs allow_pickle=True."""
    record_episode(7, 3).write(tmp_path)
    with np.load(recorder.episode_path(tmp_path, 7), allow_pickle=False) as data:
        assert set(data.files) == {*recorder.EPISODE_ARRAYS, "meta_json"}


def test_recorder_imports_without_isaac():
    """The converter reads episodes from the lerobot venv, where Isaac does not exist."""
    probe = (
        "import manus.recorder, sys;"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in {'isaacsim', 'isaaclab'});"
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=SRC_DIR, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --- Atomicity ---------------------------------------------------------------


def test_a_finished_write_leaves_no_temporary_file(tmp_path):
    record_episode(0, 3).write(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["episode_00000000.npz"]


def test_a_failed_write_leaves_nothing_behind(tmp_path, monkeypatch):
    def boom(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(recorder.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        record_episode(0, 3).write(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_unreplaced_temporary_file_is_invisible(tmp_path):
    """A generator killed between the write and the rename must not be seen."""
    write_dataset(tmp_path, {0: 3})
    crashed = tmp_path / "episode_00000001.npz.tmp"
    crashed.write_bytes(recorder.episode_path(tmp_path, 0).read_bytes())

    assert [path.name for path in recorder.episode_paths(tmp_path)] == ["episode_00000000.npz"]
    # And the manifest, which follows the ledger, ignores it just as thoroughly.
    manifest = recorder.build_manifest(tmp_path, {})
    assert manifest["counts"]["successes"] == 1


def test_episodes_are_listed_in_attempt_order(tmp_path):
    write_dataset(tmp_path, {12: 3, 2: 3, 100: 3})
    assert [path.name for path in recorder.episode_paths(tmp_path)] == [
        "episode_00000002.npz",
        "episode_00000012.npz",
        "episode_00000100.npz",
    ]


# --- Rejected input ----------------------------------------------------------


@pytest.mark.parametrize("field", ["joint_pos", "action", "timestamp"])
def test_non_finite_values_are_rejected_at_write_time(tmp_path, field):
    episode = recorder.EpisodeRecorder(0)
    for step in range(3):
        values = {
            "joint_pos": np.zeros(recorder.NUM_JOINTS),
            "action": np.zeros(recorder.NUM_JOINTS),
            "timestamp": step / recorder.CONTROL_HZ,
        }
        if step == 2:
            if field == "timestamp":
                values["timestamp"] = np.inf
            else:
                values[field] = np.full(recorder.NUM_JOINTS, np.nan)
        episode.add_step(make_frame(step), **values)

    with pytest.raises(ValueError, match="non-finite"):
        episode.write(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_empty_episode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="nothing recorded"):
        recorder.EpisodeRecorder(0).write(tmp_path)


def test_reserved_meta_keys_cannot_be_overridden(tmp_path):
    with pytest.raises(ValueError, match="written by the recorder"):
        record_episode(0, 2).write(tmp_path, {"control_hz": 1000})


def test_a_frame_of_the_wrong_kind_is_rejected():
    episode = recorder.EpisodeRecorder(0)
    with pytest.raises(ValueError, match="uint8"):
        episode.add_step(np.zeros(FRAME_SHAPE), np.zeros(6), np.zeros(6), 0.0)


def test_a_frame_shape_change_mid_episode_is_rejected():
    episode = recorder.EpisodeRecorder(0)
    episode.add_step(make_frame(0), np.zeros(6), np.zeros(6), 0.0)
    with pytest.raises(ValueError, match="frame shape changed"):
        episode.add_step(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros(6), np.zeros(6), 0.033)


def test_a_state_row_of_the_wrong_width_is_rejected():
    episode = recorder.EpisodeRecorder(0)
    with pytest.raises(ValueError, match="must have 6 entries"):
        episode.add_step(make_frame(0), np.zeros(5), np.zeros(6), 0.0)


# --- Ledger ------------------------------------------------------------------


def test_the_ledger_is_append_only_and_ordered(tmp_path):
    for attempt_index, outcome in enumerate(["success", "dropped", "no_ik"]):
        recorder.append_attempt(
            tmp_path,
            recorder.AttemptRecord(attempt_index, 99 + attempt_index, {"object_x": 0.2}, outcome),
        )
    records = recorder.read_attempts(tmp_path)

    assert [record["outcome"] for record in records] == ["success", "dropped", "no_ik"]
    assert [record["attempt_index"] for record in records] == [0, 1, 2]
    assert records[0] == {
        "attempt_index": 0,
        "seed": 99,
        "draw": {"object_x": 0.2},
        "outcome": "success",
        "episode_file": None,
    }
    # One line per attempt, and nothing rewritten in place.
    assert len((tmp_path / recorder.LEDGER_NAME).read_text().splitlines()) == 3


def test_reading_a_ledger_that_does_not_exist_yet(tmp_path):
    assert recorder.read_attempts(tmp_path) == []


def test_a_torn_ledger_line_names_itself(tmp_path):
    recorder.append_attempt(tmp_path, recorder.AttemptRecord(0, 0, {}, recorder.SUCCESS))
    with open(tmp_path / recorder.LEDGER_NAME, "a", encoding="utf-8") as handle:
        handle.write('{"attempt_index": 1, "seed"\n')
    with pytest.raises(ValueError, match=":2: malformed ledger line"):
        recorder.read_attempts(tmp_path)


def test_an_empty_outcome_is_rejected():
    with pytest.raises(ValueError, match="outcome must be non-empty"):
        recorder.AttemptRecord(0, 0, {}, "").to_dict()


# --- Manifest ----------------------------------------------------------------


def test_manifest_counts_successes_failures_and_frames(tmp_path):
    write_dataset(tmp_path, {0: 3, 1: 4})
    recorder.append_attempt(tmp_path, recorder.AttemptRecord(2, 2, {}, "dropped"))

    manifest = recorder.build_manifest(tmp_path, {"numpy": "2.4.4"})

    assert manifest["dataset_name"] == tmp_path.name
    assert manifest["counts"] == {"attempts": 3, "successes": 2, "failures": 1, "frames": 7}
    assert manifest["success_rate"] == pytest.approx(2 / 3)
    assert manifest["env"] == {"numpy": "2.4.4"}
    assert manifest["temporal"]["contract"] == recorder.TEMPORAL_CONTRACT


def test_the_validation_split_is_every_twentieth_success(tmp_path):
    write_dataset(tmp_path, {0: 3, 1: 3, 20: 3, 21: 3, 40: 3})
    recorder.append_attempt(tmp_path, recorder.AttemptRecord(60, 60, {}, "dropped"))

    manifest = recorder.build_manifest(tmp_path, {})

    # Successes only: attempt 60 failed, so it is not part of any split.
    assert manifest["val_episode_indices"] == [0, 20, 40]
    assert manifest["val_split_modulus"] == recorder.VAL_MODULUS


def test_regenerating_the_manifest_reproduces_it(tmp_path):
    write_dataset(tmp_path, {0: 3, 1: 4})
    first = recorder.build_manifest(tmp_path, {"numpy": "2.4.4"})
    second = recorder.build_manifest(tmp_path, {"numpy": "2.4.4"})

    assert first["dataset_id"] == second["dataset_id"]
    # Everything but the wall clock, which is deliberately outside the hash.
    assert {key: value for key, value in first.items() if key != "created"} == {
        key: value for key, value in second.items() if key != "created"
    }


def test_the_dataset_id_follows_the_pixels(tmp_path):
    original, altered = tmp_path / "original", tmp_path / "altered"
    write_dataset(original, {0: 3, 1: 3})
    write_dataset(altered, {0: 3, 1: 3})
    baseline = recorder.build_manifest(original, {})["dataset_id"]

    # Same episodes in a different directory: same content, same id.
    assert recorder.build_manifest(altered, {})["dataset_id"] == baseline

    # One pixel of one frame differs -> a different dataset entirely.
    episode = recorder.EpisodeRecorder(1)
    for step in range(3):
        frame = make_frame(step)
        if step == 1:
            frame[0, 0] = (0, 0, 0)
        episode.add_step(
            frame,
            np.full(recorder.NUM_JOINTS, 0.1 * step, dtype=np.float32),
            np.full(recorder.NUM_JOINTS, 0.1 * step + 0.05, dtype=np.float32),
            step / recorder.CONTROL_HZ,
        )
    episode.write(altered)
    assert recorder.build_manifest(altered, {})["dataset_id"] != baseline


def test_the_dataset_id_follows_the_actions(tmp_path):
    original, altered = tmp_path / "original", tmp_path / "altered"
    write_dataset(original, {0: 3})
    write_dataset(altered, {0: 3})
    baseline = recorder.build_manifest(original, {})["dataset_id"]

    episode = recorder.EpisodeRecorder(0)
    for step in range(3):
        episode.add_step(
            make_frame(step),
            np.full(recorder.NUM_JOINTS, 0.1 * step, dtype=np.float32),
            np.full(recorder.NUM_JOINTS, 0.1 * step + 0.06, dtype=np.float32),  # was 0.05
            step / recorder.CONTROL_HZ,
        )
    episode.write(altered)
    assert recorder.build_manifest(altered, {})["dataset_id"] != baseline


def test_a_duplicated_attempt_index_is_refused(tmp_path):
    write_dataset(tmp_path, {0: 3})
    recorder.append_attempt(
        tmp_path, recorder.AttemptRecord(0, 0, {}, recorder.SUCCESS, "episode_00000000.npz")
    )
    with pytest.raises(ValueError, match="duplicated attempt indices \\[0\\]"):
        recorder.build_manifest(tmp_path, {})


def test_a_success_row_without_its_episode_is_refused(tmp_path):
    recorder.append_attempt(
        tmp_path, recorder.AttemptRecord(0, 0, {}, recorder.SUCCESS, "episode_00000000.npz")
    )
    with pytest.raises(FileNotFoundError, match="is missing"):
        recorder.build_manifest(tmp_path, {})

    (tmp_path / recorder.LEDGER_NAME).unlink()
    recorder.append_attempt(tmp_path, recorder.AttemptRecord(0, 0, {}, recorder.SUCCESS))
    with pytest.raises(ValueError, match="no episode_file"):
        recorder.build_manifest(tmp_path, {})


def test_the_manifest_round_trips_through_disk(tmp_path):
    write_dataset(tmp_path, {0: 3})
    manifest = recorder.build_manifest(tmp_path, {"objects": ["cube_3cm"]})
    recorder.write_manifest(tmp_path, manifest)

    assert recorder.read_manifest(tmp_path) == manifest
    assert json.loads((tmp_path / recorder.MANIFEST_NAME).read_text())["dataset_id"]
    assert not list(tmp_path.glob("*.tmp"))


def test_git_provenance_is_recorded_for_a_dataset_in_the_repository():
    """The real repo, not a temp tree: this is where git actually answers."""
    git = recorder._git_provenance(SRC_DIR)
    assert git["sha"] and len(git["sha"]) == 40
    assert isinstance(git["dirty"], bool)


def test_git_provenance_outside_a_repository_is_unknown_not_fatal(tmp_path):
    assert recorder._git_provenance(tmp_path) == {"sha": None, "dirty": None}


def test_the_episode_file_path_is_zero_padded(tmp_path):
    assert recorder.episode_path(tmp_path, 42).name == "episode_00000042.npz"
    assert os.fspath(recorder.episode_path("some/dir", 0)).endswith("episode_00000000.npz")
