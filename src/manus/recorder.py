"""Pickle-free episode files, the attempts ledger and the dataset manifest.

The intermediate format between the two halves of the pipeline: Isaac writes
episodes here under one numpy, the lerobot converter reads them back under
another (see ``docs/PIPELINE.md``). Everything is stored as plain dtypes, so
every reader can — and does — load with ``allow_pickle=False``.

One successful attempt becomes one ``episode_<attempt_index:08d>.npz``:

===============  ===================  ==================================================
key              dtype/shape          contents
===============  ===================  ==================================================
``jpeg_blob``    uint8 (bytes,)       every wrist frame, JPEG-encoded and concatenated
``jpeg_offsets`` int64 (steps,)       cumulative *end* offset of each frame in the blob
``joint_pos``    float32 (steps, 6)   measured joint positions, radians
``actions``      float32 (steps, 6)   commanded joint targets, radians
``timestamps``   float64 (steps,)     simulated time of each control step, seconds
``meta_json``    ``<U`` scalar        compact JSON: the draw, the temporal contract, …
===============  ===================  ==================================================

Every episode is written to ``<name>.npz.tmp`` and only then moved into place
with ``os.replace``, so a reader never sees a half-written file and a crashed
generator leaves behind a ``.tmp`` that :func:`episode_paths` does not see.

Temporal contract, copied verbatim into every episode meta and every manifest
(``control_hz=30``, ``physics_dt=1/120``, ``decimation=4`` physics steps per
control step):

    action[t] is the joint target written before the step whose resulting state is joint_pos[t+1]

Alongside the episodes, a dataset directory holds ``attempts.jsonl`` — the
append-only ledger that is the source of truth for what was attempted — and
``manifest.json``, which is a pure function of the two and can be regenerated
at any time (:func:`build_manifest`).

Sim-free: this module never imports Isaac Sim. Lengths are metres, angles
radians, times seconds.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manus import specs

# --- Temporal contract -------------------------------------------------------

CONTROL_HZ = 30
"""Control (and capture) rate, in hertz: one recorded step per control tick."""

PHYSICS_DT = 1.0 / 120.0
"""Simulation step, in seconds."""

DECIMATION = 4
"""Physics steps per control step (``1 / (CONTROL_HZ * PHYSICS_DT)``)."""

TEMPORAL_CONTRACT = (
    "action[t] is the joint target written before the step "
    "whose resulting state is joint_pos[t+1]"
)
"""The index convention between :data:`actions` and :data:`joint_pos`.

Stated once, here, and copied verbatim into every episode meta and manifest —
an off-by-one against this sentence is invisible in the data and fatal to the
policy, so it travels with the data rather than living only in documentation.
"""

# --- Format ------------------------------------------------------------------

FORMAT_VERSION = 1
"""Bumped whenever the npz key set or meta contract changes."""

MANIFEST_VERSION = 1
"""Bumped whenever the manifest key set changes."""

NUM_JOINTS = len(specs.JOINT_NAMES)
"""Width of the ``joint_pos``/``actions`` rows."""

JPEG_QUALITY = 92
"""Encoder quality for wrist frames: visually lossless at 320x240, ~8 kB/frame."""

EPISODE_GLOB = "episode_*.npz"
"""Pattern matching finished episodes only — a crashed ``.npz.tmp`` cannot match it."""

LEDGER_NAME = "attempts.jsonl"
MANIFEST_NAME = "manifest.json"

SUCCESS = "success"
"""``AttemptRecord.outcome`` for a grasped-and-lifted attempt; anything else is a failure mode."""

VAL_MODULUS = 20
"""Validation split: successes whose attempt index is divisible by this are held out."""

EPISODE_ARRAYS = ("jpeg_blob", "jpeg_offsets", "joint_pos", "actions", "timestamps")
"""Array keys of an episode npz, in the order the content digest consumes them."""

_RESERVED_META = frozenset(
    {
        "attempt_index",
        "num_frames",
        "frame_shape",
        "jpeg_quality",
        "control_hz",
        "physics_dt",
        "decimation",
        "temporal_contract",
        "content_sha256",
        "format_version",
    }
)
"""Meta keys the recorder owns; a caller passing one of them is a bug, not an override."""


# --- Frame codec -------------------------------------------------------------


def encode_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """JPEG-encode one RGB frame.

    Args:
        frame: Shape (height, width, 3) uint8 array.
        quality: PIL JPEG quality.

    Returns:
        The encoded bytes.

    Raises:
        ValueError: If the frame is not a uint8 HxWx3 array.
    """
    array = np.ascontiguousarray(frame)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"frame must be uint8 (H, W, 3), got {array.dtype} {array.shape}")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def decode_jpeg(blob: bytes) -> np.ndarray:
    """Decode JPEG bytes to a writable (height, width, 3) uint8 array."""
    with Image.open(io.BytesIO(blob)) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


# --- Episodes ----------------------------------------------------------------


def episode_path(directory: str | os.PathLike[str], attempt_index: int) -> Path:
    """Canonical path of one attempt's episode file inside `directory`."""
    return Path(directory) / f"episode_{attempt_index:08d}.npz"


def episode_paths(directory: str | os.PathLike[str]) -> list[Path]:
    """Every finished episode in `directory`, in attempt order.

    Zero-padded names sort lexicographically in numeric order, and unfinished
    ``.npz.tmp`` files are excluded by construction (see :data:`EPISODE_GLOB`).
    """
    return sorted(Path(directory).glob(EPISODE_GLOB))


def _as_joint_vector(values: Any, name: str) -> np.ndarray:
    """Coerce `values` to a (NUM_JOINTS,) float32 row, or raise ValueError."""
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != NUM_JOINTS:
        raise ValueError(f"{name} must have {NUM_JOINTS} entries, got {array.size}")
    return array


def _reject_non_finite(name: str, array: np.ndarray, attempt_index: int) -> None:
    """Raise ValueError if `array` holds a NaN or an infinity.

    Enforced at write time rather than per step: a diverged controller or a
    dropped camera frame must never reach the dataset, and a file that never
    exists is far easier to reason about than one quarantined afterwards.
    """
    finite = np.isfinite(array)
    if not finite.all():
        rows = np.flatnonzero(~finite.reshape(len(array), -1).all(axis=1))
        raise ValueError(
            f"attempt {attempt_index}: {name} holds non-finite values at step(s) "
            f"{rows[:5].tolist()}{'...' if rows.size > 5 else ''}"
        )


def _content_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """SHA-256 over an episode's payload: pixels, state, actions and timing.

    Keyed by name, dtype and shape as well as raw bytes, so a reshaped or
    retyped array cannot collide with the original. This is what makes a
    dataset's ``dataset_id`` a content hash: the per-episode meta carries the
    digest, and the manifest hashes the metas.
    """
    digest = hashlib.sha256()
    for name in EPISODE_ARRAYS:
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}:{array.dtype.str}:{array.shape}:".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Episode:
    """One episode loaded from disk, with its frames still JPEG-encoded.

    Decoding is deferred to :meth:`frame` — a 1000-episode dataset is tens of
    gigabytes decoded and a few hundred megabytes as stored, so consumers that
    only need the state arrays (the ledger tooling, the replay gate) never pay
    for the pixels.

    Attributes:
        path: File the episode was read from.
        joint_pos: Shape (steps, 6) float32 measured joint positions, radians.
        actions: Shape (steps, 6) float32 commanded joint targets, radians.
        timestamps: Shape (steps,) float64 simulated times, seconds.
        meta: Decoded ``meta_json`` (see :meth:`EpisodeRecorder.write`).
        jpeg_blob: Shape (bytes,) uint8 concatenation of the encoded frames.
        jpeg_offsets: Shape (steps,) int64 cumulative end offsets into the blob.
    """

    path: Path
    joint_pos: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    meta: dict[str, Any]
    jpeg_blob: np.ndarray
    jpeg_offsets: np.ndarray

    def __len__(self) -> int:
        """Number of recorded control steps, as counted by the frame offsets."""
        return int(self.jpeg_offsets.size)

    @property
    def attempt_index(self) -> int:
        """Attempt this episode came from, per its meta."""
        return int(self.meta["attempt_index"])

    def jpeg(self, index: int) -> bytes:
        """Raw JPEG bytes of frame `index` (negative indices count from the end)."""
        index = self._checked(index)
        start = int(self.jpeg_offsets[index - 1]) if index else 0
        return self.jpeg_blob[start : int(self.jpeg_offsets[index])].tobytes()

    def frame(self, index: int) -> np.ndarray:
        """Decode frame `index` to a (height, width, 3) uint8 array."""
        return decode_jpeg(self.jpeg(index))

    def _checked(self, index: int) -> int:
        count = len(self)
        if not -count <= index < count:
            raise IndexError(f"frame {index} out of range for {count} frames in {self.path.name}")
        return index % count


def load_episode(path: str | os.PathLike[str]) -> Episode:
    """Load one episode file written by :meth:`EpisodeRecorder.write`.

    Deliberately permissive: it decodes the container and nothing else. Whether
    the arrays agree in length, stay finite and hold plausible pixels is the
    business of ``scripts/verify_dataset.py``, which must be able to *report*
    such damage rather than trip over it.

    Args:
        path: Path to an ``episode_*.npz``.

    Returns:
        The :class:`Episode`.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return Episode(
            path=path,
            joint_pos=data["joint_pos"],
            actions=data["actions"],
            timestamps=data["timestamps"],
            meta=json.loads(str(data["meta_json"].item())),
            jpeg_blob=data["jpeg_blob"],
            jpeg_offsets=data["jpeg_offsets"],
        )


class EpisodeRecorder:
    """Accumulates one episode in memory, then writes it as a single npz.

    One :meth:`add_step` per control tick, one :meth:`write` at the end. The
    recorder holds the whole episode (a few megabytes of JPEG at 600 steps) so
    that a failed attempt costs nothing but discarded memory: nothing touches
    the filesystem until the outcome is known.

    Example:
        >>> recorder = EpisodeRecorder(attempt_index=7)
        >>> recorder.add_step(frame, joint_pos, action, timestamp)  # doctest: +SKIP
        >>> recorder.write(dataset_dir, {"draw": draw.to_dict()})   # doctest: +SKIP
    """

    def __init__(self, attempt_index: int, *, jpeg_quality: int = JPEG_QUALITY) -> None:
        """Start recording attempt `attempt_index`, encoding frames at `jpeg_quality`."""
        if attempt_index < 0:
            raise ValueError(f"attempt_index must be non-negative, got {attempt_index}")
        self.attempt_index = attempt_index
        self.jpeg_quality = jpeg_quality
        self._jpegs: list[bytes] = []
        self._joint_pos: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._frame_shape: tuple[int, ...] | None = None

    def __len__(self) -> int:
        """Control steps recorded so far."""
        return len(self._jpegs)

    def add_step(
        self,
        frame: np.ndarray,
        joint_pos: Any,
        action: Any,
        timestamp: float,
    ) -> None:
        """Record one control step.

        Args:
            frame: Wrist-camera RGB frame, uint8 (height, width, 3). Encoded to
                JPEG immediately, so the caller may reuse the buffer.
            joint_pos: The 6 measured joint positions *after* the step, radians.
            action: The 6 joint targets written *before* it, radians (see
                :data:`TEMPORAL_CONTRACT`).
            timestamp: Simulated time of the step, seconds.

        Raises:
            ValueError: On a malformed frame, a wrong-width state row, or a
                frame whose shape differs from the ones already recorded.
        """
        jpeg = encode_jpeg(frame, self.jpeg_quality)
        shape = tuple(np.shape(frame))
        if self._frame_shape is None:
            self._frame_shape = shape
        elif shape != self._frame_shape:
            raise ValueError(
                f"attempt {self.attempt_index}: frame shape changed mid-episode, "
                f"{self._frame_shape} -> {shape}"
            )
        self._jpegs.append(jpeg)
        self._joint_pos.append(_as_joint_vector(joint_pos, "joint_pos"))
        self._actions.append(_as_joint_vector(action, "action"))
        self._timestamps.append(float(timestamp))

    def write(
        self, directory: str | os.PathLike[str], meta: Mapping[str, Any] | None = None
    ) -> Path:
        """Write the episode to ``<directory>/episode_<attempt_index:08d>.npz``.

        Atomic: the payload lands in a sibling ``.npz.tmp``, is flushed and
        fsynced, and only then replaces the destination. An interrupted write
        therefore leaves either the previous file or nothing at all — never a
        truncated episode — and the leftover ``.tmp`` is invisible to
        :func:`episode_paths`.

        Args:
            directory: Dataset directory (or its ``failures/`` subdirectory);
                created if absent.
            meta: Episode metadata to record alongside the arrays — the draw,
                the object, the outcome. The recorder adds the temporal
                contract, the frame geometry and the content digest, and
                refuses to let a caller overwrite them.

        Returns:
            The path written.

        Raises:
            ValueError: If nothing was recorded, if `meta` collides with a
                recorder-owned key, or if any array holds a non-finite value.
        """
        if not self._jpegs:
            raise ValueError(f"attempt {self.attempt_index}: nothing recorded")

        arrays = self._arrays()
        for name in ("joint_pos", "actions", "timestamps"):
            _reject_non_finite(name, arrays[name], self.attempt_index)
        meta_json = json.dumps(
            self._meta(arrays, meta or {}), sort_keys=True, separators=(",", ":")
        )

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = episode_path(directory, self.attempt_index)
        tmp = path.with_name(path.name + ".tmp")
        try:
            # A file object, not a filename: np.savez appends ".npz" to any
            # path that lacks it, which would defeat the ".npz.tmp" scheme.
            with open(tmp, "wb") as handle:
                np.savez(handle, meta_json=meta_json, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def _arrays(self) -> dict[str, np.ndarray]:
        """Pack the accumulated steps into the npz payload arrays."""
        return {
            "jpeg_blob": np.frombuffer(b"".join(self._jpegs), dtype=np.uint8),
            "jpeg_offsets": np.cumsum([len(jpeg) for jpeg in self._jpegs], dtype=np.int64),
            "joint_pos": np.stack(self._joint_pos).astype(np.float32),
            "actions": np.stack(self._actions).astype(np.float32),
            "timestamps": np.asarray(self._timestamps, dtype=np.float64),
        }

    def _meta(self, arrays: Mapping[str, np.ndarray], extra: Mapping[str, Any]) -> dict[str, Any]:
        """Merge caller metadata with the fields the recorder owns."""
        collisions = sorted(_RESERVED_META.intersection(extra))
        if collisions:
            raise ValueError(f"meta keys {collisions} are written by the recorder")
        return {
            **extra,
            "attempt_index": self.attempt_index,
            "num_frames": len(self._jpegs),
            "frame_shape": list(self._frame_shape or ()),
            "jpeg_quality": self.jpeg_quality,
            "control_hz": CONTROL_HZ,
            "physics_dt": PHYSICS_DT,
            "decimation": DECIMATION,
            "temporal_contract": TEMPORAL_CONTRACT,
            "content_sha256": _content_digest(arrays),
            "format_version": FORMAT_VERSION,
        }


# --- Attempts ledger ---------------------------------------------------------


@dataclass(frozen=True)
class AttemptRecord:
    """One line of ``attempts.jsonl``: what was tried, and what came of it.

    Attributes:
        attempt_index: Attempt counter within the dataset; also the episode
            file's number and the input to the per-attempt seed.
        seed: The 64-bit seed the draw was derived from
            (:func:`manus.randomize.stable_hash64`).
        draw: Serialized :class:`manus.randomize.EpisodeDraw` — the *draw*, not
            the seed, is the replay input.
        outcome: :data:`SUCCESS`, or the failure mode (``"no_ik"``,
            ``"dropped"``, …).
        episode_file: Path of the episode relative to the dataset directory, or
            None when nothing was kept.
    """

    attempt_index: int
    seed: int
    draw: dict[str, Any]
    outcome: str
    episode_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping, with the fields normalised to plain types."""
        if not self.outcome:
            raise ValueError(f"attempt {self.attempt_index}: outcome must be non-empty")
        return {
            "attempt_index": int(self.attempt_index),
            "seed": int(self.seed),
            "draw": dict(self.draw),
            "outcome": str(self.outcome),
            "episode_file": None if self.episode_file is None else str(self.episode_file),
        }


def append_attempt(dataset_dir: str | os.PathLike[str], record: AttemptRecord) -> Path:
    """Append one attempt to the dataset's ledger, durably.

    Append-only and fsynced per line: the ledger is the source of truth for a
    generation run that may be killed at any moment (shared GPU, wall-clock
    ceilings), so a line that has been written must survive the kill, and a
    line that has not must leave no trace.

    Args:
        dataset_dir: Dataset directory; created if absent.
        record: The attempt to record.

    Returns:
        Path of the ledger.
    """
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / LEDGER_NAME
    line = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def read_attempts(dataset_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read the ledger in write order.

    Args:
        dataset_dir: Dataset directory.

    Returns:
        One decoded record per line; empty if the ledger does not exist.

    Raises:
        ValueError: On a line that is not valid JSON (a torn write).
    """
    path = Path(dataset_dir) / LEDGER_NAME
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: malformed ledger line: {error}") from error
    return records


# --- Manifest ----------------------------------------------------------------


def _git_provenance(cwd: Path) -> dict[str, Any]:
    """Commit sha and dirty flag of the repository containing `cwd`.

    Both are None outside a repository (a dataset generated into a scratch
    directory, or a test) — recording "unknown" beats failing the write.
    """

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=cwd, capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "dirty": None}
    status = run("status", "--porcelain")
    return {"sha": sha, "dirty": None if status is None else bool(status)}


def _episode_meta_json(path: Path) -> str:
    """Read just the ``meta_json`` string out of an episode file."""
    with np.load(path, allow_pickle=False) as data:
        return str(data["meta_json"].item())


def build_manifest(
    dataset_dir: str | os.PathLike[str], env_block: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive a dataset's manifest from its ledger and episode files.

    A pure function of what is on disk (plus `env_block` and the clock): the
    manifest is a *view*, never a second source of truth, so it can be
    regenerated after any interrupted run and must come out identical.
    ``dataset_id`` — the SHA-256 over the sorted per-episode meta blobs, each
    of which carries that episode's own content digest — deliberately excludes
    the wall clock, so the same episodes always hash to the same id.

    Args:
        dataset_dir: Dataset directory holding ``attempts.jsonl`` and the
            episodes.
        env_block: Environment provenance recorded verbatim under ``env``:
            package versions, CUDA/driver, GPU, physics settings, and the
            ``objects`` list the catalogue reports.

    Returns:
        The manifest, ready for :func:`write_manifest`.

    Raises:
        ValueError: On a duplicated attempt index or a success row with no
            episode file.
        FileNotFoundError: If a success row's episode file is missing.
    """
    dataset_dir = Path(dataset_dir)
    attempts = read_attempts(dataset_dir)

    index_counts = Counter(record["attempt_index"] for record in attempts)
    duplicates = sorted(index for index, count in index_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{dataset_dir / LEDGER_NAME}: duplicated attempt indices {duplicates}")

    successes = [record for record in attempts if record["outcome"] == SUCCESS]
    blobs, frames = [], 0
    for record in successes:
        episode_file = record.get("episode_file")
        if not episode_file:
            raise ValueError(f"attempt {record['attempt_index']}: success row has no episode_file")
        path = dataset_dir / episode_file
        if not path.exists():
            raise FileNotFoundError(f"attempt {record['attempt_index']}: {path} is missing")
        blob = _episode_meta_json(path)
        blobs.append(blob)
        frames += int(json.loads(blob).get("num_frames", 0))

    digest = hashlib.sha256()
    for blob in sorted(blobs):
        digest.update(blob.encode("utf-8"))
        digest.update(b"\n")

    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": dataset_dir.name,
        "dataset_id": digest.hexdigest(),
        "counts": {
            "attempts": len(attempts),
            "successes": len(successes),
            "failures": len(attempts) - len(successes),
            "frames": frames,
        },
        "success_rate": len(successes) / len(attempts) if attempts else 0.0,
        "val_split_modulus": VAL_MODULUS,
        "val_episode_indices": sorted(
            record["attempt_index"]
            for record in successes
            if record["attempt_index"] % VAL_MODULUS == 0
        ),
        "temporal": {
            "control_hz": CONTROL_HZ,
            "physics_dt": PHYSICS_DT,
            "decimation": DECIMATION,
            "contract": TEMPORAL_CONTRACT,
        },
        "env": dict(env_block),
        "git": _git_provenance(dataset_dir),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def write_manifest(
    dataset_dir: str | os.PathLike[str], manifest: Mapping[str, Any]
) -> Path:
    """Write ``manifest.json`` atomically (indented, so a diff is readable)."""
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / MANIFEST_NAME
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_manifest(dataset_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Read ``manifest.json``. Raises FileNotFoundError if it was never built."""
    return json.loads((Path(dataset_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
