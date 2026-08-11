"""Convert a raw episode dataset into a LeRobotDataset. Runs in `.venv-lerobot`.

The crossing point between the two halves of the pipeline. Isaac wrote
``datasets/raw/<name>/`` under its own numpy; this reads it back under
lerobot's and writes ``datasets/lerobot/<name>/`` in the v3 schema that
``scripts/train_act.py`` consumes:

.. code-block:: bash

    ./.venv-lerobot/bin/python scripts/convert_dataset.py --dataset grasp_cube_dev
    ./.venv-lerobot/bin/python scripts/verify_dataset.py --dataset grasp_cube_dev --stage lerobot

The two environments disagree about numpy on purpose (2.4.4 writing, 2.2.6
reading — lerobot pins ``numpy<2.3``), and that combination is verified
interoperable for the recorder's dtypes, so this script **prints both versions
and asserts neither**. Pinning them equal would fail every legitimate run;
printing them is what makes a silent-corruption bug attributable later.
``manus.recorder`` is sim-free and numpy-only by construction, which is what
makes importing it here legal at all.

What is written, and why in this shape:

* ``observation.images.wrist`` — the wrist frames, 320x240. Raw frames are
  already recorded at that size (``scripts/gen_dataset.py`` downscales the
  camera's native 640x480 with PIL); anything else is resampled here, so a raw
  dataset recorded at native resolution still converts.
* ``observation.state`` / ``action`` — the six joint positions and the six
  joint targets, ``<joint>.pos`` in ``manus.specs.JOINT_NAMES`` order.
* ``task`` — ``"Pick up the cube"`` on every frame, as v3 requires.

The feature dictionary is built with lerobot's own helpers
(``hw_to_dataset_features`` / ``combine_feature_dicts``) rather than by hand,
so the schema follows the installed version instead of this file's memory of
it. The bookkeeping columns (``timestamp``, ``frame_index``, ``episode_index``,
``index``, ``task_index``) are lerobot's ``DEFAULT_FEATURES`` and are added for
us.

**Validation split.** The raw manifest owns it (``attempt_index % 20 == 0``,
recorded as ``val_episode_indices``). lerobot 0.6.1 has nowhere clean to put
per-episode metadata of our own — ``validate_episode_buffer`` rejects keys that
are not declared features — so the mapping is written as a sidecar
``val_split.json`` inside the dataset root, carrying both index spaces
(``attempt_index`` and the lerobot ``episode_index``) plus the episode order, so
the trainer never has to re-derive the correspondence.

**Verification is a reload, not a claim.** After ``finalize()`` the dataset is
re-opened from disk and checked against the manifest: episode count, frame
count, and one decoded video frame against the raw JPEG at the same index
(mean absolute difference under 3 levels — the codec's own noise floor is
~2, and an off-by-one frame is ~11).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the src-layout package importable without installing it. manus.recorder
# is sim-free (numpy + Pillow only), which is what lets this venv read episodes
# written by the Isaac one.
sys.path.insert(0, str(REPO_ROOT / "src"))

TASK = "Pick up the cube"
"""Per-frame natural-language task string, required by the v3 schema."""

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
"""Converted frame size (width, height)."""

IMAGE_KEY = "wrist"
"""Camera name; becomes ``observation.images.wrist``."""

ROBOT_TYPE = "so101_follower"
"""Recorded in the dataset metadata: which arm the joint names belong to."""

MAX_MEAN_ABS_DIFF = 3.0
"""Reload check tolerance in 8-bit levels; matches ``scripts/verify_dataset.py``."""

VAL_SPLIT_NAME = "val_split.json"
"""Sidecar carrying the validation split inside the lerobot dataset root."""


def _startup() -> tuple[Any, str]:
    """Import lerobot, or explain which interpreter to use. Returns the module and its schema version."""
    try:
        import lerobot
        from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    except ImportError as error:  # pragma: no cover - environment guard
        raise SystemExit(
            f"lerobot is not importable ({error}). Run this script with the conversion venv:\n"
            f"  ./.venv-lerobot/bin/python {Path(__file__).name} --dataset <name>"
        ) from None
    return lerobot, CODEBASE_VERSION


def build_features() -> dict[str, dict]:
    """The dataset schema, built by lerobot's own helpers.

    ``hw_to_dataset_features`` turns a hardware-shaped description (a float per
    joint, an ``(H, W, C)`` tuple per camera) into the v3 feature spec, and
    ``combine_feature_dicts`` merges the observation and action halves. Doing it
    this way means the schema is whatever the installed lerobot says it is —
    including any bookkeeping it adds — rather than a literal transcribed from
    documentation that may have moved on.
    """
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

    from manus import specs

    joints = {f"{name}.pos": float for name in specs.JOINT_NAMES}
    return combine_feature_dicts(
        hw_to_dataset_features(
            {**joints, IMAGE_KEY: (FRAME_HEIGHT, FRAME_WIDTH, 3)}, OBS_STR, use_video=True
        ),
        hw_to_dataset_features(joints, ACTION),
    )


def _image(frame: np.ndarray) -> np.ndarray:
    """One raw frame at the converted size, resampled only if it has to be."""
    if frame.shape[:2] == (FRAME_HEIGHT, FRAME_WIDTH):
        return frame
    from PIL import Image

    return np.asarray(
        Image.fromarray(frame).resize((FRAME_WIDTH, FRAME_HEIGHT), Image.LANCZOS),
        dtype=np.uint8,
    )


def success_records(dataset_dir: Path) -> list[dict[str, Any]]:
    """Successful attempts in ledger order — the order episodes are written in."""
    from manus import recorder

    return [
        record
        for record in recorder.read_attempts(dataset_dir)
        if record["outcome"] == recorder.SUCCESS and record.get("episode_file")
    ]


def convert(dataset_dir: Path, lerobot_dir: Path, repo_id: str) -> dict[str, Any]:
    """Write every successful raw episode into a fresh LeRobotDataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame

    from manus import recorder, specs

    features = build_features()
    print(f"features: {json.dumps({key: value['shape'] for key, value in features.items()})}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=recorder.CONTROL_HZ,
        features=features,
        root=lerobot_dir,
        robot_type=ROBOT_TYPE,
        use_videos=True,
    )

    records = success_records(dataset_dir)
    order = []
    for episode_index, record in enumerate(records):
        episode = recorder.load_episode(dataset_dir / record["episode_file"])
        for step in range(len(episode)):
            observation = {
                f"{name}.pos": float(episode.joint_pos[step][column])
                for column, name in enumerate(specs.JOINT_NAMES)
            }
            observation[IMAGE_KEY] = _image(episode.frame(step))
            action = {
                f"{name}.pos": float(episode.actions[step][column])
                for column, name in enumerate(specs.JOINT_NAMES)
            }
            dataset.add_frame(
                {
                    **build_dataset_frame(features, observation, OBS_STR),
                    **build_dataset_frame(features, action, ACTION),
                    "task": TASK,
                }
            )
        dataset.save_episode()
        order.append(
            {
                "episode_index": episode_index,
                "attempt_index": int(record["attempt_index"]),
                "frames": len(episode),
            }
        )
        print(
            f"  episode {episode_index:3d} <- attempt {record['attempt_index']:5d} "
            f"({len(episode)} frames)"
        )

    # Mandatory: without it the parquet footers are never written and the
    # dataset is unreadable, in a way that only shows up on reload.
    dataset.finalize()
    return {"episodes": order, "frames": sum(item["frames"] for item in order)}


def write_val_split(lerobot_dir: Path, manifest: dict[str, Any], order: list[dict]) -> Path:
    """Record the raw manifest's validation split next to the converted data."""
    val_attempts = set(manifest.get("val_episode_indices") or [])
    payload = {
        "source": "datasets/raw/<name>/manifest.json: val_episode_indices",
        "rule": f"attempt_index % {manifest.get('val_split_modulus')} == 0",
        "dataset_id": manifest.get("dataset_id"),
        "val_attempt_indices": sorted(val_attempts),
        "val_episode_indices": [
            item["episode_index"] for item in order if item["attempt_index"] in val_attempts
        ],
        "train_episode_indices": [
            item["episode_index"] for item in order if item["attempt_index"] not in val_attempts
        ],
        "episode_to_attempt": {
            str(item["episode_index"]): item["attempt_index"] for item in order
        },
    }
    path = lerobot_dir / VAL_SPLIT_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _as_uint8_hwc(image: Any) -> np.ndarray:
    """Normalise a lerobot image item to (height, width, 3) uint8.

    LeRobotDataset returns channel-first float32 in [0, 1] unless opened with
    ``return_uint8``.
    """
    array = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    if array.shape[0] == 3 and array.shape[2] != 3:
        array = array.transpose(1, 2, 0)
    if np.issubdtype(array.dtype, np.floating):
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0)
    return array.astype(np.uint8)


def verify_reload(
    dataset_dir: Path, lerobot_dir: Path, repo_id: str, manifest: dict[str, Any]
) -> bool:
    """Re-open the written dataset from disk and check it against the manifest."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from manus import recorder

    dataset = LeRobotDataset(repo_id, root=lerobot_dir)
    counts = manifest.get("counts") or {}
    checks = [
        (
            "episode_count",
            int(dataset.num_episodes) == counts.get("successes"),
            f"{dataset.num_episodes} vs {counts.get('successes')} successes",
        ),
        (
            "frame_count",
            int(dataset.num_frames) == counts.get("frames"),
            f"{dataset.num_frames} vs {counts.get('frames')} recorded frames",
        ),
    ]

    # One decoded frame against the raw JPEG at the same index: the check that
    # catches an off-by-one, a reordering or a channel swap, none of which move
    # the counts at all.
    records = success_records(dataset_dir)
    episode = recorder.load_episode(dataset_dir / records[0]["episode_file"])
    frame_index = len(episode) // 2
    item = dataset[frame_index]  # episode 0 starts at global index 0
    located = (int(item["episode_index"]), int(item["frame_index"]))
    difference = float(
        np.abs(
            _as_uint8_hwc(item[f"observation.images.{IMAGE_KEY}"]).astype(np.int16)
            - _image(episode.frame(frame_index)).astype(np.int16)
        ).mean()
    )
    checks.append(("frame_lands_where_expected", located == (0, frame_index), str(located)))
    checks.append(
        (
            "frame_matches_raw",
            difference < MAX_MEAN_ABS_DIFF,
            f"mean abs diff {difference:.2f} < {MAX_MEAN_ABS_DIFF}",
        )
    )

    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name} -- {detail}")
    return all(passed for _, passed, _ in checks)


def main(argv: list[str] | None = None) -> int:
    """Convert one dataset and verify the result. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="dataset name, e.g. grasp_cube_dev")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--repo-id", default=None, help="lerobot repo id (default manus/<dataset>)")
    parser.add_argument(
        "--overwrite", action="store_true", help="delete an existing converted dataset first"
    )
    args = parser.parse_args(argv)

    lerobot, codebase_version = _startup()
    from manus import recorder

    dataset_dir = args.root / "datasets" / "raw" / args.dataset
    lerobot_dir = args.root / "datasets" / "lerobot" / args.dataset
    repo_id = args.repo_id or f"manus/{args.dataset}"
    if not dataset_dir.is_dir():
        raise SystemExit(f"no such dataset: {dataset_dir}")
    manifest = recorder.read_manifest(dataset_dir)

    # Provenance, never an assertion — see the module docstring.
    print(
        f"numpy: {np.__version__} converting, "
        f"{(manifest.get('env') or {}).get('numpy') or 'unrecorded'} wrote the episodes"
    )
    print(f"lerobot {lerobot.__version__}, codebase {codebase_version}, repo_id {repo_id}")

    if lerobot_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{lerobot_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(lerobot_dir)

    written = convert(dataset_dir, lerobot_dir, repo_id)
    split = write_val_split(lerobot_dir, manifest, written["episodes"])
    print(f"wrote {split.relative_to(args.root)}")

    passed = verify_reload(dataset_dir, lerobot_dir, repo_id, manifest)
    size = sum(path.stat().st_size for path in lerobot_dir.rglob("*") if path.is_file())
    print(
        f"{'PASS' if passed else 'FAIL'}: {args.dataset} -> {lerobot_dir} "
        f"({len(written['episodes'])} episodes, {written['frames']} frames, "
        f"{size / 1e6:.1f} MB)"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
