"""Pins the ``.gitignore`` negation contract: blobs out, provenance in.

Datasets and run outputs are ignored wholesale, then a handful of small files
that make them traceable are negated back in. That is easy to break silently —
adding one broad pattern, or dropping a directory negation, quietly stops
manifests and reports from ever being committed.

``git check-ignore`` evaluates patterns against pathnames, so these paths need
not (and mostly do not) exist; the test runs against the real repository rules.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

IGNORED_PATHS = [
    "datasets/raw/grasp_cube_v1/episode_00000001.npz",
    "datasets/raw/grasp_cube_v1/failures/episode_00000002.npz",
    "runs/train/act__grasp_cube_v1__20260811-1200__d0e26f3/checkpoints/step_2000.pt",
    "runs/eval/act__grasp_cube_v1__20260811-1200__d0e26f3/videos/attempt_0.mp4",
    ".venv-lerobot/lib/python3.12/site-packages/lerobot/__init__.py",
]
"""Bulk artefacts that must never enter git history."""

TRACKED_PATHS = [
    "datasets/raw/grasp_cube_v1/manifest.json",
    "datasets/raw/grasp_cube_v1/attempts.jsonl",
    "runs/eval/act__grasp_cube_v1__20260811-1200__d0e26f3/report.md",
    "runs/eval/act__grasp_cube_v1__20260811-1200__d0e26f3/run.json",
    "runs/expert_gate/report.md",
]
"""Provenance files the negations must keep committable."""


def _is_ignored(path: str) -> bool:
    """Whether git's exclude rules would ignore `path` (which need not exist)."""
    result = subprocess.run(["git", "check-ignore", "-q", "--", path], cwd=REPO_ROOT)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore failed on {path!r}: rc={result.returncode}")
    return result.returncode == 0


@pytest.mark.parametrize("path", IGNORED_PATHS)
def test_bulk_artifacts_are_ignored(path):
    assert _is_ignored(path)


@pytest.mark.parametrize("path", TRACKED_PATHS)
def test_provenance_files_are_not_ignored(path):
    assert not _is_ignored(path)
