# Synthetic Grasping Pipeline

How a simulated SO-ARM101 grasp becomes a LeRobot dataset, an ACT policy, and a closed-loop
eval number — and every version pin that makes it reproducible.

> **Status: skeleton.** Only *Environment & pins* below is real (written at plan Step 10, the
> venv build). Every section marked **PLACEHOLDER** is filled in at Step 18, once the code it
> describes exists. Do not cite a placeholder as if it were settled — the contract until then
> is [`plans/TODO_synthetic_grasping.md`](../plans/TODO_synthetic_grasping.md).

## Architecture

> **PLACEHOLDER (Step 18).** Stage diagram (mermaid) and the object → dataset → run → eval
> trace: kinematics (`src/manus/kinematics.py`) → scripted expert (`src/manus/expert.py`) →
> raw episode factory (`src/manus/recorder.py`, `scripts/gen_dataset.py`) → LeRobot conversion
> (`scripts/convert_dataset.py`) → ACT training (`scripts/train_act.py`) → closed-loop eval
> across the venv boundary (`scripts/policy_server.py` on CPU ‖ `scripts/eval_policy.py` in
> Isaac). Plan §Executive summary is the interim source.

## Temporal contract

> **PLACEHOLDER (Step 9/11, written up at Step 18).** The authoritative statement lives in the
> `EpisodeRecorder` docstring and is copied into every dataset manifest; this section reproduces
> it alongside the control/physics rates, render decimation, and the action↔state index
> convention. Plan §Design decisions holds the locked values until then.

## Dataset & run layout

> **PLACEHOLDER (Step 18).** `datasets/raw/<name>/` (`attempts.jsonl` ledger + `manifest.json`
> with content-hashed `dataset_id`), `datasets/lerobot/<name>/`, `runs/<kind>/<run_name>/`
> (`run.json` + `report.md`), and the generated catalogs `datasets/DATASETS.md` / `docs/RUNS.md`.

## Environment & pins

Two Python environments, never mixed. Isaac Sim and lerobot cannot share a process: they
disagree on numpy (below), and only one GPU process may run on this machine at a time.

| | `~/isaaclab-env` | `.venv-lerobot` (repo root, gitignored) |
|---|---|---|
| Role | scene, expert, data generation, closed-loop eval client | dataset conversion, ACT training, CPU policy server |
| Python | 3.12.3 | 3.12.3 (`/usr/bin/python3.12`) |
| numpy | 2.4.4 | 2.2.6 (lerobot pins `numpy<2.3.0`) |
| Pins | vendored Isaac Lab 3.0 stack | [`requirements-lerobot.txt`](../requirements-lerobot.txt) |

The two talk only through files (`episode_<attempt_index>.npz`, `allow_pickle=False`) and a
length-prefixed socket (Step 14) — never by importing each other.

### Recreating `.venv-lerobot`

```bash
cd <repo root>
/usr/bin/python3.12 -m venv --without-pip .venv-lerobot
PYTHONPATH=/usr/lib/python3/dist-packages ./.venv-lerobot/bin/python -m pip install -U pip setuptools wheel
./.venv-lerobot/bin/python -m pip install -r requirements-lerobot.txt
```

`--without-pip` plus the `PYTHONPATH` bootstrap is not a preference: this Ubuntu ships
`python3.12` without `ensurepip` (it lives in the unavailable `python3.12-venv` package), so a
plain `python3.12 -m venv` aborts with *"ensurepip is not available"*. Borrowing the system
pip once installs a real pip **inside** the venv, after which everything is ordinary. 5.4 GB
on disk, dominated by the bundled CUDA libraries.

`requirements-lerobot.txt` is a verbatim `pip freeze` (83 packages) — regenerate it with
`./.venv-lerobot/bin/python -m pip freeze > requirements-lerobot.txt`, and keep it free of
hand edits so it stays exactly reproducible.

### Pins that matter

| Package | Version | Note |
|---|---|---|
| `lerobot` | 0.6.1 | current PyPI stable; installed as `lerobot[dataset]` |
| `torch` | 2.11.0+cu130 | CUDA 13.0, stock PyPI wheel |
| `torchvision` | 0.26.0 | ACT's ResNet backbone |
| `torchcodec` | 0.11.1+cpu | video decode; links the system FFmpeg 8.0.1 |
| `datasets` | 4.8.5 | HF datasets, via the `dataset` extra |
| `numpy` | 2.2.6 | ceiling imposed by lerobot |

Installing bare `lerobot` is not enough: `lerobot.datasets` raises `ImportError` on import
without the `dataset` extra (`datasets`, `pandas`, `pyarrow`, `torchcodec`, `av`, `jsonlines`),
which the converter needs. Training extras (`lerobot[training]`: wandb, accelerate) are
deliberately **not** installed — `scripts/train_act.py` is ours and does not use them.

### GPU compatibility (RTX 2080, sm_75)

The stock PyPI wheel works — no CUDA-flavoured index URL needed. Verified, not assumed:

```
torch.cuda.get_arch_list() -> ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
torch 2.11.0+cu130 · cuda 13.0 · is_available() True · NVIDIA GeForce RTX 2080 · lerobot 0.6.1
```

`sm_75` (Turing) survives in CUDA 13 builds even though Maxwell/Pascal/Volta did not, so no
downgrade is required. Host driver 580.173.02. The card is shared — hold to the plan's
pre-flight (≥6500 MiB free, ≤5500 MiB ours, one GPU process machine-wide, which is why the
eval policy server runs on CPU).

### Cross-venv numpy gap

Isaac writes episodes under numpy 2.4.4; the converter reads them under numpy 2.2.6. Verified
by round-tripping an `.npz` with the recorder's exact dtypes (`uint8` blob, `int64` offsets,
`float32` joint state, `float64` timestamps, `<U` JSON string) between the two interpreters:
readable with `allow_pickle=False`, dtypes preserved. Step 11's converter still asserts both
numpy versions into the run provenance, because this is a silent-corruption class of bug.

### lerobot 0.6.1 API, as installed

Recorded **before** the converter was written, so it is transcription and not memory. All
paths verified by import in this venv.

```python
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION   # 'v3.0'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig     # draccus name: "act"
from lerobot.policies.act.modeling_act import ACTPolicy          # ACTPolicy(config, **kwargs)
```

Dataset lifecycle — create, fill frame by frame, finalize, then reload from disk as the
verification step:

```
LeRobotDataset.create(repo_id: str, fps: int, features: dict, root: str | Path | None = None,
    robot_type: str | None = None, use_videos: bool = True, tolerance_s: float = 0.0001,
    image_writer_processes: int = 0, image_writer_threads: int = 0,
    video_backend: str | None = None, batch_encoding_size: int = 1,
    rgb_encoder: RGBEncoderConfig | None = None, depth_encoder: DepthEncoderConfig | None = None,
    metadata_buffer_size: int = 10, streaming_encoding: bool = False,
    encoder_queue_maxsize: int = 30, encoder_threads: int | None = None,
    video_files_size_in_mb: int | None = None, data_files_size_in_mb: int | None = None)
    -> LeRobotDataset                                                        # classmethod

LeRobotDataset.add_frame(frame: dict) -> None      # frame MUST carry a 'task' key
LeRobotDataset.save_episode(episode_data: dict | None = None, parallel_encoding: bool = True)
LeRobotDataset.finalize()                          # mandatory; no finalize, no valid dataset
LeRobotDataset.resume(...)                         # classmethod, for chunked/resumable writes

LeRobotDataset.__init__(repo_id: str, root: str | Path | None = None,
    episodes: list[int] | None = None, episode_filter: Callable[[dict], bool] | None = None,
    image_transforms: Callable | None = None, delta_timestamps: dict[str, list[float]] | None = None,
    tolerance_s: float = 0.0001, revision: str | None = None, force_cache_sync: bool = False,
    download_videos: bool = True, video_backend: str | None = None, return_uint8: bool = False,
    depth_output_unit: str = 'mm', batch_encoding_size: int = 1,
    rgb_encoder: RGBEncoderConfig | None = None, depth_encoder: DepthEncoderConfig | None = None,
    encoder_threads: int | None = None, streaming_encoding: bool = False,
    encoder_queue_maxsize: int = 30, *, token: str | bool | None = None)     # read path
```

Notes for the converter (Step 11) and the trainer (Step 13):

- `LeRobotDatasetMetadata.create(...)` (`lerobot.datasets.dataset_metadata`) is the metadata-only
  half of the same call, if the schema is needed without a writer.
- Build the `features` dict with lerobot's own helpers — `hw_to_dataset_features`,
  `build_dataset_frame`, `combine_feature_dicts` in `lerobot.utils.feature_utils` — never hand
  literals. The bookkeeping columns (`timestamp`, `frame_index`, `episode_index`, `index`,
  `task_index`) are `DEFAULT_FEATURES` in `lerobot.utils.constants` and are added for you.
- `ACTConfig` defaults: `chunk_size=100`, `n_action_steps=100`, `n_obs_steps=1`,
  `vision_backbone='resnet18'`, `optimizer_lr=1e-5`, `device=None`, `use_amp=False`.
- `temporal_ensemble_coeff` defaults to `None` (ensembling off) and enabling it requires
  `n_action_steps == 1` — the config rejects any other combination.

## Commands actually run

> **PLACEHOLDER (Step 18).** Every command that produced a tracked artefact, in order, with
> its wall clock. Seeded with Step 10:

```bash
# Step 10 — build the conversion/training venv (see Recreating .venv-lerobot for the
# --without-pip rationale), then pin it.
/usr/bin/python3.12 -m venv --without-pip .venv-lerobot
PYTHONPATH=/usr/lib/python3/dist-packages ./.venv-lerobot/bin/python -m pip install -U pip setuptools wheel
./.venv-lerobot/bin/python -m pip install "lerobot[dataset]==0.6.1"
./.venv-lerobot/bin/python -m pip freeze > requirements-lerobot.txt
```

## Decisions recorded here

- **No lerobot async-inference stack.** Eval is a plain cross-venv sim loop (CPU policy server
  ‖ Isaac client, length-prefixed socket, no pickle); lerobot's robot abstraction buys nothing
  for a simulated arm we already drive directly.

> **PLACEHOLDER (Step 18).** Remaining out-of-scope calls and any decision the execution
> falsified, per plan §Out of scope.
