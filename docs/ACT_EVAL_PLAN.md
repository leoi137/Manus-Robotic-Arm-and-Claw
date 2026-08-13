# How the Stage-1 ACT policy gets evaluated

*Written 2026-08-12, alongside the Stage-1 training launch.*

## There is no `scripts/eval_act.py`, on purpose

The evaluation infrastructure this run needs **already exists and already
works**. Writing a second, parallel eval script would mean a second
implementation of success detection, of held-out placement sampling, and of
observation preprocessing — and the moment two implementations of preprocessing
exist, they drift, and the number you report stops meaning what you think it
means.

The existing pieces:

| file | role |
|---|---|
| `scripts/policy_server.py` | loads a checkpoint, serves action chunks over a unix socket, **owns all normalization** |
| `scripts/eval_policy.py` | drives Isaac Sim, does temporal ensembling, scores with the expert's own success predicate, writes `runs/eval/<run>/` |
| `src/manus/expert.py` `GraspSuccessMonitor` | the success predicate — imported, never reimplemented |
| `src/manus/randomize.py` `draw_episode` | held-out placements (`attempt_index >= 10_000_000`) |

They are split across two interpreters — `.venv-lerobot` (torch, CPU) and
`~/isaaclab-env` (Isaac Sim, GPU) — because the two dependency trees cannot
coexist. The socket is the seam.

**Verified present on the rented box (2026-08-12):** `/root/isaaclab-env` exists
and `import isaacsim, isaaclab` succeeds. It prompts for the Omniverse EULA on
first launch in a new shell, so every eval command below must be run with
`OMNI_KIT_ACCEPT_EULA=YES` exported. Two prior eval runs are on disk under
`/root/Manus/runs/eval/`, so this path has been walked before — for v1.

## What is new for this run, and the one thing that must change

Stage 1 trains **one policy on five objects**. `scripts/eval_policy.py` takes a
single `--object` catalogue key and evaluates one object per run. That is
correct and needs no code change — it just means **five eval runs, not one**, and
the headline number is five success rates, not one.

The mapping from dataset to catalogue key (`src/manus/objects.py`):

| dataset | `--object` |
|---|---|
| `grasp_cube_v2` | `cube_3cm` |
| `grasp_die_v2` | `die_16mm` |
| `grasp_domino_v2` | `domino_20x40` |
| `grasp_duplo_v2` | `duplo_32x64` |
| `grasp_pingpong_v2` | `pingpong_40mm` |

Use a **distinct `--namespace` per object** (e.g. `eval_act_s1_cube`). The seed
is a hash of namespace *and* attempt index, so a shared namespace would hand
every object the same placement sequence — not wrong, but it would correlate the
five results and make a bad draw look like a bad policy across the board.

## The exact commands, after training finishes

Two shells (or two tmux panes) on the remote box. Run this **five times**, once
per row of the table above, changing `--object` and `--namespace` together.

```bash
# shell A — the policy, on the CPU
cd /root/Manus
/root/.venv-lerobot/bin/python scripts/policy_server.py \
    --ckpt runs/train/<RUN_NAME>/checkpoints/best \
    --warmup \
    --stats-path runs/eval/latency_cube.json

# shell B — the simulator, which owns the GPU alone
cd /root/Manus
export OMNI_KIT_ACCEPT_EULA=YES
~/isaaclab-env/bin/python scripts/eval_policy.py \
    --ckpt-run <RUN_NAME> \
    --object cube_3cm \
    --namespace eval_act_s1_cube \
    --episodes 200 \
    --video-every 20 \
    --headless
```

`<RUN_NAME>` is the directory under `runs/train/` — for this launch,
`train__grasp_v2_mix__<timestamp>__<sha8>`.

**The policy server must run on the CPU** (its default). `eval_policy.py` gives
the GPU entirely to Isaac Sim; a torch process holding VRAM alongside it is how
you get a mid-eval OOM three hours in. ACT inference on a CPU is ~20-40 ms,
comfortably inside a 33 ms control tick with temporal ensembling smoothing over
the jitter.

## How to read the result

Each run writes `runs/eval/<run>/run.json` + `report.md` + sample videos. The
number that matters is the **Wilson 95% lower bound**, not the point estimate —
at 200 episodes a 90% point estimate has a lower bound near 85%, and it is the
lower bound the plan gates on.

Expected outcomes, and what each means for the project:

* **LB95 > ~70% on cube/domino/duplo** — the pipeline works end to end. Proceed
  to Stage 2 (SmolVLA fine-tune) as `docs/TRAINING_PLAN.md` describes.
* **Uniformly near 0% with a healthy, converged training loss** — the classic
  v1 failure repeating: the policy fits the demonstrations but cannot *locate*
  the object at episode start, because the wrist camera does not see it there.
  Fix the observation (add a third-person camera to the datasets), not the
  model. Note the tour doc: adding a camera is a data change, not a code change.
* **Cube fine, pingpong near 0%** — expected and acceptable. Pingpong's expert
  itself only succeeded 84.7% of the time; a policy cannot exceed its
  demonstrations.
* **Everything low AND training loss plateaued high** — suspect the data, not
  the policy. Re-run `scripts/replay_check.py` before spending money on Stage 2.

## Before the instance is terminated

Per `docs/TRAINING_PLAN.md` housekeeping: `datasets/raw/` (12 GB of master `.npz`
episodes) still exists **only** on this instance. Sync it, plus
`runs/train/<RUN_NAME>/` and every `runs/eval/*`, before the rental ends.
