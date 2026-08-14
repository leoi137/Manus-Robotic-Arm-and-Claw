# Scan pilot — findings (2026-08-14)

One-object pilot testing the scan-phase recipe: 300 `cube_3cm` episodes with
the blind object-independent scan (commit `bd95d43`), ACT trained 50k steps,
evaluated in sim. **Verdict: the policy failed — 0 successes in 9 scored
episodes before the run was stopped.** The pilot did its job: it falsified the
recipe for ~$5 before the ~$40 eight-object generation.

## The run

| stage | result |
|---|---|
| Generation | 300/332 successes (90.4%), every scan exited `seen`, 137,518 frames |
| Raw verify | 11/11 after widening `FRAME_COUNT_RANGE` for scan-length episodes |
| LeRobot verify | 4/4 (`datasets/lerobot/grasp_cube_scan_pilot`, 221.6 MB) |
| Training | 50k steps, batch 64, train L1 0.027, **val L1 0.053** |
| Eval | **0/9 successes**, all `no_lift`, path lengths 22–26 rad |

Artifacts (all local): `datasets/{raw,lerobot}/grasp_cube_scan_pilot`,
`runs/train/train__pilot_scan__20260814-0248` (checkpoints, curves, logs),
`runs/eval/eval__grasp_cube_scan_pilot__20260814-0724__91a7499c` (incl. the
student rollout video `videos/ep0000_no_lift.mp4`).

## Diagnosis (working hypothesis, high confidence)

**The decisive lesson is one frame in a thousand, and the L1 imitation loss
barely weights it.** ~Half of every episode is "keep panning at constant
speed" — predictable from proprioception alone, no vision needed. The moment
that matters (cube enters the wrist camera's view → stop panning → approach)
occupies a handful of frames per ~485-frame episode. A policy can minimise
nearly all of the loss by sweeping competently and never truly reacting to the
image. The eval behavior matches exactly: 22–26 rad of path = endless
scanning, straight through the stop signal. This is the v1 image-blindness
failure in subtler form — the observation now *contains* the object, but
nothing in the objective *forces* the policy to act on it.

Secondary factor: train/val gap (0.027 vs 0.053; the 5000-episode run had
~none) = mild overfitting on 300 episodes. Real, but not the primary killer —
the imbalance would sink this recipe at 1000 episodes too.

## Fix candidates, in order of expected value

1. **Oversample the transition.** Weight training samples so windows
   overlapping the `seen → stop` moment (and the approach right after)
   dominate the gradient, instead of the pan monoculture. Cheapest change,
   attacks the root cause directly (sampling weights in the dataloader; the
   ledger already records `scan_steps` per episode, which locates the
   transition).
2. **Slow the scan near detection** (or globally) so the transition occupies
   more frames per episode.
3. **Shorter action chunks** (50 → ~20) so a stop decision is not buried
   mid-chunk; pairs with temporal-ensembling retuning.
4. **More episodes** (300 → 1000) against the overfitting gap.
5. Consider a small auxiliary objective ("is the object in view?" binary head)
   to force the vision pathway to carry signal — a model change, last resort.

## Also fixed/learned along the way

- `eval_policy.py` `MAX_CONTROL_STEPS` 450 → 1000 (remote) for scan-length
  episodes — else every scan episode times out as a failure regardless of
  policy quality. Needs porting into the repo.
- `train_act.py` invocations MUST pass `--num-workers` (default 0 = 7-9 s/step
  on this data; 6 workers = 0.2-0.33 s/step). Also `--ceiling-minutes` default
  45 truncates any real run.
- Training-time success % does not exist by construction (supervised learning
  never runs the robot). The scale-up run should add **periodic eval during
  training** (~20 episodes every 10k steps) so a failing recipe is caught
  hours earlier.
- Ops: killing a trainer that checkpoints-on-SIGTERM makes the wrapping
  pipeline read it as success and advance — stage gates need an explicit
  sentinel, not rc==0. And `pkill -f <pattern>` from an ssh one-liner matches
  its own shell; use `[b]racket` patterns.
