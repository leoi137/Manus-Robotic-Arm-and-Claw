# Training plan — from sim datasets to a language-commanded real arm

*Written 2026-08-13, after the v2 data generation campaign completed.*

## End goal

A real SO-101 arm on the desk, a camera, and natural-language commands:
"grab the plastic cup", "grab the book", "grab the rubik's cube". Voice input
comes later (speech-to-text in front of the same command interface); it needs
no robot retraining.

## Where we are

All five v2 grasp datasets are generated, QC'd, converted to LeRobot format,
and downloaded locally (`datasets/lerobot/`, gitignored — ~2.6 GB):

| Dataset | Episodes | Success rate |
|---|---|---|
| grasp_cube_v2 | 1000 | 100.0% |
| grasp_die_v2 | 1000 | 95.2% |
| grasp_domino_v2 | 1000 | 100.0% |
| grasp_duplo_v2 | 1000 | 99.9% |
| grasp_pingpong_v2 | 1000 | 84.7% |

All were generated on a rented Vast.ai GPU (RTX 5080) and verified 4/4 at the
lerobot stage. The raw `.npz` episodes (12 GB, master copies for re-conversion)
still live only on the instance — sync before terminating it.

Two catalogue objects remain **failed/experimental**: `cylinder_3cm` (standing)
and `puck_d40x10`. The automated expert shoves them instead of clamping (the
moving jaw contacts first and pushes; the cylinder tips at 0.29 N vs 0.64 N
slide resistance, the puck squirts out). They are excluded from the sweep.

## Stage 1 — ACT baseline (diagnostic, do first)

Train **ACT** (Action Chunking with Transformers, the standard LeRobot recipe)
on the five datasets, on the rented GPU.

- ~80M-param encoder–decoder transformer; ImageNet-pretrained ResNet vision
  backbone, transformer trained from scratch. Predicts action chunks
  (~50 steps) from camera images + joint state.
- Cheap (hours on the 5080/5090 class) and the proven baseline for SO-101.
- **Purpose: validate the pipeline, not ship a product.** It answers whether
  the v2 data actually fixed v1's image-blindness (third-person camera, state
  dropout, collision fix) with minimal confounders. If ACT cannot grasp in sim
  eval, no bigger model will fix the data.

## Stage 2 — VLA fine-tune (the deliverable)

Fine-tune a pretrained **vision-language-action** model on the same LeRobot
datasets so the policy takes language commands natively:

- **SmolVLA** (~450M, LeRobot-native, fits the rented GPU) — first choice.
- π0 or GR00T are alternatives if we rent bigger VRAM.
- Pretrained on large multi-robot corpora (incl. SO-100/101 data), so words →
  objects → motions comes largely for free; fine-tuning specializes it to our
  arm, camera, and objects.
- **Pipeline prerequisite:** per-episode task strings ("grasp the cube",
  "grasp the ping pong ball", …). LeRobot has a task field; inject the label
  per dataset at conversion time (or patch the converted sets) — no
  regeneration needed.

## Stage 3 — real arm

- Buy the arm; run the Stage-2 policy with the real camera.
- Expect a round of real-world teleop demos to close the sim-to-real gap for
  real household objects — the sim datasets bootstrap the policy and validate
  the stack; real demos make it deployable.
- Command interface: text first ("grab the X"), then voice via speech-to-text.

## Parallel track — the two failed objects

Independent of training; feeds a later retrain/fine-tune.

1. `tools/arm_poser.html` — interactive WebGL posing tool with exact URDF
   kinematics (verified against `manus.kinematics` to ~1e-7). Hand-find grasp
   poses the IK expert can't: pose the arm, close the gripper, read exact
   joint angles in sim convention, export as JSON/Python/q-array.
2. Encode the working hand-found poses into `expert.py` (new approach/close
   strategy for side and low-flat grasps).
3. Sanity sweep → generate ~1000 episodes each → QC → convert → add to the
   training mix.

## Housekeeping before terminating the rental

- [ ] rsync `datasets/raw/` (12 GB) to local — the only unsynced master data.
- [ ] Decide: train Stage 1/2 on this instance first (it is set up and warm).
