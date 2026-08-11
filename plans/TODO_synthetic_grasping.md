# Synthetic Grasping Pipeline (Sim-to-Real Stages 1–3) — Implementation Plan

## Status
In progress

## Snapshot
main@d0e26f3 · research: none inline — plan hardened by a 3-lens adversarial review panel (math / isaac-lerobot / CTO), 2026-08-11; all confirmed findings folded in below. Full findings: workflow run wf_95621d81-a2b.

## Goal
The simulated SO-ARM101 grasps objects mechanically via IK at ≥95% success, generates a categorized LeRobot-format synthetic dataset from the wrist camera, and an ACT policy trained on it clears a pre-registered statistical gate (Wilson LB95 ≥ 0.75 over ≥200 held-out placements) in closed-loop sim — with folder structure, machine-written provenance, and preview videos that make every object → dataset → run → eval traceable.

## Acceptance criteria
- [ ] **FK exact:** pure-numpy FK over the URDF chain (axes = local **+Z**, explicit `<axis xyz="0 0 1"/>`) matches Isaac `body_link_pos_w/quat_w` to <0.5 mm / <0.1° on 100 random configs (kinematics-only: `sim.forward()`, never `step()`); plus a sim-free golden check: home TCP ≈ (0.3914, 0, 0.2265).
- [ ] **IK solves:** 5-residual DLS (px, py, pz, pitch-sum, yaw) reaches ≥99% of a *fixed, explicit* GRASP_REGION (r∈[0.111, 0.220] m about pan axis (0.0388353, 0), |azimuth| ≤ 105° (measured shave from 110°: TCP roll-swing costs 4.08° of pan travel; boundary-sweep test is the real gate), minus base keep-out) within 1 mm position, **<0.5° tilt from vertical**, <1° yaw; round-trip pytest green.
- [ ] **Expert gate:** scripted expert (servo-to-converge waypoints, hover +3 cm, LIFT = joint-space retraction) succeeds ≥95% over ≥200 seeded placements; `runs/expert_gate/report.md` (tracked via gitignore negation) with per-region stats; failures kept and eyeballed.
- [ ] **Data traceable & QC'd:** `grasp_cube_dev` (50 successes) and `grasp_cube_v1` (1000 successes) exist as raw (attempts.jsonl ledger + manifest with `dataset_id` content-hash + env block) and converted LeRobotDataset (v3 schema from lerobot's own constructor, task string, `finalize()` + reload-verified); `scripts/verify_dataset.py` green on both; replay test green (3 episodes re-driven open-loop match recorded joint_pos — catches index shift/ordering/units).
- [ ] **Policy gate (pre-registered):** ACT trained on v1 (val split = attempt_index % 20 == 0, best-by-val real) evaluated closed-loop on ≥200 held-out placements (seed namespace attempt_index ≥ 10 000 000, zero-overlap asserted): **Wilson 95% lower bound ≥ 0.75**, reported over full region AND expert-feasible region; `runs/eval/<run>/report.md` + `run.json` provenance + videos.
- [ ] **Legibility:** `datasets/DATASETS.md` and `docs/RUNS.md` are *generated* (deterministic from manifests/run.json, lockfile-style pytest diff); small tracked previews in `media/datasets/` + `media/eval/`; `docs/PIPELINE.md` documents architecture, temporal contract, version pins, and every command actually run; `pytest tests/` green sim-free.

## Executive summary
Three stages, hard-gated in sequence. (1) **Kinematics:** numpy FK transcribed from the URDF (origins AND axes unit-verified against a re-parse; axes are local +Z), validated against Isaac link poses captured kinematically; IK = analytic seed (pan = −atan2 about the pan axis with the 7.9 mm TCP-offset fixed-point correction; roll = yaw − 177.211° + pan, branch-clamped) + damped-least-squares on a square 5-residual system. (2) **Expert + data factory:** FSM grasp expert that *servos to convergence* on each IK waypoint (open-loop misses by 5–13 mm of gravity droop under the vendored PD gains — measured, not assumed), hover +3 cm, retract in joint space after CLOSE (vertical-tool ceiling is 9.03 cm TCP height); chunked, ledger-driven, per-attempt-seeded episode generation writing pickle-free npz (JPEG blob + offsets + JSON meta) with render decimation; converted in a dedicated `.venv-lerobot` (py3.12, pinned lerobot + `requirements-lerobot.txt` committed) to LeRobotDataset; automated QC + open-loop replay gates. (3) **Training + eval:** ACT on the 2080 within shared-GPU ceilings (checkpoint retention: last/best/every-10k), evaluated closed-loop through a **CPU** policy server (msgpack/length-prefixed protocol — no pickle; server owns normalization; ONNX is the fallback, gated by a bit-parity test) driving the Isaac client. Privileged state is generator-only; the policy sees wrist RGB + joint state. Every dataset/run carries machine-written provenance (content hash, git sha + dirty bit, env versions, physics settings).

## Environment facts (grounded; panel-corrected)
- Repo main@d0e26f3; Isaac Lab 3.0 stack as landed (kw-only `*_index` setters, `ProxyArray.torch`, (x,y,z,w) quats in cfgs, `enable_cameras` forced, nested `Robot/Geometry/<chain>` prim paths).
- **URDF joints: all revolute rotate about local +Z (explicit `<axis xyz="0 0 1"/>`)**; USD physics agrees (`physics:axis = "Z"`). Origins carry calibrated offsets: wrist_roll rpy pitch 0.0486795 rad is a **yaw zero-offset about the roll axis** (not an approach tilt); elbow y −0.028; wrist_flex y +0.0052; gripper_frame (TCP) at (−0.0079, −0.000218, −0.0981) rpy (0,π,0) from gripper_link; the lateral offsets cancel to y=−0.00018 at the wrist_roll origin — the only real coupling is the 7.9 mm TCP offset from the roll axis.
- **Vertical-tool geometry (panel-verified):** pitch-sum constraint q_lift+q_elbow+q_wflex = 90°; wrist_flex limit ±95° caps vertical-tool TCP height at **0.0903 m**; grasp-height band r∈[0.062, 0.269]; hover+3cm band r∈[0.111, 0.220] (5° joint margin) — that intersection is GRASP_REGION. Base bottom sits 2.4 mm below world ground at current spawn — pin, don't "fix".
- Home TCP (q=0): (0.3914, 0.0000, 0.2265). Pan axis at (0.0388353, 0); positive pan swings −y (pan world axis is (0,0,−1)); tool yaw = 177.211° − pan + roll.
- Gravity droop under vendored gains: ~5–13 mm at TCP across the region (kp 17.8) — expert must converge, not count steps; `gravity_compensation_forces` exists (`isaaclab_physx .../articulation_data.py`) as optional feed-forward.
- `sim.forward()` = kinematics-only update (`simulation_context.py:710`); `body_link_pos_w/quat_w` are the URDF-frame-matching accessors (`body_state_w` deprecated).
- Camera per-episode jitter: NOT via Camera API/CameraCfg — use `scene["wrist_cam"]._view.set_local_poses(...)`, applied before first render; verify via rendered pixels (`update_latest_camera_pose` defaults False, `data.pos_w` lies). **LANDMINE (found by pixel-verification): `set_local_poses`' docstring claims (w,x,y,z) but the USD implementation consumes (x,y,z,w)** — `Vt.QuatdArray.FromNumpy` reads rows as (x,y,z,real). Always pass xyzw to FrameView writes; only trust conventions proven by rendered pixels.
- Per-episode color/light: direct USD attribute writes (object shader `inputs:diffuseColor`, dome `inputs:intensity`); add a `DistantLightCfg` so light *direction* randomization acts on something. Runtime friction: `root_view.get/set_material_properties` (surface friction; distinct from joint friction).
- Rendering: `sim.step(render=...)` decimation is the 4× throughput lever — render only on capture ticks; measure and record the factor. 1050 episodes ≈ 3–5 h with decimation.
- lerobot: pin current stable (expect ~0.6.x) on `/usr/bin/python3.12` venv, torch CUDA sm_75 OK; LeRobotDataset v3-era API: features from lerobot's constructor (no hand literals), per-frame `task` string required, `finalize()` mandatory, reload-from-disk as verification; record `lerobot.__version__` + `CODEBASE_VERSION` before writing the converter.
- **Shared GPU (Bestiary ceilings):** pre-flight ≥6500 MiB free before every GPU run; ≤5500 MiB ours; ONE GPU process machine-wide (⇒ eval policy server runs on **CPU**); chunked/resumable long runs with declared wall-clock ceilings; `du -s runs datasets` pre-flight too (60 GB cap; checkpoint retention enforced).

## Design decisions (locked unless execution falsifies them)
- Work surface = ground plane; objects: `cube_3cm` primary, `cylinder_3cm` secondary, sphere deferred. Colors randomized ⇒ not part of identity.
- Expert FSM: PREGRASP(hover **+3 cm**, servo-to-converge |q−target|<0.02 rad) → DESCEND(TCP z = object_grasp_z via named `TCP_TO_PAD_CENTRE` constant, tuned Step 7) → CLOSE(tuned per-object constant target; effort limit squeezes) → **LIFT = joint-space retraction toward a safe pose (no verticality constraint)** → HOLD. Success: object z ≥ spawn+5 cm sustained 30 steps with gripper closed.
- TCP convention: `gripper_frame_link` everywhere (7.3 mm above jaw tips; fix scene.py's 0.101 comment when touched).
- IK: 5-residual (px,py,pz,pitch-sum,yaw) DLS, adaptive λ, limit-clamped; seed per the corrected formulas; 2–3 fixed-point passes for the roll/position coupling.
- Intermediate format (pickle-free, `allow_pickle=False`): `episode_<attempt_index>.npz` = `jpeg_blob` (uint8 concat), `jpeg_offsets` (int64), `joint_pos`, `actions`, `timestamps` (float64), `meta_json` (str). Atomic write (`.tmp` + `os.replace`). Temporal contract in recorder docstring + manifest: `control_hz=30`, `physics_dt=1/120`, decimation=4, "action[t] is the target written before the step whose resulting state is joint_pos[t+1]".
- Generation ledger: append-only `attempts.jsonl` (attempt_index, seed, draw, outcome, failure_mode, file|null) is source of truth; `manifest.json` regenerable from it; per-attempt seed = `default_rng(stable_hash(dataset_name, attempt_index))`; failures kept (capped 50, sampled across modes) under `failures/`; CLI: `--target-successes/--max-attempts`.
- Provenance: manifest gains `dataset_id` (sha256 over sorted per-episode meta) + `env` block (versions, cuda, driver, GPU, physics_dt, control_hz, solver) + git sha + `dirty` bool. Every train/eval run writes `runs/<kind>/<run_name>/run.json` (dataset_name+id, converter args, ckpt path+sha256, versions, seeds, ceilings hit); run naming `<kind>__<dataset>__<YYYYMMDD-HHMM>__<gitsha8>`.
- Splits: val = `attempt_index % 20 == 0` (recorded in manifest; train_act computes val loss itself); eval namespace `attempt_index ≥ 10_000_000`, overlap-asserted.
- Socket protocol: JSON header + length-prefixed raw bytes (JPEG in, float32 chunk out). **No pickle** (CVE-shaped). Server: CPU, owns normalization. Fallback: ONNX (not TorchScript), gated by bit-parity test vs in-venv direct call.
- `.gitignore` via directory-negation (`datasets/raw/**`, `!datasets/raw/`, `!datasets/raw/*/`, `!datasets/raw/*/manifest.json`, `!datasets/raw/*/attempts.jsonl`; `runs/**`, `!runs/`, `!runs/**/`, `!runs/**/report.md`, `!runs/**/run.json`) + sim-free pytest using `git check-ignore` both ways.
- DATASETS.md / RUNS.md generated by `scripts/gen_catalog_md.py` from manifests + run.json; lockfile-style pytest diff. Tracked mini-previews: `media/datasets/<name>.gif`, `media/eval/<run>.gif`.

## Files (create / modify / delete)
- `src/manus/kinematics.py` — create — `KinematicChain` (FK, +Z axes), `ik_solve()` (seed + 5-residual DLS), `GRASP_REGION`, `TCP_TO_PAD_CENTRE`.
- `src/manus/objects.py` — create — `OBJECTS: dict[str, ObjectSpec]` (cube_3cm, cylinder_3cm).
- `src/manus/randomize.py` — create — seeded per-attempt samplers (pose in region, color, dome+distant light, camera jitter ±3 mm/±2°, ground albedo, friction); draw fully serialized into episode meta (the draw, not the seed, is the replay input — GPU PhysX isn't bit-reproducible across drivers).
- `src/manus/task_scene.py` — create — `GraspSceneCfg(SoArmSceneCfg)` (+object, +DistantLight, physics materials) and `apply_randomization(scene, draw)` (USD attr writes, camera `_view.set_local_poses` with quat-order helper).
- `src/manus/expert.py` — create — `ScriptedGraspExpert` FSM (sim-free logic, unit-tested transitions). [DONE]
- `src/manus/recorder.py` — create — `EpisodeRecorder` (npz format above), `DatasetManifest`, `attempts.jsonl` ledger.
- `scripts/dump_fk_fixture.py` — create — kinematics-only fixture dump (forward(), body_link_*, read-back q, root-relative poses, body ordering recorded).
- `scripts/demo_expert.py`, `scripts/gen_workspace_map.py` (+`--gate` mode), `scripts/gen_dataset.py`, `scripts/make_previews.py`, `scripts/verify_dataset.py`, `scripts/gen_catalog_md.py` — create.
- `scripts/convert_dataset.py`, `scripts/train_act.py`, `scripts/policy_server.py` — create — (.venv-lerobot side).
- `scripts/eval_policy.py` — create — (isaaclab-env client; ≥200 placements; Wilson LB; per-region breakdown; videos; run.json).
- `tests/test_kinematics.py`, `tests/test_objects.py`, `tests/test_expert_logic.py`, `tests/test_recorder.py`, `tests/test_gitignore_contract.py`, `tests/test_catalog_md.py`, `tests/fixtures/fk_fixture.json` — create — sim-free.
- `datasets/` + generated `DATASETS.md`; `docs/PIPELINE.md`, generated `docs/RUNS.md`, `docs/workspace_map.{png,json}`; `requirements-lerobot.txt` — create.
- `README.md` — modify (section + links); `.gitignore` — modify (negation block); `pyproject.toml` — modify (dev deps: numpy, pillow).

## Checklist

### Phase 1: Kinematics core (CPU; one kinematics-only GPU fixture run)
- [x] Step 1 (done: golden TCP (0.3914, −0.0000, 0.2265) ✓; URDF re-parse asserts origins AND axes; tool axis = gripper_frame +Z pointing world −Z at pitch-sum +90°; yaw relation verified to 0.0002°; 29 tests) — FK in `kinematics.py`: constants transcribed from URDF (origins AND `<axis>` — consistency unit test re-parses the URDF and asserts both), homogeneous FK to all link frames + TCP; sim-free golden test home TCP ≈ (0.3914, 0, 0.2265) — Verify: pytest green (axis error dies here, pre-GPU).
- [x] Step 2 (run by lead: 100 configs × 8 bodies, worst dev 0.0003 mm / 0.0001°) — `dump_fk_fixture.py` (GPU pre-flight): 100 random in-limit configs; `write_joint_state_to_sim` → `sim.forward()` → `robot.update(0.0)` → `body_link_pos_w/quat_w`; record read-back q + body ordering; poses root-relative; **never `sim.step()`** — Verify: fixture written, spot-sane.
- [x] Step 3 (green against the real fixture) — `test_fk_matches_isaac`: <0.5 mm / <0.1°, all 100 — Verify: green. **Foundation gate — halt on red.**
- [x] Step 4 (100% convergence over 1500 targets + 58k stress; worst 7.66 µm / 0.0007° tilt; analytic seed exact (0 DLS iters in-region); warm-start yaw-branch defect found+fixed (hint can never be worse than no seed — matters for the expert's servoing); azimuth cap MEASURED down to 105° (pan travel 110° minus 4.08° roll-swing) with a mutation-tested boundary-sweep gate) — `ik_solve()` + `GRASP_REGION` (explicit geometry incl. base keep-out): corrected seed, fixed-point TCP-offset pass, 5-residual DLS — Verify: ≥99% of region within 1 mm / <0.5° tilt / <1° yaw; round-trip green; margin data saved for the map.
- [ ] (phase boundary) pytest green sim-free; kinematics trusted.

### Phase 2: Task scene + expert (GPU, serialized)
- [x] Step 5 (DONE incl. GPU smoke: settle drift 0.000 mm; jitter→pixels 9.66 mean diff, visually a subtle POV shift; full randomization 37.84, lighting/albedo visibly changed. Lead found+fixed a REAL BUG the smoke exposed: `set_local_poses` consumes (x,y,z,w) despite its docstring claiming (w,x,y,z) — `Vt.QuatdArray.FromNumpy` reads (x,y,z,real); wxyz input produced a ~180° camera flip. Fixed in both write_wrist_camera_jitter and write_light_state) — `objects.py`, `randomize.py`, `task_scene.py` (incl. DistantLight, quat-order helper + its unit test, USD-write randomization) — Verify: unit tests; GPU smoke: spawn+settle (object at rest <1 mm drift); camera-jitter propagation proven via pixel diff of two renders.
- [x] Step 6 (done; real find: TCP-aim is infeasible on this hand — fixed-jaw geometry needs pad_lateral_offset −17 mm and 4-branch yaw planning w/ π-flip-substitution rejection; per-joint droop integrator with leak, unit-tested) — `expert.py` FSM (servo-to-converge; retraction LIFT) + `demo_expert.py` — Verify: one grasp on video, watched; commanded-vs-measured droop logged.
- [x] Step 7 (20/20 twice; TCP_TO_PAD_CENTRE 0.007→0.004 measured from jaw STLs; close_target 0.05 rad past-contact squeeze; close_ramp 20→60 steps — 2.2 rad/s jaw speed EJECTS the cube, caught on video; constants re-derived from STLs in tests) — Grasp tuning.
- [x] Step 8 (**PASS 200/200 = 100%**, fresh full re-run after fixing the 2-failure ejection mode found at 198/200; min height margin 28.8 mm — no marginal successes; zero timeouts; videos verified frame-by-frame by agent AND lead) — `gen_workspace_map.py` (map + `--gate`): ≥200 seeded placements chunked → **≥95%**; `runs/expert_gate/report.md` (tracked) per-region stats; failures kept + eyeballed (edge-cases OK, mid-region = bug) — Verify: gate met.
- [ ] (phase boundary) Expert trusted; region mapped (`docs/workspace_map.*`).

### Phase 3: Data factory (GPU gen + CPU convert)
- [x] Step 9 (grasp_cube_dev 50/50, 9836 frames, QC 11/11; raw at 320×240 by design (verify-gate contract + recorder docstring); episodes cut at success latch; render decimation measured honestly: 2.68× — frame grab dominates, not sim.step; 10.8 s/episode ⇒ 1000 eps ≈ 3 h) — `recorder.py` + `gen_dataset.py` (ledger, per-attempt seeds, atomic writes, render decimation, failure retention) + `make_previews.py` + `verify_dataset.py` (QC: counts/finiteness/monotone timestamps/frame-hash stall/pixel-roundtrip) — Verify: `grasp_cube_dev` (50 successes) generated; QC green; preview watched; recorder tests green.
- [x] Step 10 (done early: lerobot[dataset]==0.6.1, torch 2.11.0+cu130 sm_75 ✓ cuda True; CODEBASE_VERSION='v3.0'; LeRobotDataset.create signature captured in PIPELINE.md; venv built --without-pip (no python3.12-venv pkg) — recipe documented; NOTE numpy 2.4.4(writer)/2.2.6(reader) gap verified interoperable — Step 11 records both, no equality assert) — `.venv-lerobot` (`/usr/bin/python3.12`), pinned lerobot + torch (sm_75); commit `requirements-lerobot.txt` (real freeze); record `lerobot.__version__` + `CODEBASE_VERSION` in PIPELINE.md **before** writing the converter — Verify: `import lerobot, torch; torch.cuda.is_available()` True.
- [x] Step 11 (convert+reload 4/4: 50 eps / 9836 frames match, frame diff 1.66<3.0; val split as datasets/lerobot/<name>/val_split.json sidecar — lerobot 0.6.1 rejects undeclared per-episode keys; datasets/lerobot/** gitignored as derived artifact, test-pinned) — `convert_dataset.py` (features from lerobot's constructor; task string; 320×240; val indices into metadata; `finalize()`; numpy-version assert) — Verify: reload-from-disk; frame counts match manifest; decoded frame pixel-matches raw; QC re-run on converted.
- [x] Step 12 (PASS: max 14.5 mrad vs 50 tolerance across first/middle/last episodes; ramp-start sanity 6/6 joints correct direction) — **Replay gate:** re-drive 3 recorded episodes open-loop from `actions` at same draw → joint_pos matches within tolerance — Verify: green (catches index shift / ordering / units).
- [ ] (phase boundary) Dev dataset flows raw→LeRobot with QC + replay proofs.

### Phase 4: Training + closed-loop eval slice (GPU train; CPU server)
- [x] Step 13 (smoke: loss 10.65→1.11, val ×4 falling, best 0.1322@1500; VRAM peak 1336 MiB (batch 8 — huge headroom under 5500); 0.089 s/step ⇒ 60k ≈ 90 min; SIGTERM/resume proven with optimizer+RNG restore — a frozen-val-curve symptom caught a real optimizer/policy binding bug; chunk 50 justified vs 197-frame episodes) — `train_act.py` (VRAM-measured batch ≤5500 MiB; checkpoints 2k; retention last/best/every-10k; run.json; val-loss loop) + smoke 2k steps on dev (ceiling 45 min) — Verify: loss ↓; kill/resume works; run.json complete.
- [x] Step 14 (parity EXACT 0.0 vs both in-process and independent lerobot reference; mini-eval 10/10 mechanical — every joint moves, videos are the wire JPEGs themselves; smoke policy 0/10 as expected, learned descend-not-servo; latency p50 43.6 ms — synchronous stepping, wall-clock only; 380 tests green) — `policy_server.py` (CPU, owns normalization) + `eval_policy.py` (temporal ensembling; videos; run.json; Wilson LB reporting) + **parity test** (server vs direct call, identical output) + 10-placement mini-eval with smoke policy — Verify: loop runs mechanically; latency + VRAM recorded; parity green. (No success bar — smoke policy is weak by construction.)
- [ ] (phase boundary) Whole pipeline proven at dev scale.

### Phase 6 (user-added 2026-08-11): jaw collision fidelity — fix the visual/physics contact gap
User observation (screenshot): the fixed jaw visually never touches the held cube. Diagnosis hypothesis: convex-hull collision approximation bulges past the visual mesh (+ PhysX contact offset + the expert's 2 mm seating clearance), so physics clamps on an invisible fatter surface. Key for sim-to-real: policy images show a gap a real camera never would, and grasp-width calibration shifts by the bulge.
- [x] Step 19 (CPU-only, GPU untouched. **Hypothesis confirmed and quantified.** Both jaw colliders are `physics:approximation = "convexHull"` in `payloads/instances.usda` (converter `collision_type: Convex Hull`); collision meshes == visual meshes, so the hull IS the only error. No `contactOffset`/`restOffset` authored anywhere and `objects.py` sets none ⇒ both at schema default `-inf` = PhysX autocompute, and `restOffset`≈0 is what governs resting separation, so offsets contribute ~0 to the gap. **Bulge, ray-cast hull-vs-triangles over the cube's contact window in the TCP frame: fixed jaw 6.6 mm at the fingertip step / 8.6 mm at the cube's top edge (hull volume 2.58× mesh); moving jaw 1.8–2.1 mm (2.33×).** Cross-validated: narrowest *visual* opening hits 30 mm at jaw 0.195 rad, narrowest *hull* opening only at 0.320 rad — dead centre of the 0.27–0.35 rad the Step 8 gate measured. Predicted gap = 6.6 mm fixed + 1.8 mm moving ≈ the 8.9 mm the gap curves differ by; JAW_CLEARANCE contributes 0 (the effort-limited squeeze consumes it). Side find: the fixed jaw's hull intrudes 4.6 mm into the cube during DESCEND, so every episode shoves the cube ~4.6 mm outward before closing. Fix designed as SDF (both STLs watertight; 256 ⇒ 0.41/0.36 mm grid vs ~5 mm residual max for convex decomposition; triangle mesh is illegal on these dynamic articulation links). Scripts written, not run) — Investigate + quantify (CPU-only while training runs): parse the converted USD's collision prims/approximation attrs for both jaw meshes; compute convex-hull-vs-mesh deviation at the pad faces from the STLs (mm number); read converter/Isaac Lab collision options (SDF mesh / convex decomposition / contact-offset cfg); write (not run) `scripts/contact_probe.py` (contact-report readout proving where pad contacts occur) and the fix implementation — Verify: written analysis with the measured bulge; fix + probe scripts ready.
- [~] Step 20 PARTIAL (user-prioritized minimal path): SDF fix APPLIED (fix_jaw_collision.py, idempotent, backup kept); single filmed grasp SUCCEEDED with no retuning — jaw stall measured 0.189 rad vs 0.195 predicted for true surfaces (was 0.27–0.35 under the hull); HOLD frames verified by lead: both jaws visibly touch the cube (runs/collision_fix/{expert_demo_0000.mp4,hold_closeup.png}). DEFERRED on user priority: contact-probe before/after readout, constant re-tune sweep, fresh ≥200 expert re-gate. — Apply fix + GPU-verify (after F2/F3 free the GPU): apply collision fix to the jaw meshes (SDF or convex decomposition; trim contact offset if needed); contact probe shows both pads contacting where they visually touch; render check: held-cube frame shows no gap; re-run expert gate ≥200 (physics changed ⇒ retune close_target/TCP_TO_PAD_CENTRE if needed) ≥95% — Verify: probe + pixels + gate.
- [ ] Step 21 — Regenerate `grasp_cube_v2` (1000 successes) + QC + convert + replay; retrain ACT; re-eval ≥200 (same protocol); compare v1-baseline vs v2 policy (the v1 numbers are the control) — Verify: full cascade green; comparison table in eval report.

### Phase 5: Full scale + docs (long-running, ceilings enforced)
- [x] Step 15 (1000/1004 = 99.6%; 196,674 frames; all gates first-try green incl. replay 0.0000 rad and pixel-aligned conversion across the episode/attempt index shift; 3.4 h @ 10.8 s/ep dead-flat over 21 boots; 4 failures broken down honestly — azimuth 4-0 split p≈0.13 suggestive-not-significant, flagged for Step 17's dual-region report; dataset_id e5e250d3c76d) — Generate `grasp_cube_v1` (1000 successes; chunks of 100; resumable; ledger) + QC + previews + convert + catalog regen — Verify: counts; QC; preview watched; DATASETS.md regenerated + test green.
- [x] Step 16 (60000/60000 in 105.8 min; best val L1 0.0395 @ 55k on the FULL 10,015-frame val set; monotone descent, no overfit turn; batch 8 vs 16 settled by equal-wallclock A/B (b8 val 0.108 vs b16 0.161); val cadence 2500 chosen over --val-batches truncation to keep best-by-val trustworthy; retention verified live at 5k/15k/25k/40k; VRAM 1336 MiB peak; run train__grasp_cube_v1__20260811-1154__d0e26f30, best ckpt sha ed8c1fe9) — Full ACT training.
- [x] Step 17 — **Policy gate: FAIL, 6/200 = 3.0% (LB95 0.014 vs 0.75)** — protocol run exactly as pre-registered. ROOT CAUSE (3 independent proofs): policy is image-blind — cube is NEVER in the wrist camera's view at episode start (fixed home pose), so ACT collapses onto proprioception (image-swap moves action 0.003 rad vs 0.14 between-episode; corr(azimuth, pan)=0.005) and replays one canned trajectory; the 3% are placements that happen to lie on it. Val L1 0.0395 certified next-action prediction, not visual grounding. Serving/weights/harness verified correct (offline reproduces expert to 0.01–0.02 rad; 6 clean successes are the harness's positive control). ITERATION DECISION (lead): the ONE allowed iteration = observability fix (pre-sanctioned in Open questions) — add fixed third-person camera + break the proprioceptive shortcut (state dropout and/or randomized start pose) — BUNDLED with the Phase-6 collision fix into a single v2 regen. More episodes/randomization ruled out with evidence; run eval__grasp_cube_v1__20260811-1347__d0e26f30.
- [ ] Step 18 — `docs/PIPELINE.md` (architecture + mermaid + temporal contract + pins + commands actually run), catalog regen, README section, `media/` mini-previews — Verify: newcomer can trace object→dataset→run→eval from docs alone; all tests green.
- [ ] (phase boundary) All acceptance criteria re-checked.

## Changed manifest
- `src/manus/kinematics.py` — created+completed — FK surface + `ik_solve (seed+DLS, warm-start-safe), GRASP_REGION (105°), TCP_TO_PAD_CENTRE, in_grasp_region, in_base_keepout, ik_errors`
- `tests/test_kinematics.py` — created+extended — 50 tests incl. boundary sweep, warm-start guards, fixture gate (URDF-consistency, golden, invariants, fixture-consumer skips w/o fixture)
- `scripts/dump_fk_fixture.py` — created — kinematics-only fixture dump (forward(), _index writer, root-relative, fail-fast guards) — NOT yet run (GPU busy)
- `src/manus/objects.py` — created — `ObjectSpec, OBJECTS, make_spawn_cfg` (lazy isaac imports)
- `src/manus/randomize.py` — created (region constants now imported from kinematics) — `EpisodeDraw, draw_episode, stable_hash64, in_grasp_region, in_base_keepout, xyzw_to_wxyz, quat_* helpers`
- `src/manus/task_scene.py` — created, then lead-fixed — `GraspSceneCfg, apply_randomization, sun_orientation_xyzw, write_*` (GPU smoke green; xyzw fix in write_wrist_camera_jitter + write_light_state)
- `tests/test_objects.py` — created — objects + randomize tests
- `tests/test_gitignore_contract.py` — created — check-ignore both directions
- `.gitignore` — modified — negation block + .venv-lerobot/
- `pyproject.toml` — modified — dev deps numpy, pillow
- `requirements-lerobot.txt` — created — pip freeze (lerobot==0.6.1, torch 2.11.0+cu130)
- `docs/PIPELINE.md` — created — pins real, architecture placeholders
- `scripts/contact_probe.py` — created (Step 19, NOT run — GPU busy) — one scripted grasp to HOLD, then per-contact readout via `sensor.contact_view.get_contact_data(dt)` (position/normal/force/separation) for both jaws vs the object, expressed in the TCP frame; saves the HOLD wrist frame + json
- `scripts/fix_jaw_collision.py` — created (Step 19, NOT applied) — CPU pxr patch of `payloads/instances.usda`: the two jaw colliders convexHull → `sdf` + `PhysxSDFMeshCollisionAPI` (res 256); `--dry-run`/`--revert`/`--approximation convexDecomposition`/`--deinstance`/offset flags; idempotent, backs up to `*.orig-convexhull` (round-trip verified on a copy: apply → no-op re-apply → byte-exact revert; composed stage resolves `sdf` on both jaws only)
- `README.md` — modified — regenerate note: re-apply `fix_jaw_collision.py` after any URDF→USD reconversion
- `.gitignore` — modified — ignore `*.orig-convexhull` (the fix script's pristine backup)
- `.venv-lerobot/` — created (gitignored env, 5.4 GB)

## Validation
- Sim-free: `~/isaaclab-env/bin/python -m pytest tests/ -q`
- Fixture: `.../python scripts/dump_fk_fixture.py --headless` (once) → pytest.
- Expert: `.../python scripts/demo_expert.py --headless --video`; gate: `.../python scripts/gen_workspace_map.py --gate`.
- Data: `.../python scripts/gen_dataset.py --dataset grasp_cube_dev --target-successes 50`; `.venv-lerobot/bin/python scripts/convert_dataset.py --dataset grasp_cube_dev`; `.../python scripts/verify_dataset.py --dataset grasp_cube_dev --stage both`.
- Train/eval: `.venv-lerobot/bin/python scripts/train_act.py --dataset grasp_cube_v1 ...`; `.venv-lerobot/bin/python scripts/policy_server.py --device cpu --ckpt ...` ‖ `.../python scripts/eval_policy.py --run ...`
- Pre-flight (every GPU run): `nvidia-smi ... ≥ 6500` AND `du -s runs datasets` under budget.

## Migration / rollout
None. Data/checkpoints gitignored (negations track manifests/ledgers/reports/run.json); code, docs, generated catalogs, mini-previews committed later via /commit-push.

## Out of scope
Real-hardware stages 4–5, RL, multi-object clutter, sphere, VLA models, cloud training, lerobot async-inference stack (decision recorded in PIPELINE.md: cross-venv sim loop, no robot abstraction needed).

## Open questions
- (non-blocking) lerobot exact pin resolved at Step 10 (expect 0.6.x); converter written against the installed API, not memory.
- (non-blocking) If wrist-only obs caps success below gate after the one allowed iteration: add fixed third-person camera to the task scene (documented, sim-only) as the next plan.
- `src/manus/recorder.py` — created — `EpisodeRecorder, Episode, load_episode, AttemptRecord, append_attempt, build_manifest, TEMPORAL_CONTRACT, content hashing`
- `scripts/verify_dataset.py` — created — raw+lerobot QC CLI, writes verify_result into manifest
- `scripts/gen_catalog_md.py` — created — deterministic DATASETS.md / RUNS.md renderer
- `tests/test_recorder.py` — created — 39 tests
- `tests/test_catalog_md.py` — created — 7 tests (golden + idempotency)
- `tests/fixtures/fk_fixture.json` — created — 100-config Isaac ground truth
- `src/manus/expert.py` — created — `ScriptedGraspExpert, ExpertConfig, plan_grasp, tcp_target, pad_lateral_offset` (94 tests)
- `scripts/demo_expert.py` — created — single/multi-attempt driver, --tuning grid, --video, telemetry
- `scripts/gen_workspace_map.py` — created — map mode + --gate (chunked, resumable) + --report-only
- `tests/test_expert_logic.py` — created — 94 sim-free tests
- `docs/workspace_map.json`, `docs/workspace_map.png` — created — solvable-region map
- `src/manus/kinematics.py` — modified — TCP_TO_PAD_CENTRE 0.004 (STL-measured)
- `src/manus/objects.py` — modified — CLOSE_TARGET_30MM_RAD 0.05
- `scripts/gen_dataset.py` — created — chunked/resumable ledger-driven generator, --probe-render, env-block provenance
- `scripts/make_previews.py` — created — contact-sheet mp4 + tracked ≤3 MB GIF
- `scripts/convert_dataset.py` — created — raw→LeRobotDataset v3 w/ finalize+reload verification
- `scripts/replay_check.py` — created — open-loop replay gate
- `tests/test_dataset_artifacts.py` — created — 24 tests (incl. gitignore pins for lerobot/preview/media)
- `.gitignore` — modified — + datasets/lerobot/** (derived artifact)
- `datasets/raw/grasp_cube_dev/{manifest.json,attempts.jsonl}` — created (tracked); episodes+preview gitignored
- `datasets/DATASETS.md`, `docs/RUNS.md` — generated
- `media/datasets/grasp_cube_dev.gif` — created (tracked, 2.63 MB)
- `scripts/train_act.py` — created — ACT trainer: val-split honoring, VRAM probe, retention, run.json, SIGTERM-safe, resumable
- `scripts/policy_server.py` — created — CPU inference server, length-prefixed JSON+JPEG protocol
- `scripts/eval_policy.py` — created — closed-loop client, temporal ensembling, Wilson LB, per-region cells, videos
- `tests/test_train_act.py`, `tests/test_policy_protocol.py`, `tests/test_eval_policy.py`, `tests/test_policy_parity.py`, `tests/parity_reference.py` — created — 94 tests
- `datasets/raw/grasp_cube_v1/{manifest.json,attempts.jsonl}` — created (tracked); 1000 episodes + preview gitignored (1.79 GiB)
- `datasets/lerobot/grasp_cube_v1/**` — created (gitignored, 340 MB) + val_split.json (51 val / 949 train)
- `media/datasets/grasp_cube_v1.gif` — created (tracked, 2.24 MiB)
- `runs/train/train__grasp_cube_v1__20260811-1154__d0e26f30/{run.json,report.md}` — created (tracked via negations); 7 checkpoints on disk (4.1 GB, gitignored)
- `runs/eval/eval__grasp_cube_v1__20260811-1347__d0e26f30/{run.json,report.md}` — created (tracked); 21 videos gitignored
- `media/eval/eval__grasp_cube_v1__20260811-1347__d0e26f30.gif` — created (tracked, 2.17 MiB, success-vs-failure side by side)
- `assets/so101/usd/so101_new_calib/payloads/instances.usda` — modified — jaw colliders convexHull → SDF (resolution 256); backup *.orig-convexhull (gitignored)
- `scripts/fix_jaw_collision.py`, `scripts/contact_probe.py` — created (Step 19; fix script run + verified idempotent; probe not yet run)
- `runs/collision_fix/{expert_demo_0000.mp4(untracked), hold_closeup.png(untracked), demo.json}` — created — visual proof of the fix
