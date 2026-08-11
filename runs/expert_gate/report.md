# Expert gate — scripted grasp over the full grasp region

`expert_gate` draws 0..199, full per-attempt randomization (placement, object yaw, colour, friction, both lights, ground albedo, wrist-camera mount jitter). Success predicate: object centre ≥ spawn + 50 mm for 30 consecutive control steps with the jaws closed.

**200/200 = 100.0%** — gate is ≥ 95%: **PASS**

## Outcomes

| outcome | n |
| --- | --- |
| success | 200 |

Height margin over the 50 mm bar: min 28.8 mm, median 47.3 mm, max 58.3 mm — the successes are not marginal.

## Configuration

Every constant the gate was run with, read out of the code at report time.

| constant | value | where |
| --- | --- | --- |
| `TCP_TO_PAD_CENTRE` | 4.0 mm | kinematics |
| `pad_lateral_offset` | -17.0 mm | expert |
| `JAW_FIXED_FACE_X` | 0.0 mm | expert |
| `JAW_CLEARANCE` | 2.0 mm | expert |
| `close_target_rad` | 0.050 rad | objects[cube_3cm] |
| `gripper_open` | 1.500 rad | expert.ExpertConfig |
| `close_ramp` | 60 steps | expert.ExpertConfig |
| `hover_height` | 30 mm | expert.ExpertConfig |
| `converge_tol` | 20 mrad | expert.ExpertConfig |
| `state_budget` | 240 steps | expert.ExpertConfig |
| `hold_steps` | 45 steps | expert.ExpertConfig |
| `lift_rise` | 90 mm | expert.ExpertConfig |
| `droop gain/leak/limit` | 0.12 / 0.97 / 300 mrad | expert.ExpertConfig |
| `control rate` | 30 Hz (physics 1/120 s, decimation 4) | the pipeline contract |

## By region cell

| cell | bounds | n | successes | rate |
| --- | --- | --- | --- | --- |
| `r0_az0` | r 0.111-0.147 m, az -105..-70 deg | 7 | 7 | 100% |
| `r0_az1` | r 0.111-0.147 m, az -70..-35 deg | 6 | 6 | 100% |
| `r0_az2` | r 0.111-0.147 m, az -35..+0 deg | 7 | 7 | 100% |
| `r0_az3` | r 0.111-0.147 m, az +0..+35 deg | 12 | 12 | 100% |
| `r0_az4` | r 0.111-0.147 m, az +35..+70 deg | 7 | 7 | 100% |
| `r0_az5` | r 0.111-0.147 m, az +70..+105 deg | 9 | 9 | 100% |
| `r1_az0` | r 0.147-0.184 m, az -105..-70 deg | 16 | 16 | 100% |
| `r1_az1` | r 0.147-0.184 m, az -70..-35 deg | 17 | 17 | 100% |
| `r1_az2` | r 0.147-0.184 m, az -35..+0 deg | 11 | 11 | 100% |
| `r1_az3` | r 0.147-0.184 m, az +0..+35 deg | 15 | 15 | 100% |
| `r1_az4` | r 0.147-0.184 m, az +35..+70 deg | 15 | 15 | 100% |
| `r1_az5` | r 0.147-0.184 m, az +70..+105 deg | 7 | 7 | 100% |
| `r2_az0` | r 0.184-0.220 m, az -105..-70 deg | 12 | 12 | 100% |
| `r2_az1` | r 0.184-0.220 m, az -70..-35 deg | 16 | 16 | 100% |
| `r2_az2` | r 0.184-0.220 m, az -35..+0 deg | 10 | 10 | 100% |
| `r2_az3` | r 0.184-0.220 m, az +0..+35 deg | 10 | 10 | 100% |
| `r2_az4` | r 0.184-0.220 m, az +35..+70 deg | 13 | 13 | 100% |
| `r2_az5` | r 0.184-0.220 m, az +70..+105 deg | 10 | 10 | 100% |

## Per-state behaviour

Convergence is servo-to-converge: a state ends when the measured joints reach the waypoint, and the droop column is the integral bias the expert had to hold to get them there — i.e. the commanded-minus-measured offset gravity costs.

| state | steps (mean/max) | ‖q−target‖∞ at exit [mrad] | TCP error [mm] | droop bias [mrad] | exits |
| --- | --- | --- | --- | --- | --- |
| PREGRASP | 47 / 50 | 18.0 / 20.0 | 6.3 / 9.0 | 9.0 / 17.1 | {'converged': 200} |
| DESCEND | 31 / 59 | 16.7 / 20.0 | 4.7 / 7.1 | 9.3 / 79.9 | {'converged': 200} |
| CLOSE | 69 / 80 | 0.0 / 0.0 | 0.0 / 0.0 | 9.3 / 79.9 | {'stalled': 200} |
| LIFT | 31 / 39 | 15.6 / 20.0 | 6.5 / 9.1 | 15.6 / 28.8 | {'converged': 200} |
| HOLD | 45 / 45 | 0.0 / 0.0 | 0.0 / 0.0 | 15.6 / 28.8 | {'elapsed': 200} |

## Failures

None.

## Video evidence

Wrist POV with the state overlay burned in, recorded by `scripts/demo_expert.py --namespace expert_gate --attempt-list ... --video` (untracked — regenerate from the draw index in the filename).

- `diagnosis_close_ramp20_ejection_0117.mp4`
- `expert_gate_0036.mp4`
- `expert_gate_0107.mp4`

## Reproduce

```bash
~/isaaclab-env/bin/python scripts/gen_workspace_map.py --gate --headless  # x4, 50/boot
~/isaaclab-env/bin/python scripts/gen_workspace_map.py --report-only
```

Ledger: `runs/expert_gate/attempts.jsonl` (one JSON line per attempt, append-only, resumable).
