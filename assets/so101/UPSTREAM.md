# Upstream provenance — SO-ARM101 model

- **Source repo:** https://github.com/TheRobotStudio/SO-ARM100
- **Commit:** `7629d2ad9853d10fb903093a33ef6114099d97e5` (shallow clone, fetched 2026-08-11)
- **License:** Apache License 2.0 (repo `LICENSE`)
- **Files vendored verbatim (no edits) from `Simulation/SO101/`:**
  - `so101_new_calib.urdf` → `urdf/so101_new_calib.urdf`
  - `joints_properties.xml` → `urdf/joints_properties.xml` (vendor system-identified actuator params)
  - `README.md` → `urdf/README.md`
  - `assets/*.stl` (13 meshes) → `urdf/assets/`

## Ground truth extracted from the vendored files

**Joints (URDF names == LeRobot SO-101 motor names), all revolute:**

| # | Joint | Limits (rad) |
|---|-------|--------------|
| 1 | `shoulder_pan` | −1.91986 … +1.91986 |
| 2 | `shoulder_lift` | −1.74533 … +1.74533 |
| 3 | `elbow_flex` | −1.69 … +1.69 |
| 4 | `wrist_flex` | −1.65806 … +1.65806 |
| 5 | `wrist_roll` | −2.74385 … +2.84121 |
| 6 | `gripper` | −0.174533 … +1.74533 |

Links: `base_link → shoulder_link → upper_arm_link → lower_arm_link → wrist_link → gripper_link` (+ fixed `gripper_frame_link` tool frame, + `moving_jaw_so101_v1_link` on the `gripper` joint).

**Vendor-calibrated STS3215 actuator model** (`joints_properties.xml`, MuJoCo `sts3215` class — system-identified, preferred over datasheet guesses):
- position kp = 17.8, kv = 0.0
- joint damping = 0.60, frictionloss = 0.052, armature = 0.028
- forcerange = ±3.35 N·m
- (backlash class: ±0.5° — not modeled in our Isaac articulation)

Note: the URDF's own `<limit effort="10" velocity="10">` values are generic placeholders; the calibrated forcerange above is authoritative for actuator effort limits.
