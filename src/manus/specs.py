"""Ground-truth constants for the SO-ARM101 (SO-101) arm.

Extracted from the vendored official model in ``assets/so101/urdf/``:
``so101_new_calib.urdf`` (joint names, limits, link chain) and
``joints_properties.xml`` (vendor system-identified STS3215 parameters).

Sim-free: imports without Isaac Sim. All joint angles are radians.
"""

# Actuated joints in Feetech bus motor-ID order (1..6); names match LeRobot.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Feetech bus motor ID -> joint name.
FEETECH_BUS_MAP: dict[int, str] = dict(enumerate(JOINT_NAMES, start=1))

# Travel (lower, upper) in radians, from the URDF <limit> tags.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),
}

# Neutral pose: every joint at zero (radians).
HOME_POSE: dict[str, float] = {name: 0.0 for name in JOINT_NAMES}

# --- STS3215 actuator model -------------------------------------------------
# Vendor system-identified values (joints_properties.xml, MuJoCo "sts3215"
# class). Authoritative: they supersede the URDF's generic effort/velocity
# placeholders (effort=10, velocity=10) and the datasheet figures below.
STS3215_KP = 17.8  # position-servo stiffness
STS3215_KV = 0.0  # position-servo velocity gain
STS3215_DAMPING = 0.60  # joint damping
STS3215_FRICTION = 0.052  # joint friction loss
STS3215_ARMATURE = 0.028  # reflected rotor inertia
STS3215_EFFORT_LIMIT = 3.35  # N·m, from forcerange="-3.35 3.35"

# Servo no-load speed class (rad/s): STS3215 is 5.45 @7.4 V / 4.72 @12 V
# (Feetech datasheet), Hiwonder HX-30HM is 5.51 @11.1 V. Used as the sim
# velocity limit.
SERVO_VELOCITY_LIMIT = 5.5

# Datasheet stall torques (N·m) of the two Feetech variants. REFERENCE ONLY:
# recorded for hardware sizing, never used by the actuator model above.
STS3215_STALL_TORQUE_7_4V = 1.91  # 19.5 kg·cm @ 7.4 V
STS3215_STALL_TORQUE_12V = 2.94  # 30 kg·cm @ 12 V (ST-3215-C018)

# The assembled Amazon kit (Hiwonder SO-ARM101, listing B0GT9BS7FZ) ships
# Hiwonder HX-30HM (follower) / HX-10HM (leader) 12 V-class bus servos, NOT
# Feetech STS3215 — same body size, 4096-count encoders, and torque class,
# so the actuator model above still applies. REFERENCE ONLY:
HX30HM_STALL_TORQUE = 2.94  # N·m, 30 kg·cm @ 11.1 V
HX30HM_NO_LOAD_SPEED = 5.51  # rad/s, 0.19 s/60° @ 11.1 V

# --- Links ------------------------------------------------------------------
# Serial chain, base outward; each consecutive pair is joined by JOINT_NAMES[i].
LINK_CHAIN: tuple[str, ...] = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
)

GRIPPER_FRAME_LINK = "gripper_frame_link"  # fixed tool frame on gripper_link
MOVING_JAW_LINK = "moving_jaw_so101_v1_link"  # child of the "gripper" joint
