"""Kinematics for the SO-ARM101 (SO-101) arm: FK, vertical-tool IK, workspace.

Pure numpy -- no Isaac Sim, no scipy -- so the chain is unit-testable on CPU
and reusable off-sim. The constants below are transcribed from the vendored
``assets/so101/urdf/so101_new_calib.urdf``; ``tests/test_kinematics.py``
re-parses that file and asserts every origin *and* every ``<axis>`` still
matches, so a re-export of the model cannot silently drift from this table.

Each joint contributes its fixed parent->child ``<origin>`` transform followed
by a rotation of its joint angle about the child frame's **local +Z** (the
URDF writes an explicit ``<axis xyz="0 0 1"/>`` on all six revolute joints).

Lengths are metres, angles radians. Poses are expressed in ``base_link``
coordinates -- the articulation root frame, which is what Isaac's
``body_link_pos_w``/``body_link_quat_w`` reduce to once the root pose is
divided out.

The tool frame is ``gripper_frame_link`` (the TCP): its **+Z** axis is the
approach direction, pointing *out of* the jaws, and it aims straight down
(world -Z) at the tool-vertical configuration :data:`PITCH_SUM_VERTICAL`.

:func:`ik_solve` inverts the chain for a whole :class:`ToolFamily` -- a choice
of pitch sum plus the convention for the one orientation freedom that is left
over. Two exist: :data:`TOOL_VERTICAL`, the top-down family the pipeline was
built on, over :data:`GRASP_REGION`; and :data:`TOOL_HORIZONTAL`, the
side-approach family a "grab it like a cup" grasp needs, over
:data:`SIDE_GRASP_REGION`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manus import specs

# Repo root is three parents up from src/manus/kinematics.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]

SO101_URDF_PATH: Path = _REPO_ROOT / "assets" / "so101" / "urdf" / "so101_new_calib.urdf"
"""Vendored URDF these constants are transcribed from (the consistency test's input)."""

Z_AXIS: tuple[float, float, float] = (0.0, 0.0, 1.0)
"""The only revolute axis in the model, in the joint's child frame."""


@dataclass(frozen=True)
class JointFrame:
    """One URDF joint: its fixed parent->child origin plus its rotation axis.

    Attributes:
        name: URDF joint name.
        parent: URDF parent link name.
        child: URDF child link name.
        xyz: ``<origin xyz>`` translation in the parent frame, metres.
        rpy: ``<origin rpy>`` fixed-axis roll/pitch/yaw, radians (Rz @ Ry @ Rx).
        axis: ``<axis xyz>`` unit rotation axis in the child frame, or None for
            a fixed joint (which the URDF writes as ``0 0 0``).
    """

    name: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float] | None

    def __post_init__(self) -> None:
        # FK below rotates about +Z unconditionally; refuse to be constructed
        # with an axis that would make that a silent lie.
        if self.axis is not None and self.axis != Z_AXIS:
            raise ValueError(f"joint {self.name!r}: FK assumes local +Z, got axis {self.axis}")


ARM_JOINTS: tuple[JointFrame, ...] = (
    JointFrame(
        name="shoulder_pan",
        parent="base_link",
        child="shoulder_link",
        xyz=(0.0388353, -8.97657e-09, 0.0624),
        rpy=(3.14159, 4.18253e-17, -3.14159),
        axis=Z_AXIS,
    ),
    JointFrame(
        name="shoulder_lift",
        parent="shoulder_link",
        child="upper_arm_link",
        xyz=(-0.0303992, -0.0182778, -0.0542),
        rpy=(-1.5708, -1.5708, 0.0),
        axis=Z_AXIS,
    ),
    JointFrame(
        name="elbow_flex",
        parent="upper_arm_link",
        child="lower_arm_link",
        xyz=(-0.11257, -0.028, 1.73763e-16),
        rpy=(-3.63608e-16, 8.74301e-16, 1.5708),
        axis=Z_AXIS,
    ),
    JointFrame(
        name="wrist_flex",
        parent="lower_arm_link",
        child="wrist_link",
        xyz=(-0.1349, 0.0052, 3.62355e-17),
        rpy=(4.02456e-15, 8.67362e-16, -1.5708),
        axis=Z_AXIS,
    ),
    # The 0.0486795 rad pitch here is a calibrated *yaw zero-offset about the
    # roll axis*, not an approach tilt: it shows up in TOOL_YAW_OFFSET below.
    JointFrame(
        name="wrist_roll",
        parent="wrist_link",
        child="gripper_link",
        xyz=(5.55112e-17, -0.0611, 0.0181),
        rpy=(1.5708, 0.0486795, 3.14159),
        axis=Z_AXIS,
    ),
)
"""The five actuated arm joints, base outward. Order defines the ``q`` vector."""

TCP_JOINT: JointFrame = JointFrame(
    name="gripper_frame_joint",
    parent="gripper_link",
    child=specs.GRIPPER_FRAME_LINK,
    xyz=(-0.0079, -0.000218121, -0.0981274),
    rpy=(0.0, 3.14159, 0.0),
    axis=None,
)
"""Fixed joint carrying the tool frame; the 7.9 mm x offset is the only real
coupling between wrist_roll and TCP position."""

CHAIN_JOINTS: tuple[JointFrame, ...] = (*ARM_JOINTS, TCP_JOINT)
"""Full serial chain, base_link -> gripper_frame_link."""

ARM_JOINT_NAMES: tuple[str, ...] = tuple(joint.name for joint in ARM_JOINTS)
"""Arm joint names in ``q`` order (``specs.JOINT_NAMES`` minus the gripper)."""

NUM_ARM_JOINTS: int = len(ARM_JOINTS)
"""Length of the ``q`` vector accepted by :meth:`KinematicChain.fk`."""

BASE_LINK: str = specs.LINK_CHAIN[0]
"""Root link; every FK pose is expressed in this frame."""

TCP_LINK: str = specs.GRIPPER_FRAME_LINK
"""Tool centre point frame (see the module docstring for its axis convention)."""

CHAIN_LINKS: tuple[str, ...] = (BASE_LINK, *(joint.child for joint in CHAIN_JOINTS))
"""Every link :meth:`KinematicChain.fk` reports, base outward."""

PITCH_SUM_VERTICAL: float = np.pi / 2
"""``shoulder_lift + elbow_flex + wrist_flex`` (radians) that points the tool
approach axis (gripper_frame_link +Z) straight down along world -Z. The three
joints share one pitch axis, so only their sum sets the tool tilt."""

PITCH_SUM_HORIZONTAL: float = 0.0
"""Pitch sum (radians) that lays the tool approach axis **horizontal**.

Measured by sweeping the sum with FK (``tests/test_kinematics.py`` re-derives
it): the pitch axis is horizontal, so dropping the sum from
:data:`PITCH_SUM_VERTICAL` to 0 swings the approach axis a quarter turn up from
world -Z, and it comes to rest pointing **radially outward** -- at
``shoulder_pan = 0`` it is exactly world +X, and in general its world azimuth is
``-shoulder_pan``. Outward is the useful sense: the wrist ends up
:data:`TCP_TO_WRIST_HORIZONTAL` *behind* the TCP, between the object and the
base, so the hand reaches a standing object from the base's side and the jaws
close across it. (The opposite sense is ``pitch_sum = pi``, which would ask the
arm to reach past the object and grasp back toward itself; nothing does.)"""

TOOL_YAW_OFFSET: float = np.deg2rad(177.211)
"""Constant term of the tool-yaw relation, radians. At any configuration
satisfying :data:`PITCH_SUM_VERTICAL`::

    tool_yaw(q) == TOOL_YAW_OFFSET - shoulder_pan + wrist_roll   (mod 2*pi)

It is 180 deg minus the wrist_roll origin's 0.0486795 rad calibration offset."""

TOOL_ROLL_OFFSET: float = np.deg2rad(-87.211)
"""Constant term of the tool-roll relation, radians. At any configuration
satisfying :data:`PITCH_SUM_HORIZONTAL`::

    tool_roll(q) == TOOL_ROLL_OFFSET - wrist_roll                (mod 2*pi)

The horizontal family's counterpart of :data:`TOOL_YAW_OFFSET`, and it is *not*
that relation with a different constant -- two things change (both measured, see
``tests/test_kinematics.py``):

* **shoulder_pan drops out.** Panning swings the whole arm about the vertical,
  which turns the approach axis but not the tool's rotation *about* it. So the
  wrist_roll a level-jawed side grasp needs is the same everywhere in the
  region, and -- unlike the top-down grasp, where a 40 deg band of tool yaws has
  no representative at a given pan (:func:`_yaw_branch`) -- it is never out of
  reach: ``tool_roll = 0`` wants wrist_roll -87.2 deg and ``tool_roll = pi``
  wants +92.8 deg, both comfortably inside the joint's [-157.2, 162.8] travel.
* **wrist_roll enters with the opposite sign**, because the reference direction
  the roll is measured from (:func:`tool_roll_of`) is built from world +Z rather
  than from the approach axis, so the two conventions wind opposite ways.

Numerically it is -90 deg plus the same 0.0486795 rad wrist_roll calibration
offset that puts the 177.211 in :data:`TOOL_YAW_OFFSET`."""

TCP_TO_WRIST_HORIZONTAL: float = 0.1592
"""How far behind the TCP the wrist_flex axis sits along the approach axis at
:data:`PITCH_SUM_HORIZONTAL`, metres. **Measured by FK.**

Literally the vertical family's 0.159 m *drop* from the wrist to the tool,
tipped on its side: the component along the tool's own +Z does not depend on the
tilt at all, and at the horizontal family the wrist ends up within 0.2 mm of the
tool's own height rather than 159 mm above it. It is the number that makes
:data:`SIDE_GRASP_REGION` a different annulus from :data:`GRASP_REGION` rather
than the same one at a different tool angle -- the arm has to stand a hand's
length back from whatever it is grasping."""


def _rotation_from_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw (radians) as a 3x3 rotation: Rz @ Ry @ Rx."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _transform(xyz: tuple[float, float, float], rotation: np.ndarray) -> np.ndarray:
    """Homogeneous 4x4 from a translation (metres) and a 3x3 rotation."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = xyz
    return matrix


def _rotation_z(angle: float) -> np.ndarray:
    """Homogeneous 4x4 rotation of `angle` radians about the local +Z axis."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cos, -sin, 0.0, 0.0],
            [sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


_ORIGINS: tuple[np.ndarray, ...] = tuple(
    _transform(joint.xyz, _rotation_from_rpy(joint.rpy)) for joint in CHAIN_JOINTS
)
"""Fixed parent->child transform of every chain joint, in :data:`CHAIN_JOINTS`
order. Built once: :class:`KinematicChain` walks them and the IK seed picks
individual ones apart. Treat as read-only."""


def rotation_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    """3x3 rotation from an ``(x, y, z, w)`` quaternion.

    That component order is Isaac Lab 3.x's convention for
    ``body_link_quat_w`` and friends -- 2.x used ``(w, x, y, z)``, so the
    order is spelled out in the name rather than left to the caller's memory.
    """
    x, y, z, w = (float(component) for component in np.asarray(quat, dtype=float))
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("cannot build a rotation from a zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two 3x3 rotations, in degrees.

    Built on ``arctan2`` of the relative rotation's sine and cosine rather than
    ``arccos`` of its trace: arccos halves the precision near zero, which would
    leave a ~1e-3 deg noise floor underneath the 0.1 deg agreement this is
    asked to certify.
    """
    relative = np.asarray(a).T @ np.asarray(b)
    # Antisymmetric part; its norm is |sin(angle)|, its axis the rotation axis.
    sine = 0.5 * np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(np.degrees(np.arctan2(np.linalg.norm(sine), cosine)))


def pitch_sum(q: np.ndarray) -> float:
    """``shoulder_lift + elbow_flex + wrist_flex`` of `q` (radians)."""
    return float(np.asarray(q, dtype=float)[1:4].sum())


def wrist_flex_for_vertical(shoulder_lift: float, elbow_flex: float) -> float:
    """wrist_flex (radians) completing :data:`PITCH_SUM_VERTICAL` for that pair.

    The caller still has to check the result against
    ``specs.JOINT_LIMITS["wrist_flex"]``: most of the lift/elbow plane has no
    reachable vertical-tool solution.
    """
    return PITCH_SUM_VERTICAL - shoulder_lift - elbow_flex


def wrist_flex_for_pitch_sum(
    shoulder_lift: float, elbow_flex: float, target: float = PITCH_SUM_VERTICAL
) -> float:
    """wrist_flex (radians) completing a pitch sum of `target` for that pair.

    :func:`wrist_flex_for_vertical` for any tool tilt, not just straight down;
    same caveat about the joint limit, and more of it -- the horizontal family
    runs its wrist_flex right up against the stop at the inner edge of
    :data:`SIDE_GRASP_REGION`.
    """
    return target - shoulder_lift - elbow_flex


def tool_yaw_of(rotation: np.ndarray) -> float:
    """World azimuth of the tool's +X axis, radians in (-pi, pi].

    The orientation coordinate the **vertical** family is solved in: with the
    approach axis pointing straight down, the tool's +X is horizontal and its
    azimuth is the direction the jaws close along. Degenerate at a horizontal
    approach, where +X is free to point up -- see :func:`tool_roll_of`.
    """
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def horizontal_tool_rotation(approach_azimuth: float, tool_roll: float) -> np.ndarray:
    """The 3x3 TCP rotation :func:`tool_roll_of` inverts, for a horizontal tool.

    The tool pose of the :data:`TOOL_HORIZONTAL` family written out: approach
    axis (+Z) horizontal at `approach_azimuth`, and the jaws' closing axis (+X)
    turned `tool_roll` off level about it. At ``tool_roll = 0`` the columns come
    out as ``(tangential, world +Z, radial)``; at ``pi`` the first two flip.

    What a planner needs it for is the *inverse* problem :func:`ik_solve` does
    not answer: where to stand the tool so a given point ends up at a given
    place between the pads.

    Args:
        approach_azimuth: World azimuth of the approach axis, radians.
        tool_roll: Jaw tilt off level about that axis, radians.

    Returns:
        A (3, 3) rotation whose columns are the tool's X, Y and Z in base_link.
    """
    approach = np.array([np.cos(approach_azimuth), np.sin(approach_azimuth), 0.0])
    level = np.cross((0.0, 0.0, 1.0), approach)
    up = np.cross(approach, level)
    tool_x = np.cos(tool_roll) * level + np.sin(tool_roll) * up
    return np.column_stack([tool_x, np.cross(approach, tool_x), approach])


def tool_roll_of(rotation: np.ndarray) -> float:
    """Tilt of the tool's +X axis off level, about the approach axis, radians.

    The orientation coordinate the **horizontal** family is solved in. With the
    approach axis horizontal, the plane the jaws' closing direction lives in is
    spanned by the tangential direction ``level = zhat x approach`` and world
    +Z; this is the angle from ``level`` to the tool's own +X, measured
    right-handed about the approach axis. So:

    * ``0`` -- jaws closing **horizontally** (tangentially), tool +Y pointing
      world up: the cup grasp,
    * ``pi`` -- the same grasp with the two fingers swapped, tool +Y pointing
      world down,
    * ``+/- pi/2`` -- jaws closing vertically, one pad above the other.

    Why :func:`tool_yaw_of` cannot serve here: with the approach axis
    horizontal the tool's +X is perpendicular to it, so the azimuth of +X is
    the approach azimuth +/- 90 deg *whatever the roll is* -- a two-valued step
    function with a zero derivative, which a damped-least-squares row cannot
    steer on. This one is smooth in wrist_roll everywhere the approach is not
    vertical.

    Raises:
        ValueError: the approach axis is vertical, where "level" is undefined.
    """
    approach = np.asarray(rotation, dtype=float)[:, 2]
    level = np.cross((0.0, 0.0, 1.0), approach)
    norm = float(np.linalg.norm(level))
    if norm < 1e-9:
        raise ValueError("tool roll is undefined for a vertical approach axis")
    level = level / norm
    up = np.cross(approach, level)
    tool_x = np.asarray(rotation, dtype=float)[:, 0]
    return float(np.arctan2(tool_x @ up, tool_x @ level))


class KinematicChain:
    """Forward kinematics of the SO-101 serial chain, base_link outward.

    Stateless and cheap to construct: the fixed origin transforms are built
    once at import (:data:`_ORIGINS`) so :meth:`fk` is a handful of 4x4 products.
    """

    def __init__(self) -> None:
        self.joints: tuple[JointFrame, ...] = CHAIN_JOINTS
        self._origins: tuple[np.ndarray, ...] = _ORIGINS

    def fk(self, q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Pose of every chain link in ``base_link`` coordinates.

        Args:
            q: Shape (5,) arm joint angles in radians, ordered
                :data:`ARM_JOINT_NAMES` (shoulder_pan .. wrist_roll). The
                gripper joint is off-chain and takes no slot.

        Returns:
            ``{link_name: (position, rotation)}`` for each of
            :data:`CHAIN_LINKS`, where position is a (3,) array in metres and
            rotation a (3, 3) matrix. ``base_link`` maps to the identity pose.
        """
        angles = np.asarray(q, dtype=float)
        if angles.shape != (NUM_ARM_JOINTS,):
            raise ValueError(f"q must have shape ({NUM_ARM_JOINTS},), got {angles.shape}")
        remaining = iter(angles)

        poses = {BASE_LINK: (np.zeros(3), np.eye(3))}
        pose = np.eye(4)
        for joint, origin in zip(self.joints, self._origins, strict=True):
            pose = pose @ origin
            if joint.axis is not None:
                pose = pose @ _rotation_z(next(remaining))
            poses[joint.child] = (pose[:3, 3].copy(), pose[:3, :3].copy())
        return poses

    def fk_tcp(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP pose: ``(position, rotation)`` of :data:`TCP_LINK` in base_link."""
        return self.fk(q)[TCP_LINK]

    def tool_yaw(self, q: np.ndarray) -> float:
        """World azimuth of the tool's +X axis, radians in (-pi, pi].

        This is the gripper's rotation about the vertical when the tool points
        down; see :data:`TOOL_YAW_OFFSET` for its closed form there.
        """
        _, rotation = self.fk_tcp(q)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def tool_roll(self, q: np.ndarray) -> float:
        """:func:`tool_roll_of` the TCP pose: the jaws' tilt off level, radians.

        The side grasp's counterpart of :meth:`tool_yaw`; see
        :data:`TOOL_ROLL_OFFSET` for its closed form at a horizontal tool.
        """
        return tool_roll_of(self.fk_tcp(q)[1])

    def wrist_camera_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pose of the wrist POV camera at `q`: ``(position, rotation)``.

        The mount (:data:`manus.specs.WRIST_CAM_POS` /
        :data:`~manus.specs.WRIST_CAM_QUAT_XYZW`) carried through the FK of its
        parent link, so where the camera actually ends up in the world is a
        property of a *plan* and can be checked without an Isaac app.

        The rotation is in the camera's own convention
        (:data:`manus.specs.WRIST_CAM_CONVENTION`, i.e. OpenGL's), so its
        columns are the image's **right**, the image's **up**, and *minus* the
        view direction. Three things a POV waypoint has to get right read
        straight off it: ``position[2] > 0`` (the camera is above the table
        rather than buried under it), ``-rotation[2, 2] < 0`` (it looks down at
        the work rather than up at the sky) and the sign of ``rotation[2, 1]``
        (which way up the image is).
        """
        position, rotation = self.fk(q)[specs.WRIST_CAM_PARENT_LINK]
        mount = rotation_from_quat_xyzw(np.array(specs.WRIST_CAM_QUAT_XYZW))
        return position + rotation @ np.array(specs.WRIST_CAM_POS), rotation @ mount

    def approach_azimuth(self, q: np.ndarray) -> float:
        """World azimuth of the tool's approach axis (+Z), radians.

        Meaningful for the horizontal family, where the approach lies in the
        table plane and points radially outward: it comes out at exactly
        ``-shoulder_pan`` (measured, ``tests/test_kinematics.py``). Nothing
        pins it -- the arm has five joints and the side grasp spends all of
        them on position, pitch sum and roll -- so a caller that needs the
        object to end up *between* the pads has to close the loop on the
        azimuth the solver actually delivered.
        """
        _, rotation = self.fk_tcp(q)
        return float(np.arctan2(rotation[1, 2], rotation[0, 2]))


# --- Workspace ----------------------------------------------------------------


@dataclass(frozen=True)
class GraspRegion:
    """The patch of table the grasp pipeline is defined over.

    An annulus sector about the shoulder-pan axis, minus a keep-out rectangle
    over the robot's own base. The radii come from the vertical-tool geometry:
    the wrist_flex limit caps vertical-tool TCP height at 0.0903 m, and with a
    5 deg joint margin only this band serves both the grasp and its +3 cm hover
    waypoint. The azimuth is capped by shoulder_pan travel -- see
    :data:`GRASP_REGION`.

    Attributes:
        pan_axis_xy: World (x, y) of the shoulder-pan axis, metres; radius and
            azimuth are measured about it.
        radius: (min, max) radius of the annulus, metres.
        azimuth_max_deg: Half-width of the sector about world +x, degrees.
        keepout_x: (min, max) world x of the base keep-out rectangle, metres.
        keepout_abs_y: Half-extent of that rectangle along world y, metres.
    """

    pan_axis_xy: tuple[float, float]
    radius: tuple[float, float]
    azimuth_max_deg: float
    keepout_x: tuple[float, float]
    keepout_abs_y: float

    def polar(self, x: float, y: float) -> tuple[float, float]:
        """``(radius [m], azimuth [rad])`` of world (x, y) about the pan axis."""
        dx, dy = x - self.pan_axis_xy[0], y - self.pan_axis_xy[1]
        return float(np.hypot(dx, dy)), float(np.arctan2(dy, dx))

    def in_annulus(self, x: float, y: float) -> bool:
        """Whether world (x, y) [m] lies in the reachable annulus sector.

        Ignores the base keep-out; see :meth:`in_keepout` and :meth:`contains`.
        """
        radius, azimuth = self.polar(x, y)
        lower, upper = self.radius
        return bool(
            lower <= radius <= upper and abs(np.degrees(azimuth)) <= self.azimuth_max_deg
        )

    def in_keepout(self, x: float, y: float) -> bool:
        """Whether world (x, y) [m] falls inside the robot base footprint.

        A conservative axis-aligned rectangle, not the true base outline: an
        object placed here would be underneath or inside the arm's own pedestal
        rather than in front of it, so no grasp is meaningful regardless of IK
        feasibility. Deliberately larger than the footprint to leave clearance
        for the object's own half-width and for the jaws swinging in.
        """
        return bool(self.keepout_x[0] <= x <= self.keepout_x[1] and abs(y) <= self.keepout_abs_y)

    def contains(self, x: float, y: float) -> bool:
        """Whether world (x, y) [m] is a valid placement: in sector, off the base."""
        return self.in_annulus(x, y) and not self.in_keepout(x, y)


GRASP_REGION = GraspRegion(
    pan_axis_xy=(0.0388353, 0.0),
    radius=(0.111, 0.220),
    # shoulder_pan travels +/-110.0 deg, which is exactly the azimuth the hover
    # band was drawn to -- zero margin. The TCP hangs 7.9 mm off the wrist_roll
    # axis, so reaching a given (x, y) costs up to asin(0.0079 / 0.111) =
    # 4.08 deg of extra pan depending on the grasp yaw, leaving 105.92 deg;
    # 105 is that rounded down. Step 4 confirmed it by sweeping: the edge is
    # solvable at 106 and dead at 107, and the untrimmed 110 deg version loses
    # ~1.4% of the region, all of it beyond |azimuth| 107.3.
    azimuth_max_deg=105.0,
    keepout_x=(-0.12, 0.14),
    keepout_abs_y=0.10,
)
"""The top-down workspace; :mod:`manus.randomize` samples placements in it."""

SIDE_GRASP_REGION = GraspRegion(
    pan_axis_xy=(0.0388353, 0.0),
    # Both edges swept with ik_solve over the horizontal family, planning the
    # 30 mm cylinder's side grasp *and* its 40 mm retracted pregrasp at every
    # azimuth (tests/test_kinematics.py re-derives them):
    #
    #   inner  the hand is TCP_TO_WRIST_HORIZONTAL long, so reaching closer in
    #          means folding the forearm further back, and at r = 0.326 m the
    #          pregrasp sits exactly on wrist_flex's -95 deg stop. 0.352 is the
    #          first millimetre with >= 10 deg of travel left on every joint at
    #          both waypoints -- the same order of margin GRASP_REGION's own
    #          edges carry (11.8 deg inner, 11.6 deg outer, measured).
    #   outer  the arm runs out of reach at 0.434 m, and it runs out *fast*:
    #          near full extension the joint-margin measure is useless (still
    #          11 deg at 0.432, infeasible at 0.434), so this edge is a reach
    #          margin instead -- 10 mm of radius in hand before the plan stops
    #          solving at all.
    #
    # Both edges moved out ~6 mm and ~2 mm when the side grasp was **raised**
    # to the cup height (manus.expert.grasp_height: 30 -> 40 mm of TCP height,
    # which is what buys the hand its table clearance): standing the hand higher
    # unfolds the shoulder a little, which is worth reach at both ends. The band
    # is re-derived at the height it is actually used at rather than kept at the
    # old one, since a placement sampler that draws outside it draws attempts
    # the planner cannot serve.
    #
    # The band does not overlap GRASP_REGION at all, and that is the point: a
    # side grasp stands the whole hand out along the table instead of hanging
    # it down, which is worth ~160 mm of reach.
    radius=(0.352, 0.422),
    # The same cap as GRASP_REGION, and for the same reason -- shoulder_pan's
    # +/-110 deg minus what the tool's lateral stand-off costs. The stand-off is
    # the same 17 mm but the radius is twice as big, so it costs half as much
    # (2.4 deg); the measured edge dies between 108 and 110 deg. Sharing the
    # number leaves 3.6 deg rather than the top-down region's ~2, and keeps one
    # azimuth convention across both modes.
    #
    # The stand-off points to whichever side of the object the *static* jaw is
    # on, so flipping manus.expert.SIDE_GRASP_ROLL mirrors which end of the
    # sector is the tight one -- 3.6 deg of pan left at +105 deg and 6.4 at
    # -105 at the level roll the pipeline plans at, the mirror image of the
    # other branch's 3.7/6.3 (measured, tests/test_kinematics.py). The cap is
    # symmetric because the region is, and it is the tighter end that sets it.
    azimuth_max_deg=105.0,
    # Carried over unchanged and never reached: the innermost point of this
    # annulus that is inside the keep-out's x band sits at |y| = 0.34 m, three
    # times the rectangle's half-width. Kept so both regions are the same
    # dataclass with the same membership rules.
    keepout_x=(-0.12, 0.14),
    keepout_abs_y=0.10,
)
"""Where a **side** grasp is defined: an annulus a hand's length further out.

:data:`GRASP_REGION`'s counterpart for the :data:`TOOL_HORIZONTAL` family. See
the comments in its construction for how each edge was measured."""

GRASP_REGIONS: dict[str, GraspRegion] = {"top": GRASP_REGION, "side": SIDE_GRASP_REGION}
"""Placement region per :attr:`~manus.objects.ObjectSpec.grasp_mode`.

The one mapping from a grasp mode to the patch of table it is defined over;
:func:`manus.randomize.placement_region` is what the samplers reach it through.
"""

# Flat aliases -- the region's parts are referred to by name all over the
# pipeline (and by manus.randomize, which re-exports them).
PAN_AXIS_XY: tuple[float, float] = GRASP_REGION.pan_axis_xy
REGION_R: tuple[float, float] = GRASP_REGION.radius
REGION_AZ_DEG: float = GRASP_REGION.azimuth_max_deg
BASE_KEEPOUT_X: tuple[float, float] = GRASP_REGION.keepout_x
BASE_KEEPOUT_ABS_Y: float = GRASP_REGION.keepout_abs_y
in_grasp_region = GRASP_REGION.in_annulus
in_base_keepout = GRASP_REGION.in_keepout

TCP_TO_PAD_CENTRE: float = 0.004
"""Height from the jaw pads' centre up to the TCP, metres.

To grasp an object whose grasp height is ``z``, drive the TCP to
``z + TCP_TO_PAD_CENTRE``. **TUNED at Step 7** (was 0.007, the geometric first
guess). Measured off the vendored jaw meshes: the fingertips reach 6.3 mm past
the TCP along the approach axis and the inner faces only span the last ~20 mm,
so a 30 mm object resting on the ground is gripped near the tips. 4 mm splits
the difference -- the fingertips clear the ground by 12.7 mm, so gravity droop
cannot plant them in the table, while 17 mm of the object's height sits between
the pads. :mod:`manus.expert` carries the *lateral* half of the same geometry
(the jaws are centred on the wrist_roll axis, not on the TCP), which is the
larger correction of the two."""


# --- Inverse kinematics -------------------------------------------------------

_CHAIN = KinematicChain()
"""Shared FK instance for the IK helpers below (the chain is stateless)."""

_WRIST_ORIGIN_LINK: str = ARM_JOINTS[3].child
"""Link whose origin the planar 2R sub-problem places: the wrist_flex axis."""

_ARM_LOWER: np.ndarray = np.array([specs.JOINT_LIMITS[name][0] for name in ARM_JOINT_NAMES])
_ARM_UPPER: np.ndarray = np.array([specs.JOINT_LIMITS[name][1] for name in ARM_JOINT_NAMES])

_SEED_PASSES: int = 3
"""Fixed-point passes coupling the wrist_roll branch to the pan/2R sub-problem."""

_JACOBIAN_STEP: float = 1e-6
"""Central-difference step for the numeric Jacobian, radians."""

_DAMPING_INIT: float = 1e-4
_DAMPING_MIN: float = 1e-12
_DAMPING_MAX: float = 1e4
"""Levenberg-Marquardt damping schedule: start near Gauss-Newton, back off by
:data:`_DAMPING_UP` on a rejected step, tighten by :data:`_DAMPING_DOWN` on an
accepted one, and give up once damping saturates (the residual left is a joint
limit, not a step-size problem)."""

_DAMPING_UP: float = 4.0
_DAMPING_DOWN: float = 0.5

_MIN_PROGRESS: float = 1e-12
"""Residual-norm improvement below which the refinement has stalled."""


def _wrap(angle: float) -> float:
    """Fold an angle (radians) into [-pi, pi)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def grasp_yaw_error(actual_yaw: float, target_yaw: float) -> float:
    """Signed tool-yaw error folded into [-pi/2, pi/2), radians.

    Parallel jaws are pi-periodic: half a turn of wrist_roll swaps the two
    fingers and grasps identically, so a yaw and its opposite are the same
    request. :func:`ik_solve` answers with whichever of the two the wrist_roll
    travel can reach, and this is the error metric that says so.
    """
    return float((actual_yaw - target_yaw + np.pi / 2) % np.pi - np.pi / 2)


def _vertical_roll_for(target_yaw: float, pan: float) -> float:
    """wrist_roll giving tool yaw `target_yaw` at `pan`, from :data:`TOOL_YAW_OFFSET`."""
    return _wrap(target_yaw - TOOL_YAW_OFFSET + pan)


def _horizontal_roll_for(target_roll: float, pan: float) -> float:
    """wrist_roll giving tool roll `target_roll`, from :data:`TOOL_ROLL_OFFSET`.

    `pan` is accepted and ignored: panning turns the approach axis but not the
    tool's rotation about it, so a level-jawed side grasp wants the same
    wrist_roll everywhere in the region.
    """
    del pan
    return _wrap(TOOL_ROLL_OFFSET - target_roll)


def _tilt_from_down(rotation: np.ndarray) -> float:
    """Angle between the tool approach axis and world -Z, radians."""
    return float(np.arccos(np.clip(-rotation[2, 2], -1.0, 1.0)))


def _tilt_from_level(rotation: np.ndarray) -> float:
    """Angle between the tool approach axis and the horizontal plane, radians."""
    return float(abs(np.arcsin(np.clip(rotation[2, 2], -1.0, 1.0))))


@dataclass(frozen=True)
class ToolFamily:
    """One tool-orientation family: a pitch sum plus its leftover-freedom convention.

    The arm has five joints; a grasp pose spends three on position and one on
    the pitch sum (which is the tool tilt, since the three pitch joints share an
    axis). Exactly one freedom is left, and *what it means* depends on the tilt:
    with the tool pointing down it is the azimuth the jaws close along
    (:func:`tool_yaw_of`), and with the tool horizontal that azimuth is pinned
    by the arm's own reach direction and the freedom is instead the roll of the
    closing plane about the approach axis (:func:`tool_roll_of`). A bare pitch
    sum is therefore *not* enough to generalise :func:`ik_solve` -- the fifth
    residual row has to change with it, or it goes flat and the solver stops
    steering wrist_roll at all.

    Attributes:
        name: Human label, used in error messages.
        pitch_sum: ``shoulder_lift + elbow_flex + wrist_flex`` this family holds.
        measure: The leftover orientation coordinate, read off a 3x3 TCP
            rotation. This is what ``target_yaw`` means to :func:`ik_solve`, and
            it is pi-periodic in the same way (:func:`grasp_yaw_error`), because
            half a turn about the approach axis swaps two parallel jaws.
        tilt: Independent check on the pitch-sum row: how far the tool's own
            approach axis is off this family's cone, radians.
        roll_for: ``(target_angle, pan) -> wrist_roll`` inverting the family's
            angle relation, ignoring the joint limits (:func:`_yaw_branch`
            applies those).
        pan_and_roll: ``(target_pos, target_angle, family) -> (pan, roll,
            angle_used)`` -- the closed-form half of :func:`analytic_seed`,
            which differs between the families (see the two implementations).
    """

    name: str
    pitch_sum: float
    measure: Callable[[np.ndarray], float]
    tilt: Callable[[np.ndarray], float]
    roll_for: Callable[[float, float], float]
    pan_and_roll: Callable[[np.ndarray, float, "ToolFamily"], tuple[float, float, float]]


def _pitch_plane_polar(vector: np.ndarray, what: str) -> tuple[float, float]:
    """``(length [m], direction [rad])`` of a link vector in the pitch plane.

    The three pitch joints share one axis, so both arm links are perpendicular
    to it; a non-zero component along it would mean the transcription drifted
    and the planar sub-problem below is no longer planar.
    """
    if abs(vector[2]) > 1e-9:
        raise ValueError(f"{what} leaves the pitch plane: z={vector[2]:.3e} m")
    return float(np.hypot(vector[0], vector[1])), float(np.arctan2(vector[1], vector[0]))


# The planar 2R, in the frame shoulder_lift rotates: the upper arm origin as
# written, and the forearm origin carried through the elbow's fixed twist. The
# lengths come out at 0.1160 m and 0.1350 m, the angles are constant offsets
# between a joint angle and its link's direction.
_UPPER_ARM_LENGTH, _UPPER_ARM_ANGLE = _pitch_plane_polar(
    np.array(ARM_JOINTS[2].xyz), "upper arm (elbow_flex origin)"
)
_FOREARM_LENGTH, _FOREARM_ANGLE = _pitch_plane_polar(
    _rotation_from_rpy(ARM_JOINTS[2].rpy) @ np.array(ARM_JOINTS[3].xyz),
    "forearm (wrist_flex origin)",
)


def _shoulder_frame(pan: float) -> np.ndarray:
    """base_link -> the frame the planar 2R lives in, at ``shoulder_pan = pan``."""
    return _ORIGINS[0] @ _rotation_z(pan) @ _ORIGINS[1]


_HOME_SHOULDER_FRAME = _shoulder_frame(0.0)

_PITCH_AXIS_AZIMUTH: float = float(
    np.arctan2(_HOME_SHOULDER_FRAME[1, 2], _HOME_SHOULDER_FRAME[0, 2])
)
"""World azimuth of the (horizontal) pitch axis at ``shoulder_pan = 0``, radians."""

_ARM_PLANE_OFFSET: float = float(
    _HOME_SHOULDER_FRAME[:3, 2] @ (_HOME_SHOULDER_FRAME[:3, 3] - np.array([*PAN_AXIS_XY, 0.0]))
)
"""Signed distance from the pan axis to the arm's pitch plane, metres (-18.3 mm).

The wrist->gripper origin leans back across it by almost exactly as much, which
is why the wrist_roll axis ends up only 0.18 mm off the pan plane."""


def _pan_for_point(point: np.ndarray, plane_offset: float = _ARM_PLANE_OFFSET) -> float:
    """shoulder_pan swinging a plane `plane_offset` off the pan axis through `point`.

    "Aim at the target's azimuth", corrected for the plane missing the pan axis.
    Positive pan swings the arm to -y, hence the sign. The equation's other root
    reaches the point backwards over the base (pan ~ 180 deg away) and is
    ignored.

    The default offset is :data:`_ARM_PLANE_OFFSET`, i.e. the arm's own pitch
    plane, which is the plane the wrist_flex axis lives in. Passing a different
    one aims a plane *parallel* to it -- the trick :func:`_horizontal_pan_and_roll`
    uses to place the TCP directly instead of iterating on the wrist.
    """
    dx = point[0] - PAN_AXIS_XY[0]
    dy = point[1] - PAN_AXIS_XY[1]
    radius = max(float(np.hypot(dx, dy)), abs(plane_offset))
    lean = float(np.arccos(np.clip(plane_offset / radius, -1.0, 1.0)))
    return _wrap(_PITCH_AXIS_AZIMUTH - float(np.arctan2(dy, dx)) - lean)


def _yaw_branch(
    target_yaw: float, pan: float, family: ToolFamily
) -> tuple[float, float]:
    """``(reachable representative of target_yaw, the wrist_roll giving it)``.

    Inverts the family's own angle relation -- ``tool_yaw = TOOL_YAW_OFFSET -
    pan + wrist_roll`` for :data:`TOOL_VERTICAL`, ``tool_roll = TOOL_ROLL_OFFSET
    - wrist_roll`` for :data:`TOOL_HORIZONTAL`. wrist_roll travels 320 deg,
    40 deg short of a full turn, so at any pan there is a 40 deg band of tool
    angles with no representative -- and for the vertical family it is the band
    where the jaw axis lines up with the reach direction, so it is hit
    constantly. The pi-flipped request is the same physical grasp
    (:func:`grasp_yaw_error`) and always fits, the gap being narrower than half
    a turn. (It is the same flip either way: the two families' relations differ
    in the sign of wrist_roll, and ``+pi`` and ``-pi`` agree modulo a turn.) If
    neither fits -- impossible under the current limits, kept as a guard -- the
    roll is clamped and the caller's tolerance check reports it.
    """
    lower, upper = specs.JOINT_LIMITS["wrist_roll"]
    direct = family.roll_for(target_yaw, pan)
    if lower <= direct <= upper:
        return _wrap(target_yaw), direct
    flipped = _wrap(direct + np.pi)
    if lower <= flipped <= upper:
        return _wrap(target_yaw + np.pi), flipped
    return _wrap(target_yaw), float(np.clip(direct, lower, upper))


def _tcp_from_wrist(pan: float, roll: float, family: ToolFamily) -> np.ndarray:
    """TCP position minus wrist_flex-axis position, at `family`'s tool angle.

    Depends on pan and wrist_roll alone: the three pitch joints reach the tool
    only through their sum, which the family pins. At :data:`TOOL_VERTICAL` it
    is a constant 0.159 m drop plus, horizontally, the 7.9 mm TCP offset from
    the wrist_roll axis swinging with `roll` (and a 0.18 mm residual) -- which
    is the whole of the position/yaw coupling this IK has to resolve. At
    :data:`TOOL_HORIZONTAL` the same 0.159 m is *horizontal*
    (:data:`TCP_TO_WRIST_HORIZONTAL`), which is a far bigger lever and is why
    that family needs a different seed.
    """
    poses = _CHAIN.fk(np.array([pan, 0.0, 0.0, family.pitch_sum, roll]))
    return poses[TCP_LINK][0] - poses[_WRIST_ORIGIN_LINK][0]


def _vertical_pan_and_roll(
    target: np.ndarray, target_yaw: float, family: ToolFamily
) -> tuple[float, float, float]:
    """``(pan, wrist_roll, yaw_used)`` for the tool-vertical family.

    A fixed point, because the three quantities chase each other: the pan aims
    the arm at the *wrist*, the wrist is the target minus the TCP offset, and
    that offset swings with wrist_roll which is set by the pan. The offset's
    horizontal part is only 7.9 mm, so :data:`_SEED_PASSES` passes settle it to
    ~0.02 mm over :data:`GRASP_REGION`.

    Frozen deliberately: these are the numbers the 200-attempt Step 8 gate and
    every committed dataset were produced with, so the exact-plane solve
    :func:`_horizontal_pan_and_roll` uses -- which would serve this family too,
    and to a tighter tolerance -- is not retro-fitted here.
    """
    pan = _pan_for_point(target)
    yaw_used, roll = _yaw_branch(target_yaw, pan, family)
    for _ in range(_SEED_PASSES):
        wrist = target - _tcp_from_wrist(pan, roll, family)
        pan = _pan_for_point(wrist)
        yaw_used, roll = _yaw_branch(target_yaw, pan, family)
    return pan, roll, yaw_used


def _horizontal_pan_and_roll(
    target: np.ndarray, target_roll: float, family: ToolFamily
) -> tuple[float, float, float]:
    """``(pan, wrist_roll, roll_used)`` for the tool-horizontal family, exactly.

    No fixed point, and it could not be one: with the tool horizontal the TCP
    stands :data:`TCP_TO_WRIST_HORIZONTAL` out along the arm's own plane, so
    subtracting that offset leaves a wrist point at a much *smaller* radius,
    where the same lateral error subtends a much larger azimuth. Iterating
    ``pan -> wrist -> pan`` there is a map with gain near one: measured, it
    oscillates in a two-cycle and near the inner edge of the region it diverges.

    So the pan is solved in closed form instead. The wrist_flex axis lies in the
    arm's pitch plane, :data:`_ARM_PLANE_OFFSET` off the pan axis; the TCP lies
    in the plane *parallel* to it through the tool, and the gap between the two
    planes is ``pitch_axis . (tcp - wrist)`` -- which rotates with the arm and
    is therefore a constant, once wrist_roll is known. Since wrist_roll does not
    depend on pan in this family (:data:`TOOL_ROLL_OFFSET`), it is known first,
    and one :func:`_pan_for_point` against the shifted plane places the TCP
    outright. Measured over the region the seed lands within 0.16 um -- two
    orders better than the vertical family's fixed point.
    """
    roll_used, roll = _yaw_branch(target_roll, 0.0, family)
    plane = _ARM_PLANE_OFFSET + float(
        _HOME_SHOULDER_FRAME[:3, 2] @ _tcp_from_wrist(0.0, roll, family)
    )
    return _pan_for_point(target, plane), roll, roll_used


TOOL_VERTICAL = ToolFamily(
    name="vertical",
    pitch_sum=PITCH_SUM_VERTICAL,
    measure=tool_yaw_of,
    tilt=_tilt_from_down,
    roll_for=_vertical_roll_for,
    pan_and_roll=_vertical_pan_and_roll,
)
"""Tool pointing straight down: the top-down grasp, over :data:`GRASP_REGION`.

:func:`ik_solve`'s default, and the family every waypoint in the pipeline was
solved in before side grasps existed."""

TOOL_HORIZONTAL = ToolFamily(
    name="horizontal",
    pitch_sum=PITCH_SUM_HORIZONTAL,
    measure=tool_roll_of,
    tilt=_tilt_from_level,
    roll_for=_horizontal_roll_for,
    pan_and_roll=_horizontal_pan_and_roll,
)
"""Tool laid flat, approach axis radially outward: the side grasp, over
:data:`SIDE_GRASP_REGION`. Its ``target_yaw`` is a :func:`tool_roll_of`, not a
yaw -- 0 or pi is a level-jawed "cup" grasp."""


def ik_errors(
    q: np.ndarray,
    target_pos: np.ndarray,
    target_yaw: float,
    family: ToolFamily = TOOL_VERTICAL,
) -> tuple[float, float, float]:
    """How far `q` misses the grasp pose: (position [m], tilt [rad], angle [rad]).

    All three are magnitudes, in the units :func:`ik_solve`'s tolerances use.
    The tilt is read off the tool's own approach axis rather than off the
    pitch-sum the solver drives, so this is an independent check on a solution;
    the angle error is the pi-periodic :func:`grasp_yaw_error` of whatever
    orientation coordinate `family` measures.
    """
    position, rotation = _CHAIN.fk_tcp(q)
    tilt = family.tilt(rotation)
    yaw = family.measure(rotation)
    return (
        float(np.linalg.norm(position - np.asarray(target_pos, dtype=float))),
        tilt,
        abs(grasp_yaw_error(yaw, target_yaw)),
    )


def _planar_2r(x: float, y: float, elbow_sign: float) -> tuple[float, float]:
    """``(shoulder_lift, elbow_flex)`` putting the wrist_flex axis at (x, y).

    Coordinates are in :func:`_shoulder_frame`, where the arm reduces to two
    fixed-length links at fixed angular offsets. `elbow_sign` selects the two
    mirror solutions (+1 folds the elbow the way the wrist_flex limit allows at
    tool-vertical, -1 is its reflection); an out-of-reach (x, y) falls back to
    the fully stretched or fully folded arm pointing at it.
    """
    reach = float(np.hypot(x, y))
    cosine = (reach**2 - _UPPER_ARM_LENGTH**2 - _FOREARM_LENGTH**2) / (
        2.0 * _UPPER_ARM_LENGTH * _FOREARM_LENGTH
    )
    interior = elbow_sign * float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    lift = (
        float(np.arctan2(y, x))
        - float(
            np.arctan2(
                _FOREARM_LENGTH * np.sin(interior),
                _UPPER_ARM_LENGTH + _FOREARM_LENGTH * np.cos(interior),
            )
        )
        - _UPPER_ARM_ANGLE
    )
    return _wrap(lift), _wrap(interior - (_FOREARM_ANGLE - _UPPER_ARM_ANGLE))


def _limit_violation(q: np.ndarray) -> float:
    """Total distance (radians) by which `q` overruns ``specs.JOINT_LIMITS``."""
    return float(np.sum(np.maximum(0.0, _ARM_LOWER - q) + np.maximum(0.0, q - _ARM_UPPER)))


def analytic_seed(
    target_pos: np.ndarray, target_yaw: float, family: ToolFamily = TOOL_VERTICAL
) -> tuple[np.ndarray, float]:
    """Closed-form guess in `family`: ``(q, the tool angle it actually aims for)``.

    Not clamped to the joint limits -- :func:`ik_solve` does that -- so the raw
    guess stays inspectable. The decomposition is the same either way: solve pan
    and wrist_roll (:attr:`ToolFamily.pan_and_roll`, which is where the two
    families differ), put the wrist_flex axis at the target minus the TCP
    offset, and finish with the planar 2R and the pitch sum.

    Args:
        target_pos: Shape (3,) TCP position in base_link coordinates, metres.
        target_yaw: Desired :attr:`ToolFamily.measure` of the tool, radians --
            a :meth:`KinematicChain.tool_yaw` for :data:`TOOL_VERTICAL`, a
            :meth:`KinematicChain.tool_roll` for :data:`TOOL_HORIZONTAL`.
        family: Tool-orientation family to solve in.

    Returns:
        ``(q, target_yaw_used)`` -- the (5,) guess in radians, and the
        representative of `target_yaw` its wrist_roll aims at, which is
        `target_yaw` itself or its pi-flip (see :func:`_yaw_branch`).
    """
    target = np.asarray(target_pos, dtype=float)
    pan, roll, yaw_used = family.pan_and_roll(target, target_yaw, family)
    wrist = target - _tcp_from_wrist(pan, roll, family)

    planar = np.linalg.solve(_shoulder_frame(pan), np.append(wrist, 1.0))
    best: tuple[float, np.ndarray] | None = None
    for elbow_sign in (1.0, -1.0):
        lift, elbow = _planar_2r(planar[0], planar[1], elbow_sign)
        q = np.array(
            [pan, lift, elbow, wrist_flex_for_pitch_sum(lift, elbow, family.pitch_sum), roll]
        )
        violation = _limit_violation(q)
        if best is None or violation < best[0]:
            best = (violation, q)
    assert best is not None
    return best[1], yaw_used


def _residual(
    q: np.ndarray, target_pos: np.ndarray, target_yaw: float, family: ToolFamily
) -> np.ndarray:
    """The square 5-residual: position (m), pitch-sum (rad), tool angle (rad).

    Pitch-sum stands in for tool tilt because the two are equivalent (the pitch
    joints share an axis) and it is linear in `q`, which the Jacobian likes;
    :func:`ik_errors` re-checks the tilt off the tool axis itself. The last row
    is the family's own orientation coordinate -- see :class:`ToolFamily` for
    why it cannot just be the tool yaw in both.
    """
    position, rotation = _CHAIN.fk_tcp(q)
    return np.array(
        [
            *(position - target_pos),
            pitch_sum(q) - family.pitch_sum,
            _wrap(family.measure(rotation) - target_yaw),
        ]
    )


def _jacobian(
    q: np.ndarray, target_pos: np.ndarray, target_yaw: float, family: ToolFamily
) -> np.ndarray:
    """d(residual)/dq by central differences.

    Numeric rather than analytic: FK is ten cheap 4x4 walks, and an analytic
    Jacobian would be a second transcription of the chain to keep in sync.
    """
    jacobian = np.empty((5, NUM_ARM_JOINTS))
    for index in range(NUM_ARM_JOINTS):
        step = np.zeros(NUM_ARM_JOINTS)
        step[index] = _JACOBIAN_STEP
        jacobian[:, index] = (
            _residual(q + step, target_pos, target_yaw, family)
            - _residual(q - step, target_pos, target_yaw, family)
        ) / (2.0 * _JACOBIAN_STEP)
    return jacobian


def _refine(
    q: np.ndarray,
    target_pos: np.ndarray,
    target_yaw: float,
    solved: Callable[[np.ndarray], bool],
    max_iters: int,
    family: ToolFamily,
) -> np.ndarray:
    """Damped least squares on :func:`_residual`, from `q`, clamped to the limits.

    Levenberg-Marquardt: each iteration retries the step with more damping until
    the residual norm actually drops, then relaxes the damping again. Stops as
    soon as `solved` is happy, when damping saturates, or when a step stops
    buying anything -- all three mean more iterations cannot help, the last two
    because what is left is a joint limit rather than a step-size problem.
    """
    residual = _residual(q, target_pos, target_yaw, family)
    damping = _DAMPING_INIT
    for _ in range(max_iters):
        if solved(q):
            break
        jacobian = _jacobian(q, target_pos, target_yaw, family)
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        norm = float(np.linalg.norm(residual))
        while damping <= _DAMPING_MAX:
            step = np.linalg.solve(normal + damping * np.eye(NUM_ARM_JOINTS), -gradient)
            candidate = np.clip(q + step, _ARM_LOWER, _ARM_UPPER)
            trial = _residual(candidate, target_pos, target_yaw, family)
            if float(np.linalg.norm(trial)) < norm:
                q, residual = candidate, trial
                damping = max(damping * _DAMPING_DOWN, _DAMPING_MIN)
                break
            damping *= _DAMPING_UP
        if norm - float(np.linalg.norm(residual)) < _MIN_PROGRESS:
            break
    return q


def ik_solve(
    target_pos: np.ndarray,
    target_yaw: float,
    q_seed: np.ndarray | None = None,
    *,
    family: ToolFamily = TOOL_VERTICAL,
    max_iters: int = 50,
    pos_tol: float = 1e-3,
    yaw_tol: float = np.deg2rad(1.0),
    tilt_tol: float = np.deg2rad(0.5),
) -> tuple[np.ndarray, bool]:
    """Arm joints placing the TCP at `target_pos` at `family`'s tool orientation.

    :func:`analytic_seed` first, then :func:`_refine`'s damped least squares on
    the square 5-residual system (px, py, pz, pitch-sum, yaw) with every step
    clamped into ``specs.JOINT_LIMITS`` -- so an unreachable request comes back
    as the closest in-limit pose with ``converged=False``, never as a joint the
    articulation cannot hold.

    Over :data:`GRASP_REGION` the seed is already the answer (the decomposition
    is exact once the fixed-point passes have settled the TCP offset, to ~0.02 mm
    worst case), and the refinement runs zero iterations. It earns its keep on
    warm starts and just outside the region, where the seed's elbow branch or a
    clamped joint has to be worked out of.

    `target_yaw` is a grasp axis, not an orientation: if wrist_roll cannot reach
    it, the pi-flipped pose (identical for parallel jaws) is solved instead, and
    convergence is judged with the pi-periodic :func:`grasp_yaw_error`. Callers
    that need the yaw as actually solved can read it back with
    :meth:`KinematicChain.tool_yaw`.

    A `q_seed` is a hint, never a constraint: which of the two yaw branches is
    solved for is decided by the target alone (so repeated calls agree), and if
    refining from the hint fails the analytic seed is tried too. A caller can
    therefore pass its current pose to get the nearest solution without ever
    getting a worse answer than it would have got by passing nothing.

    Args:
        target_pos: Shape (3,) TCP position in base_link coordinates, metres.
        target_yaw: Desired :attr:`ToolFamily.measure` of the tool, radians: a
            :meth:`KinematicChain.tool_yaw` under :data:`TOOL_VERTICAL`, a
            :meth:`KinematicChain.tool_roll` under :data:`TOOL_HORIZONTAL`.
        q_seed: Optional (5,) warm start, radians; None means the analytic seed.
        family: Tool-orientation family (pitch sum *and* the meaning of the
            fifth residual row -- see :class:`ToolFamily`). The default is the
            top-down family, so every pre-existing caller is unchanged.
        max_iters: Damped-least-squares iteration cap, per start.
        pos_tol: Position convergence tolerance, metres.
        yaw_tol: Tool-angle convergence tolerance, radians.
        tilt_tol: Tolerated tool tilt off the family's own cone, radians.

    Returns:
        ``(q, converged)`` -- the (5,) solution in radians, clamped to
        ``specs.JOINT_LIMITS``, and whether all three tolerances were met.
    """
    target = np.asarray(target_pos, dtype=float)
    if target.shape != (3,):
        raise ValueError(f"target_pos must have shape (3,), got {target.shape}")
    warm = None if q_seed is None else np.asarray(q_seed, dtype=float)
    if warm is not None and warm.shape != (NUM_ARM_JOINTS,):
        raise ValueError(f"q_seed must have shape ({NUM_ARM_JOINTS},), got {warm.shape}")

    def solved(q: np.ndarray) -> bool:
        position_error, tilt, yaw_error = ik_errors(q, target, target_yaw, family)
        return position_error < pos_tol and tilt < tilt_tol and yaw_error < yaw_tol

    seed, yaw_used = analytic_seed(target, target_yaw, family)
    starts = [seed] if warm is None else [warm, seed]

    best: tuple[float, np.ndarray] | None = None
    for start in starts:
        q = _refine(
            np.clip(start, _ARM_LOWER, _ARM_UPPER), target, yaw_used, solved, max_iters, family
        )
        if solved(q):
            return q, True
        # Nothing reached the tolerances; keep whichever start got the residual
        # the solver itself minimises lowest, and report the miss.
        score = float(np.linalg.norm(_residual(q, target, yaw_used, family)))
        if best is None or score < best[0]:
            best = (score, q)
    assert best is not None
    return best[1], False
