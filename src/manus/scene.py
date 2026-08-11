"""Interactive scene holding the SO-101 arm on a ground plane.

.. warning::
    Requires a running Isaac Sim app; import only after ``AppLauncher(...)``.
    The scene always carries the wrist camera, so the app must be launched
    with cameras enabled (``AppLauncher`` arg ``enable_cameras=True``, i.e.
    the ``--enable_cameras`` CLI flag) or :class:`Camera` refuses to
    initialise.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from manus import specs
from manus.robot import SO101_CFG

# The URDF->USD converter nests each link's Xform inside its parent's, under a
# single "Geometry" scope -- so the gripper link is six levels deep rather than
# a flat sibling of the articulation root.
_GRIPPER_LINK_PATH = "{ENV_REGEX_NS}/Robot/Geometry/" + "/".join(specs.LINK_CHAIN)

WRIST_CAM_WIDTH = 640
"""Wrist camera image width in pixels (the real UVC module's native mode)."""

WRIST_CAM_HEIGHT = 480
"""Wrist camera image height in pixels."""

# Horizontal aperture and focal length share USD's tenth-of-a-world-unit scale,
# so only their ratio matters: hFOV = 2*atan(aperture / (2*focal)). Solving for
# the module's 77.3 deg hFOV at the stock 20.955 aperture gives
# focal = 20.955 / (2 * tan(77.3 deg / 2)) = 13.10, which reads back as 77.31 deg.
_WRIST_CAM_APERTURE = 20.955
_WRIST_CAM_FOCAL = 13.10


@configclass
class SoArmSceneCfg(InteractiveSceneCfg):
    """Scene with a single SO-101 arm. Entity order is load-bearing: terrain,
    then physics assets, then sensors, then lights."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Wrist POV camera, rigidly parented to the gripper link so it rides the
    # jaws. Pose is the vendor one, taken verbatim from the MuJoCo Menagerie
    # `robotstudio_so101` model (Apache-2.0), which derives from the same
    # vendor MJCF as our URDF: the camera is a direct child of the gripper body
    # with no intermediate mount, at (0, 0.055, -0.045) m and rotated -0.57 rad
    # about the body's +X.
    #
    # MuJoCo cameras look down their own -Z with +Y up, which is exactly
    # Isaac Lab's "opengl" convention, so the euler angle transfers unchanged:
    # quat(-0.57 rad about X) = (sin(-0.285), 0, 0, cos(-0.285)) in this
    # install's (x, y, z, w) component order. That aims the optical axis along
    # (0, -0.5396, -0.8419) in gripper-link coordinates -- the jaws close along
    # the link's -Z and their tips sit at z = -0.101 m, so the camera looks
    # just past the fingertips into the grasp region ~67 mm away.
    wrist_cam: CameraCfg = CameraCfg(
        prim_path=f"{_GRIPPER_LINK_PATH}/wrist_cam",
        update_period=0.0,  # seconds; 0 = re-render whenever .data is read
        width=WRIST_CAM_WIDTH,
        height=WRIST_CAM_HEIGHT,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.055, -0.045),  # metres, gripper_link frame
            rot=(-0.2811575, 0.0, 0.0, 0.9596617),  # (x, y, z, w)
            convention="opengl",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_WRIST_CAM_FOCAL,
            horizontal_aperture=_WRIST_CAM_APERTURE,
            # Near plane well inside the ~67 mm standoff to the fingertips.
            clipping_range=(0.01, 20.0),  # metres
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )
