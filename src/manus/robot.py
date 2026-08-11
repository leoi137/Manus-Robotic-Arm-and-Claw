"""Isaac Lab articulation configuration for the SO-ARM101 (SO-101) arm.

.. warning::
    Importing this module requires a running Isaac Sim app. It pulls in
    ``isaaclab``, whose extensions are only importable once
    :class:`isaaclab.app.AppLauncher` has started the simulator. Import it
    *after* ``AppLauncher(...)``, never at the top of a script. For sim-free
    constants (joint names, limits, gains) import :mod:`manus.specs` instead.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from manus import specs

# Repo root is three parents up from src/manus/robot.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]

SO101_USD_PATH: str = str(
    _REPO_ROOT / "assets" / "so101" / "usd" / "so101_new_calib" / "so101_new_calib.usda"
)
"""Absolute path to the vendored SO-101 USD articulation (fixed base)."""

SO101_CFG: ArticulationCfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=SO101_USD_PATH),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(specs.HOME_POSE)),
    # One implicit PD group covers all six STS3215 servos: they are identical
    # units driven by the same bus. Gains are the vendor's system-identified
    # values from joints_properties.xml (see manus.specs), not datasheet
    # figures. Vendor kv is 0.0, but PhysX needs non-zero damping to stay
    # stable, so the vendor joint damping is used for the D term instead.
    actuators={
        "sts3215": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=specs.STS3215_KP,
            damping=specs.STS3215_DAMPING,
            effort_limit_sim=specs.STS3215_EFFORT_LIMIT,
            velocity_limit_sim=specs.SERVO_VELOCITY_LIMIT,
            armature=specs.STS3215_ARMATURE,
            friction=specs.STS3215_FRICTION,
        )
    },
)
"""SO-101 articulation: home pose at all-zero joints (radians), STS3215 actuators."""
