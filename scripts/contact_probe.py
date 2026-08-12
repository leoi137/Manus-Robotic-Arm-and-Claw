"""Read PhysX's own contact points for the two jaws vs the held object.

Purpose (Phase 6, Steps 19-20): prove *empirically* where the jaws touch the
cube -- before and after the collision fix -- instead of arguing from meshes.
The user-visible symptom is a persistent gap between the **fixed** jaw and the
held cube; the Step 19 geometry analysis says the cause is the URDF->USD
converter's ``convexHull`` approximation, which on the fixed jaw sits up to
**6.6 mm proud** of the visual pad face (moving jaw: ~1.8-2.1 mm). This script
turns that claim into a measurement: it drives one scripted grasp to HOLD and
prints, per jaw, every contact PhysX generated -- position in the TCP frame,
normal, force, and separation -- next to where the *visual* pad face is.

**Step 25 addition -- ``--scan``.** The two sample points above are both taken
*after* the jaws have stopped, which cannot answer the other question the
colliders raise: where does contact *begin*. A puck take showed the object
starting to move at a 58 mm jaw gap when the meshes put the pads on its 40 mm
body, ~17 mm early, which is either a collider that is effectively bigger than
the mesh it was cooked from or a contact that carries no force. So ``--scan``
samples every control step of the named states (SEAT and CLOSE by default) and
separates three onsets that the earlier reading could not: the first contact
PhysX *reports*, the first one that carries *force*, and the first step the
object actually *moves*. See :func:`first_contacts`.

Run it once before applying ``scripts/fix_jaw_collision.py`` and once after.
The number to watch is ``contact x`` in the TCP frame for the fixed jaw:

* before the fix, expect ~-6.6 mm (contacts on the hull's phantom wedge, and
  bunched near the cube's top edge, TCP z ~= -11 mm);
* after the fix, expect ~0.0 mm (the real pad face) spread over the pad.

.. code-block:: bash

    # GPU pre-flight first: nvidia-smi must show >= 6500 MiB free.
    PY=~/isaaclab-env/bin/python

    # baseline (convex hull, as converted)
    $PY scripts/contact_probe.py --headless --label before

    # ... apply scripts/fix_jaw_collision.py, then:
    $PY scripts/contact_probe.py --headless --label after

Each run writes ``runs/contact_probe/<label>.json`` (every contact, both
sample points) and ``runs/contact_probe/<label>_hold.png`` (the wrist frame at
HOLD -- the pixels the user complained about).

Notes on the contact API used here
----------------------------------
Isaac Lab's :class:`~isaaclab.sensors.ContactSensor` only exposes *aggregated*
data (net force, per-filter force matrix, and the **average** contact point).
That is not enough: an average point cannot distinguish "one contact on the
pad" from "two contacts on the wedge". So this script also reaches through to
the underlying PhysX tensor view, ``sensor.contact_view.get_contact_data(dt)``,
which returns per-contact buffers -- grounded in
``omni/physics/tensors/api.py:4986`` (docstring quoted below) and in Isaac
Lab's own consumer at
``isaaclab_physx/sensors/contact_sensor/contact_sensor.py:427``::

    forces, points, normals, separations, counts, start_indices = view.get_contact_data(dt)
    #  forces      (max_contact_data_count, 1)  scalar along the normal, N
    #  points      (max_contact_data_count, 3)  world position, m
    #  normals     (max_contact_data_count, 3)  world unit normal
    #  separations (max_contact_data_count, 1)  m; negative = penetration
    #  counts      (sensor_count, filter_count) contacts for this pair
    #  start_indices (sensor_count, filter_count) offset into the buffers
    # flat sensor index = env_index * num_bodies + body_index

Contact reporting requires ``activate_contact_sensors=True`` on the spawn cfg
(it applies ``PhysxContactReportAPI`` to every rigid body), which the shipped
:class:`~manus.robot.SO101_CFG` and :func:`~manus.objects.ObjectSpec.make_spawn_cfg`
do **not** set -- so this script overrides both spawners locally rather than
changing the pipeline's own configs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the src-layout package importable without installing it; the sim-side
# manus imports stay below AppLauncher, which is what makes isaaclab importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--attempt", type=int, default=0, help="draw index into the namespace")
parser.add_argument(
    "--namespace",
    default="contact_probe",
    help="draw namespace; pass expert_gate to replay a gate attempt exactly",
)
parser.add_argument("--object", default="cube_3cm", help="catalogue key of the object to grasp")
parser.add_argument(
    "--pose",
    type=float,
    nargs=3,
    metavar=("X", "Y", "YAW"),
    help="force one placement (m, m, rad) instead of the draw's",
)
parser.add_argument(
    "--label",
    default="probe",
    help="basename for the json/png this run writes (use before/after)",
)
parser.add_argument(
    "--out-dir",
    default=str(REPO_ROOT / "runs" / "contact_probe"),
    help="where the json and png are written",
)
parser.add_argument(
    "--max-contacts",
    type=int,
    default=64,
    help="per-body contact buffer size handed to the PhysX contact view",
)
parser.add_argument(
    "--hold-samples",
    type=int,
    default=5,
    help="how many control steps of HOLD to sample contacts over",
)
parser.add_argument(
    "--scan",
    default="SEAT,CLOSE",
    help=(
        "comma-separated states to sample contacts on *every* control step of, "
        "or '' to skip. This is what answers 'where does PhysX put first "
        "contact': the close-exit sample only ever sees the end of the story"
    ),
)
parser.add_argument(
    "--contact-force-n",
    type=float,
    default=1e-3,
    help=(
        "force above which a reported contact counts as a *loaded* one. PhysX "
        "reports pairs inside the contact offset with zero force and positive "
        "separation; only the loaded ones move the object"
    ),
)
parser.add_argument(
    "--disturbance-mm",
    type=float,
    default=0.2,
    help="object displacement that counts as the jaws having disturbed it",
)
parser.add_argument("--no-video", action="store_true", help="skip the HOLD wrist frame")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The scene always carries the wrist camera and Camera refuses to initialise
# without sensor rendering -- same contract as every other script here.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, subtract_frame_transforms

from manus import expert as expert_mod
from manus import specs
from manus.expert import (
    CLOSE,
    HOLD,
    ExpertConfig,
    GraspSuccessMonitor,
    ScriptedGraspExpert,
    classify_outcome,
)
from manus.objects import (
    JAW_WIDTH_PER_RAD,
    MEASURED_STALL_30MM_RAD,
    OBJECTS,
    REFERENCE_WIDTH_M,
)
from manus.randomize import draw_episode
from manus.robot import SO101_CFG
from manus.task_scene import DEFAULT_OBJECT_POS, GraspSceneCfg, apply_randomization

PHYSICS_DT = 1.0 / 120.0
"""Simulation step, seconds -- the pipeline's locked physics rate."""

DECIMATION = 4
"""Physics steps per control step, i.e. the expert runs at 30 Hz."""

SETTLE_STEPS = 30
"""Physics steps held at home after a reset, so the object comes to rest."""

MAX_CONTROL_STEPS = 1200
"""Hard ceiling per attempt; the FSM's own budgets stop well short of it."""

_PROBE_SPEC = OBJECTS[args_cli.object]
"""Object spec chosen on the command line; the scene cfg below spawns it."""

FIXED_JAW_BODY = "gripper_link"
"""Body carrying the *static* jaw collider (``wrist_roll_follower_so101_v1``).

The URDF gives ``gripper_link`` two collision meshes -- the servo body and the
printed follower that *is* the fixed finger -- so a body-level contact sensor
here also sees servo contacts. The object can only reach the finger, and the
per-contact positions printed below make that checkable rather than assumed.
"""

MOVING_JAW_BODY = specs.MOVING_JAW_LINK
"""Body carrying the moving jaw collider (one mesh, no ambiguity)."""

TCP_BODY = specs.GRIPPER_FRAME_LINK
"""Tool frame every reported contact is expressed in.

Chosen because it is the frame the expert's jaw constants live in: ``+x``
points at the static jaw, ``+z`` is the approach axis (out of the jaws), the
static jaw's *visual* pad face sits at ``x = JAW_FIXED_FACE_X = 0``, and the
fingertips at ``z = JAW_TIP_Z = +6.3 mm``.
"""


@configclass
class ProbeSceneCfg(GraspSceneCfg):
    """Grasp scene with contact reporting turned on and two jaw contact sensors.

    Three deltas from :class:`~manus.task_scene.GraspSceneCfg`, all required to
    read contacts and none of them changing the physics being measured:

    * the robot spawner gets ``activate_contact_sensors=True`` (applies
      ``PhysxContactReportAPI`` to every rigid body -- without it the sensor
      raises "could not find any bodies with contact reporter API");
    * the object spawner gets the same, so the filtered pair resolves;
    * two body-level :class:`ContactSensorCfg` sensors, each filtered against
      the object so ``get_contact_data`` has a valid sensor/filter pair.
    """

    # Annotated, not a bare assignment: ``configclass`` builds on dataclasses,
    # where an un-annotated override of an inherited field silently keeps the
    # parent's default.
    robot: ArticulationCfg = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=SO101_CFG.spawn.replace(activate_contact_sensors=True),
    )

    # Rebuilt rather than ``.replace``d off the parent: ``configclass`` stores
    # mutable defaults behind a ``default_factory``, so the parent's field is
    # not readable as a plain class attribute.
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=_PROBE_SPEC.make_spawn_cfg().replace(activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(DEFAULT_OBJECT_POS[0], DEFAULT_OBJECT_POS[1], _PROBE_SPEC.spawn_z),
            rot=(0.0, 0.0, 0.0, 1.0),  # (x, y, z, w)
        ),
    )

    fixed_jaw_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/" + "/".join(specs.LINK_CHAIN),
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=args_cli.max_contacts,
        update_period=0.0,
    )

    moving_jaw_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/Geometry/" + "/".join(specs.LINK_CHAIN) + f"/{MOVING_JAW_BODY}"
        ),
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=args_cli.max_contacts,
        update_period=0.0,
    )


SCAN_STATES: tuple[str, ...] = tuple(
    part.strip().upper() for part in args_cli.scan.split(",") if part.strip()
)
"""States sampled on every control step; see ``--scan``."""


def jaw_gap(angle: float) -> float:
    """Opening between the pads at `angle`, metres -- the catalogue's own fit.

    The exact inverse of :func:`~manus.objects.contact_angle_for_width`, so that
    "the meshes say first contact at ``grasp_width``" is true by construction and
    a measured first-contact gap can be compared against the object's width
    directly, in millimetres, with no second fit in the way.
    """
    return REFERENCE_WIDTH_M + (angle - MEASURED_STALL_30MM_RAD) * JAW_WIDTH_PER_RAD


@dataclass
class Contact:
    """One PhysX contact, already moved into the TCP frame."""

    jaw: str
    position: tuple[float, float, float]  # TCP frame, metres
    normal: tuple[float, float, float]  # TCP frame, unit
    force: float  # newtons along the normal
    separation: float  # metres; negative = penetration

    def to_dict(self) -> dict:
        return {
            "jaw": self.jaw,
            "position_tcp_mm": [1e3 * value for value in self.position],
            "normal_tcp": list(self.normal),
            "force_n": self.force,
            "separation_mm": 1e3 * self.separation,
        }


def _numpy(array) -> np.ndarray:
    """Host numpy copy of a warp / torch / numpy array, whatever the frontend."""
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "numpy"):  # warp arrays (the frontend Isaac Lab uses)
        return np.asarray(array.numpy())
    if hasattr(array, "detach"):  # torch
        return array.detach().cpu().numpy()
    return np.asarray(array)


class ContactProbe:
    """Reads per-contact data for one jaw sensor against the object."""

    def __init__(self, sensor, jaw: str, body: str, dt: float) -> None:
        self.sensor = sensor
        self.jaw = jaw
        self.dt = dt
        names = list(sensor.body_names)
        if body not in names:
            raise RuntimeError(
                f"{jaw}: contact sensor resolved bodies {names}, which does not contain "
                f"{body!r}. The prim path or the contact-report activation is wrong."
            )
        self.body_index = names.index(body)
        self.body_names = names
        self.error: str | None = None

    def read(self, tcp_pos: torch.Tensor, tcp_quat: torch.Tensor) -> list[Contact]:
        """Every contact this jaw currently has with the filtered object.

        Never raises: a scan calls this several hundred times, and a contact
        buffer that has moved in some future Isaac Lab is not a reason to lose
        the grasp the run is also measuring. The first failure is reported once,
        in :attr:`error`, and every call after it returns nothing.

        Args:
            tcp_pos: TCP world position, shape (1, 3).
            tcp_quat: TCP world orientation (x, y, z, w), shape (1, 4).
        """
        try:
            return self._read(tcp_pos, tcp_quat)
        except Exception as failure:  # pragma: no cover - diagnostic path
            if self.error is None:
                self.error = f"{type(failure).__name__}: {failure}"
                print(f"  WARNING: {self.jaw} jaw contact read failed: {self.error}")
            return []

    def _read(self, tcp_pos: torch.Tensor, tcp_quat: torch.Tensor) -> list[Contact]:
        view = self.sensor.contact_view
        forces, points, normals, separations, counts, starts = view.get_contact_data(dt=self.dt)
        forces = _numpy(forces).reshape(-1)
        points = _numpy(points).reshape(-1, 3)
        normals = _numpy(normals).reshape(-1, 3)
        separations = _numpy(separations).reshape(-1)
        counts = _numpy(counts).reshape(view.sensor_count, view.filter_count)
        starts = _numpy(starts).reshape(view.sensor_count, view.filter_count)

        # One env, so the flat sensor index is just the body index; the single
        # filter (the object) is column 0.
        flat = self.body_index
        count = int(counts[flat, 0])
        start = int(starts[flat, 0])
        if count <= 0:
            return []

        world_pos = torch.as_tensor(
            points[start : start + count], dtype=torch.float32, device=tcp_pos.device
        )
        world_normal = torch.as_tensor(
            normals[start : start + count], dtype=torch.float32, device=tcp_pos.device
        )
        inverse = quat_inv(tcp_quat).repeat(count, 1)
        local_pos = quat_apply(inverse, world_pos - tcp_pos.repeat(count, 1))
        local_normal = quat_apply(inverse, world_normal)

        out = []
        for index in range(count):
            out.append(
                Contact(
                    jaw=self.jaw,
                    position=tuple(float(v) for v in local_pos[index].cpu()),
                    normal=tuple(float(v) for v in local_normal[index].cpu()),
                    force=float(forces[start + index]),
                    separation=float(separations[start + index]),
                )
            )
        return out

    def aggregate(self) -> dict:
        """Isaac Lab's own aggregated view, as a cross-check on the raw buffers."""
        data = self.sensor.data
        out: dict = {}
        matrix = data.force_matrix_w
        if matrix is not None:
            vector = matrix.torch[0, self.body_index, 0]
            out["force_matrix_n"] = [float(v) for v in vector.cpu()]
            out["force_matrix_norm_n"] = float(torch.linalg.norm(vector))
        position = data.contact_pos_w
        if position is not None:
            out["contact_pos_w_m"] = [float(v) for v in position.torch[0, self.body_index, 0].cpu()]
        return out


class ProbeRunner:
    """Drives one scripted grasp and samples contacts at CLOSE exit and in HOLD."""

    def __init__(self, sim, scene, spec) -> None:
        self.sim = sim
        self.scene = scene
        self.spec = spec
        self.robot = scene["robot"]
        self.object = scene["object"]
        self.camera = scene["wrist_cam"]
        self.dt = sim.get_physics_dt()
        self.device = self.robot.data.joint_pos.torch.device
        assert self.robot.joint_names == list(specs.JOINT_NAMES), (
            f"joint order mismatch: {self.robot.joint_names} != {list(specs.JOINT_NAMES)}"
        )
        self.tcp_index = self.robot.body_names.index(TCP_BODY)
        self.home = torch.tensor(
            [[specs.HOME_POSE[name] for name in specs.JOINT_NAMES]],
            dtype=torch.float32,
            device=self.device,
        )
        # Tolerant, because this script measures two things and only one of them
        # needs the sensors: a contact plumbing failure must not also cost the
        # grasp attempt it is riding on, which is a take on a shared GPU.
        self.probes: list[ContactProbe] = []
        self.probe_errors: dict[str, str] = {}
        for jaw, key, body in (
            ("fixed", "fixed_jaw_contact", FIXED_JAW_BODY),
            ("moving", "moving_jaw_contact", MOVING_JAW_BODY),
        ):
            try:
                self.probes.append(ContactProbe(scene[key], jaw, body, self.dt))
            except Exception as failure:
                self.probe_errors[jaw] = f"{type(failure).__name__}: {failure}"
                print(f"  WARNING: {jaw} jaw contact sensor unavailable: {failure}")

    # -- plumbing ---------------------------------------------------------------

    def measured(self) -> np.ndarray:
        return self.robot.data.joint_pos.torch[0].detach().cpu().numpy().astype(float)

    def tcp_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """TCP world position (1, 3) and orientation (1, 4) as (x, y, z, w)."""
        position = self.robot.data.body_link_pos_w.torch[0, self.tcp_index].unsqueeze(0)
        quaternion = self.robot.data.body_link_quat_w.torch[0, self.tcp_index].unsqueeze(0)
        return position, quaternion

    def object_pose_in_tcp(self) -> tuple[np.ndarray, np.ndarray]:
        """Object origin and orientation expressed in the TCP frame."""
        tcp_pos, tcp_quat = self.tcp_frame()
        obj_pos = self.object.data.root_link_pos_w.torch[0].unsqueeze(0)
        obj_quat = self.object.data.root_link_quat_w.torch[0].unsqueeze(0)
        position, quaternion = subtract_frame_transforms(tcp_pos, tcp_quat, obj_pos, obj_quat)
        return (
            position[0].detach().cpu().numpy().astype(float),
            quaternion[0].detach().cpu().numpy().astype(float),
        )

    def object_pos(self) -> np.ndarray:
        """Object body-origin (x, y, z) in the robot's own frame, metres."""
        world = self.object.data.root_link_pos_w.torch[0]
        return (world - self.scene.env_origins[0]).detach().cpu().numpy().astype(float)

    def object_z(self) -> float:
        return float(self.object_pos()[2])

    def object_tilt_deg(self) -> float:
        """Angle between the object's own +z and world +z, degrees.

        What the cylinder's failure *is*: a one-sided push tips it before it
        slides, and a cylinder tilted even 3 deg is wider than the closing gap
        and gets levered out. Read off the body quaternion (w, x, y, z).
        """
        w, x, y, z = (float(v) for v in self.object.data.root_link_quat_w.torch[0])
        return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))

    def advance(self, render: bool) -> None:
        self.sim.step(render=render)
        self.robot.update(self.dt)
        self.object.update(self.dt)
        for probe in self.probes:
            probe.sensor.update(self.dt, force_recompute=True)

    def reset_episode(self, draw) -> None:
        zeros = torch.zeros_like(self.home)
        self.robot.write_joint_state_to_sim_index(
            position=self.home, velocity=zeros, full_data=True
        )
        self.robot.set_joint_position_target_index(target=self.home)
        apply_randomization(self.scene, draw, self.spec)
        self.scene.reset()
        self.scene.write_data_to_sim()
        for _ in range(SETTLE_STEPS):
            self.advance(render=False)

    # -- sampling ---------------------------------------------------------------

    def sample(self, tag: str) -> dict:
        """Every contact on both jaws right now, plus the object's pose."""
        tcp_pos, tcp_quat = self.tcp_frame()
        contacts: list[Contact] = []
        aggregates: dict[str, dict] = {}
        for probe in self.probes:
            contacts.extend(probe.read(tcp_pos, tcp_quat))
            aggregates[probe.jaw] = probe.aggregate()
        obj_pos, obj_quat = self.object_pose_in_tcp()
        return {
            "tag": tag,
            "gripper_rad": float(self.measured()[specs.JOINT_NAMES.index("gripper")]),
            "object_pos_tcp_mm": [1e3 * float(v) for v in obj_pos],
            "object_quat_tcp_xyzw": [float(v) for v in obj_quat],
            "object_z_mm": 1e3 * self.object_z(),
            "contacts": [contact.to_dict() for contact in contacts],
            "aggregates": aggregates,
        }

    def scan_row(self, state: str, step: int) -> dict:
        """One compact per-step row: the jaw angle, both jaws' contacts, the object.

        Deliberately small -- a scan runs for every control step of CLOSE, which
        on a creeping object is 200+ of them, and the interesting quantity is
        per-jaw *aggregate* (how many contacts, how hard, how close) rather than
        every contact's position. The first row that shows contact keeps its
        contacts in full; see :func:`first_contacts`.
        """
        tcp_pos, tcp_quat = self.tcp_frame()
        gripper = float(self.measured()[specs.JOINT_NAMES.index("gripper")])
        row: dict = {
            "state": state,
            "step": step,
            "gripper_rad": gripper,
            "jaw_gap_mm": 1e3 * jaw_gap(gripper),
            "object_pos_m": [float(v) for v in self.object_pos()],
            "jaws": {},
        }
        for probe in self.probes:
            contacts = probe.read(tcp_pos, tcp_quat)
            loaded = [c for c in contacts if c.force > args_cli.contact_force_n]
            row["jaws"][probe.jaw] = {
                "contacts": len(contacts),
                "loaded": len(loaded),
                "force_n": sum(c.force for c in contacts),
                "min_separation_mm": (
                    min(1e3 * c.separation for c in contacts) if contacts else None
                ),
                "max_separation_mm": (
                    max(1e3 * c.separation for c in contacts) if contacts else None
                ),
                "x_mm": (
                    [min(1e3 * c.position[0] for c in contacts),
                     max(1e3 * c.position[0] for c in contacts)]
                    if contacts
                    else None
                ),
                "z_mm": (
                    [min(1e3 * c.position[2] for c in contacts),
                     max(1e3 * c.position[2] for c in contacts)]
                    if contacts
                    else None
                ),
                "detail": [c.to_dict() for c in contacts] if contacts else [],
            }
        return row

    # -- one probe run -----------------------------------------------------------

    def run(self, draw) -> dict:
        self.reset_episode(draw)
        measured = self.measured()
        expert = ScriptedGraspExpert(self.spec, config=ExpertConfig())
        plan = expert.reset(draw, q_current=measured)
        monitor = GraspSuccessMonitor(self.spec)
        print(
            f"  plan: grasp_yaw={math.degrees(plan.grasp_yaw):+.1f} deg  "
            f"close_target={plan.close_target:.3f} rad"
            + ("" if plan.ok else f"  INFEASIBLE: {plan.reason}")
        )

        samples: list[dict] = []
        scan: list[dict] = []
        motion: dict[str, dict] = {}
        seen = expert.state
        hold_seen = 0
        frame = None
        rest = self.object_pos()
        for _ in range(MAX_CONTROL_STEPS):
            if expert.done:
                break
            state, state_step = expert.state, expert.state_step
            targets = expert.step(measured)
            command = torch.tensor(
                [[targets[name] for name in specs.JOINT_NAMES]],
                dtype=torch.float32,
                device=self.device,
            )
            self.robot.set_joint_position_target_index(target=command)
            self.scene.write_data_to_sim()
            for _ in range(DECIMATION):
                self.advance(render=False)
            measured = self.measured()
            position = self.object_pos()
            monitor.update(position, measured)
            if state in SCAN_STATES:
                scan.append(self.scan_row(state, state_step))

            record = motion.setdefault(
                state,
                {
                    "start": [float(v) for v in position],
                    "start_tilt_deg": self.object_tilt_deg(),
                    "max_dxy_mm": 0.0,
                    "max_rise_mm": 0.0,
                    "max_tilt_deg": 0.0,
                },
            )
            delta = position - np.asarray(record["start"], dtype=float)
            record["max_dxy_mm"] = max(
                record["max_dxy_mm"], 1e3 * float(np.hypot(delta[0], delta[1]))
            )
            record["max_rise_mm"] = max(record["max_rise_mm"], 1e3 * float(delta[2]))
            record["max_tilt_deg"] = max(
                record["max_tilt_deg"],
                abs(self.object_tilt_deg() - record["start_tilt_deg"]),
            )

            if expert.state != seen:
                report = expert.reports[-1]
                print(
                    f"    {report.state:<8} {report.exit:<9} {report.steps:3d} steps  "
                    f"grip {report.gripper:.3f} rad"
                )
                if report.state == CLOSE:
                    # The instant the jaws stall: contact geometry with no
                    # LIFT dynamics mixed in.
                    samples.append(self.sample("close_exit"))
                seen = expert.state

            if expert.state == HOLD and hold_seen < args_cli.hold_samples:
                hold_seen += 1
                samples.append(self.sample(f"hold_{hold_seen}"))
                if frame is None and not args_cli.no_video:
                    self.camera.update(self.dt * DECIMATION)
                    frame = (
                        self.camera.data.output["rgb"]
                        .torch[0, ..., :3]
                        .to(torch.uint8)
                        .cpu()
                        .numpy()
                    )

        outcome = classify_outcome(expert, monitor)
        print(
            f"\n  {outcome.upper()}  held {monitor.best_streak}/{monitor.sustain}  "
            f"peak z {monitor.peak_z * 1e3:.1f} mm (bar {monitor.threshold_z * 1e3:.1f})  "
            f"stall {monitor.gripper:.3f} rad in band "
            f"[{monitor.stall_band[0]:.3f}, {monitor.stall_band[1]:.3f}]  "
            f"steps {expert.total_steps}"
            + (f"  timeouts {expert.timeouts}" if expert.timeouts else "")
        )
        for state, record in motion.items():
            print(
                f"    {state:<9} object dxy {record['max_dxy_mm']:6.2f} mm  "
                f"rise {record['max_rise_mm']:7.2f} mm  "
                f"tilt {record['max_tilt_deg']:6.2f} deg"
            )
        return {
            "label": args_cli.label,
            "object": self.spec.name,
            "outcome": outcome,
            "success": bool(monitor.success),
            "rest_z": float(rest[2]),
            "draw": draw.to_dict(),
            "monitor": monitor.to_dict(),
            "object_motion": motion,
            "telemetry": expert.telemetry(),
            "samples": samples,
            "scan": scan,
            "frame": frame,
        }


def first_contacts(scan: list[dict], spec) -> dict:
    """Where the jaws first touch, per jaw, against what the meshes predict.

    Three onsets, and the gap between them is the whole question this run was
    booted to answer:

    ``reported``
        the first step PhysX hands back *any* contact for the pair. PhysX
        generates contact pairs for everything inside the shapes' contact
        offset, and those come back with a **positive separation and no force**
        -- geometry, not physics. A reported contact well before the meshes
        touch is expected and means nothing on its own.
    ``loaded``
        the first step a contact carries force above ``--contact-force-n``.
        *This* is the one that can move the object, and this is the one to
        compare against the mesh: the pads reach the object's width at
        ``contact_angle_for_width(width)``, so a loaded contact at a jaw gap
        materially wider than the object's own width is the collider being
        effectively bigger than the mesh it was cooked from.
    ``disturbed``
        the first step the object has moved ``--disturbance-mm`` from where it
        stood when the scan began. If this leads *loaded*, whatever moved the
        object was not the jaw touching it.
    """
    out: dict = {
        "object_width_mm": 1e3 * spec.grasp_width_m,
        "mesh_contact_angle_rad": spec.contact_angle_rad,
        "mesh_contact_gap_mm": 1e3 * jaw_gap(spec.contact_angle_rad),
        "force_threshold_n": args_cli.contact_force_n,
        "jaws": {},
    }
    if not scan:
        return out
    origin = np.asarray(scan[0]["object_pos_m"], dtype=float)
    for jaw in ("fixed", "moving"):
        entry: dict = {}
        for key, test in (
            ("reported", lambda row: row["jaws"][jaw]["contacts"] > 0),
            ("loaded", lambda row: row["jaws"][jaw]["loaded"] > 0),
        ):
            hit = next((row for row in scan if test(row)), None)
            entry[key] = (
                None
                if hit is None
                else {
                    "state": hit["state"],
                    "step": hit["step"],
                    "gripper_rad": hit["gripper_rad"],
                    "jaw_gap_mm": hit["jaw_gap_mm"],
                    "lead_over_mesh_mm": hit["jaw_gap_mm"]
                    - 1e3 * jaw_gap(spec.contact_angle_rad),
                    "separation_mm": [
                        hit["jaws"][jaw]["min_separation_mm"],
                        hit["jaws"][jaw]["max_separation_mm"],
                    ],
                    "force_n": hit["jaws"][jaw]["force_n"],
                    "x_mm": hit["jaws"][jaw]["x_mm"],
                    "z_mm": hit["jaws"][jaw]["z_mm"],
                    "contacts": hit["jaws"][jaw]["detail"],
                }
            )
        out["jaws"][jaw] = entry
    moved = next(
        (
            row
            for row in scan
            if 1e3 * float(np.linalg.norm(np.asarray(row["object_pos_m"]) - origin))
            > args_cli.disturbance_mm
        ),
        None,
    )
    out["disturbed"] = (
        None
        if moved is None
        else {
            "state": moved["state"],
            "step": moved["step"],
            "gripper_rad": moved["gripper_rad"],
            "jaw_gap_mm": moved["jaw_gap_mm"],
            "lead_over_mesh_mm": moved["jaw_gap_mm"] - 1e3 * jaw_gap(spec.contact_angle_rad),
        }
    )
    return out


CONTACT_REPORT_SCHEMA = "PhysxContactReportAPI"
"""Applied schema a prim needs before a :class:`ContactSensor` will accept it."""


def ensure_contact_reporting(scene, paths: dict[str, str]) -> dict:
    """Apply the contact-report API to the prims the sensors actually watch.

    **This is why the shipped ``activate_contact_sensors=True`` is not enough on
    this robot.** Isaac Lab's spawner does call
    :func:`isaaclab.sim.schemas.activate_contact_sensors`, and that helper walks
    the tree looking for rigid bodies -- but it stops descending the moment it
    finds one, on the documented assumption that "nested rigid bodies are not
    allowed by SDK". This asset *is* nested: the URDF->USD converter puts every
    link inside its parent, so the eight rigid bodies form a chain from
    ``base_link`` down to the two jaws (verified by opening the USD directly).
    The helper therefore applies the API to ``base_link``, stops, and reports
    success, and the sensor at ``gripper_link`` then fails initialisation with
    "could not find any bodies with contact reporter API" -- from inside a
    physics callback, during ``sim.reset()``, where it cannot be caught usefully.

    So the API is applied here instead, by path, after the scene has spawned and
    before the sensors initialise. Every prim is checked rather than assumed,
    and what was found is returned so the run's json records it.
    """
    from pxr import Sdf, Usd, UsdPhysics

    # The scene's own stage and its own env paths, not a global lookup helper:
    # ``isaacsim.core.utils.stage`` does not exist in this Isaac Sim (6.0.1),
    # and reaching for it cost a boot. :class:`InteractiveScene` carries both
    # (``interactive_scene.py`` builds ``env_prim_paths`` at construction and
    # holds ``stage``), so there is nothing to look up.
    stage = scene.stage
    root = str(scene.env_prim_paths[0])

    found: dict[str, str] = {}
    for name, template in paths.items():
        path = template.replace("{ENV_REGEX_NS}", root)
        prim: Usd.Prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            found[name] = f"MISSING {path}"
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            found[name] = f"NOT A RIGID BODY {path}"
            continue
        if CONTACT_REPORT_SCHEMA not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema(CONTACT_REPORT_SCHEMA)
        attribute = prim.GetAttribute("physxContactReport:threshold")
        if not attribute:
            attribute = prim.CreateAttribute(
                "physxContactReport:threshold", Sdf.ValueTypeNames.Float, False
            )
        attribute.Set(0.0)
        found[name] = f"reporting {path}"
    for name, state in found.items():
        print(f"  contact reporting: {name:<10} {state}")
    return found


def collider_parameters(stage) -> dict:
    """The collision attributes actually authored on the two jaws and the object.

    Read off the live stage rather than off ``scripts/fix_jaw_collision.py``'s
    intent, because the question a probe exists to settle is what PhysX is
    using. ``contactOffset`` is the one that decides how early a *reported*
    contact appears; the SDF attributes decide where the surface is.

    Best effort: a missing attribute means "PhysX default", and any failure to
    traverse the stage is reported rather than raised -- the contact numbers are
    the run's real payload and are not worth losing to a USD schema change.
    """
    try:
        from pxr import Usd

        wanted = (
            "physxCollision:contactOffset",
            "physxCollision:restOffset",
            "physics:approximation",
            "physxSDFMeshCollision:sdfResolution",
            "physxSDFMeshCollision:sdfSubgridResolution",
            "physxSDFMeshCollision:sdfNarrowBandThickness",
            "physxSDFMeshCollision:sdfMargin",
        )
        found: dict[str, dict] = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
            if not prim.HasAPI("PhysicsCollisionAPI") and not prim.GetAttribute(
                "physics:collisionEnabled"
            ):
                continue
            values = {
                name: prim.GetAttribute(name).Get()
                for name in wanted
                if prim.HasAttribute(name)
            }
            if values:
                found[str(prim.GetPath())] = values
        return found or {"note": "no collision attributes authored; PhysX defaults apply"}
    except Exception as error:  # pragma: no cover - diagnostic only
        return {"error": f"{type(error).__name__}: {error}"}


def report(samples: list[dict]) -> None:
    """Print the per-jaw contact table -- the whole point of the script."""
    face = expert_mod.JAW_FIXED_FACE_X
    print(
        "\n  Reference geometry (TCP frame, mm): static jaw VISUAL pad face x = "
        f"{1e3 * face:.2f}, fingertips z = {1e3 * expert_mod.JAW_TIP_Z:.2f}, "
        f"planned object centre x = {1e3 * expert_mod.pad_lateral_offset(OBJECTS[args_cli.object]):.2f}"
    )
    for sample in samples:
        print(f"\n  [{sample['tag']}]  gripper {sample['gripper_rad']:.3f} rad")
        obj = sample["object_pos_tcp_mm"]
        print(
            f"    object centre in TCP frame: x {obj[0]:+7.2f}  y {obj[1]:+7.2f}  "
            f"z {obj[2]:+7.2f} mm   (height {sample['object_z_mm']:.1f} mm)"
        )
        for jaw in ("fixed", "moving"):
            rows = [row for row in sample["contacts"] if row["jaw"] == jaw]
            if not rows:
                print(f"    {jaw:<6} NO CONTACTS")
                continue
            xs = [row["position_tcp_mm"][0] for row in rows]
            zs = [row["position_tcp_mm"][2] for row in rows]
            seps = [row["separation_mm"] for row in rows]
            total = sum(row["force_n"] for row in rows)
            print(
                f"    {jaw:<6} {len(rows):2d} contacts  |F| sum {total:7.3f} N  "
                f"x [{min(xs):+7.2f} .. {max(xs):+7.2f}] mm  "
                f"z [{min(zs):+7.2f} .. {max(zs):+7.2f}] mm  "
                f"separation [{min(seps):+6.3f} .. {max(seps):+6.3f}] mm"
            )
            for row in rows:
                px, py, pz = row["position_tcp_mm"]
                nx, ny, nz = row["normal_tcp"]
                print(
                    f"        p ({px:+7.2f},{py:+7.2f},{pz:+7.2f}) mm  "
                    f"n ({nx:+5.2f},{ny:+5.2f},{nz:+5.2f})  "
                    f"F {row['force_n']:7.3f} N  sep {row['separation_mm']:+6.3f} mm"
                )
            if jaw == "fixed":
                gap = min(xs) - 1e3 * face
                print(
                    f"        => fixed-jaw contact sits {gap:+.2f} mm from the VISUAL pad face; "
                    "that difference IS the gap the user sees."
                )


def report_first_contacts(onsets: dict, colliders: dict, spec) -> None:
    """Print the onset table: reported vs loaded vs disturbed, against the mesh."""
    print(
        f"\n  First contact, per jaw. {spec.name} is {onsets['object_width_mm']:.1f} mm "
        f"across, so the meshes put the pads on it at "
        f"{onsets['mesh_contact_angle_rad']:.3f} rad "
        f"= {onsets['mesh_contact_gap_mm']:.1f} mm of jaw gap."
    )
    for jaw, entry in onsets.get("jaws", {}).items():
        for kind in ("reported", "loaded"):
            hit = entry.get(kind)
            if hit is None:
                print(f"    {jaw:<6} {kind:<8} never")
                continue
            separation = hit["separation_mm"]
            print(
                f"    {jaw:<6} {kind:<8} {hit['state']} step {hit['step']:3d}  "
                f"grip {hit['gripper_rad']:.3f} rad  gap {hit['jaw_gap_mm']:6.2f} mm  "
                f"({hit['lead_over_mesh_mm']:+.2f} mm vs mesh)  "
                f"F {hit['force_n']:.4f} N  "
                f"sep [{separation[0]:+.3f}, {separation[1]:+.3f}] mm  "
                f"x {hit['x_mm']}  z {hit['z_mm']}"
            )
    moved = onsets.get("disturbed")
    print(
        "    object  disturbed never"
        if moved is None
        else (
            f"    object  disturbed {moved['state']} step {moved['step']:3d}  "
            f"grip {moved['gripper_rad']:.3f} rad  gap {moved['jaw_gap_mm']:6.2f} mm  "
            f"({moved['lead_over_mesh_mm']:+.2f} mm vs mesh)"
        )
    )
    print("\n  Collider parameters actually authored on the stage:")
    for path, values in colliders.items():
        print(f"    {path}: {values}")


def main() -> int:
    spec = OBJECTS[args_cli.object]
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device)
    )
    # Mirror task_scene.grasp_scene_cfg: the class body bakes cube_3cm, so the
    # requested object must be swapped in before the scene spawns.
    probe_cfg = ProbeSceneCfg(num_envs=1, env_spacing=2.0)
    # ``.replace`` rather than a bare spawn cfg: the class body's whole reason
    # for existing is the contact-report flag, and building a fresh spawner here
    # would drop it and leave the filtered pair unresolvable.
    probe_cfg.object.spawn = spec.make_spawn_cfg().replace(activate_contact_sensors=True)
    probe_cfg.object.init_state.pos = (*probe_cfg.object.init_state.pos[:2], spec.spawn_z)
    scene = InteractiveScene(probe_cfg)
    chain = "{ENV_REGEX_NS}/Robot/Geometry/" + "/".join(specs.LINK_CHAIN)
    reporting = ensure_contact_reporting(
        scene,
        {
            "fixed jaw": chain,
            "moving jaw": f"{chain}/{MOVING_JAW_BODY}",
            "object": "{ENV_REGEX_NS}/Object",
        },
    )
    sim.reset()
    runner = ProbeRunner(sim, scene, spec)
    for probe in runner.probes:
        print(f"  {probe.jaw}-jaw sensor bodies: {probe.body_names}")

    draw = draw_episode(args_cli.namespace, args_cli.attempt, spec)
    if args_cli.pose is not None:
        x, y, yaw = args_cli.pose
        draw = dataclasses.replace(draw, object_x=x, object_y=y, object_yaw=yaw)

    print(f"\n[{args_cli.label}] attempt {args_cli.attempt} of {args_cli.namespace!r}")
    result = runner.run(draw)
    result["probe_errors"] = runner.probe_errors
    result["contact_reporting"] = reporting
    report(result["samples"])
    result["colliders"] = collider_parameters(scene.stage)
    result["first_contacts"] = first_contacts(result["scan"], spec)
    report_first_contacts(result["first_contacts"], result["colliders"], spec)
    # Per-contact detail is kept only where it is evidence -- the onset rows,
    # which first_contacts() has already copied. A creeping CLOSE scans 200+
    # steps at up to --max-contacts each; carrying all of that would make the
    # json tens of megabytes of duplicated positions.
    for row in result["scan"]:
        for jaw in row["jaws"].values():
            jaw.pop("detail", None)
    print(f"\n  outcome: {result['outcome'].upper()}  success={result['success']}")

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = result.pop("frame")
    if frame is not None:
        from PIL import Image

        png = out_dir / f"{args_cli.label}_hold.png"
        Image.fromarray(frame).save(png)
        print(f"  wrote {png}")
    path = out_dir / f"{args_cli.label}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"  wrote {path}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    # The traceback goes to a *file* before the app is closed, and that is not
    # belt-and-braces: ``simulation_app.close()`` tears the process down hard
    # enough that an exception unwinding through this ``finally`` never gets to
    # print itself, and the run exits 0 with no output at all. Two boots of a
    # shared GPU were spent learning that.
    code = 1
    try:
        code = main()
    except BaseException:  # noqa: BLE001 - re-raised after it has been recorded
        import traceback

        out_dir = Path(args_cli.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args_cli.label}_error.txt"
        path.write_text(traceback.format_exc())
        traceback.print_exc()
        print(f"  FAILED: wrote {path}", flush=True)
        code = 2
    finally:
        simulation_app.close()
    sys.exit(code)
