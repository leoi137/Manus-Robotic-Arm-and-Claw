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
from manus.objects import OBJECTS
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

    def read(self, tcp_pos: torch.Tensor, tcp_quat: torch.Tensor) -> list[Contact]:
        """Every contact this jaw currently has with the filtered object.

        Args:
            tcp_pos: TCP world position, shape (1, 3).
            tcp_quat: TCP world orientation (x, y, z, w), shape (1, 4).
        """
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
        self.probes = [
            ContactProbe(scene["fixed_jaw_contact"], "fixed", FIXED_JAW_BODY, self.dt),
            ContactProbe(scene["moving_jaw_contact"], "moving", MOVING_JAW_BODY, self.dt),
        ]

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
        seen = expert.state
        hold_seen = 0
        frame = None
        for _ in range(MAX_CONTROL_STEPS):
            if expert.done:
                break
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
            monitor.update(self.object_pos(), measured)

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
        return {
            "label": args_cli.label,
            "outcome": outcome,
            "success": bool(monitor.success),
            "draw": draw.to_dict(),
            "samples": samples,
            "frame": frame,
        }


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


def main() -> int:
    spec = OBJECTS[args_cli.object]
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device)
    )
    # Mirror task_scene.grasp_scene_cfg: the class body bakes cube_3cm, so the
    # requested object must be swapped in before the scene spawns.
    probe_cfg = ProbeSceneCfg(num_envs=1, env_spacing=2.0)
    probe_cfg.object.spawn = spec.make_spawn_cfg()
    probe_cfg.object.init_state.pos = (*probe_cfg.object.init_state.pos[:2], spec.spawn_z)
    scene = InteractiveScene(probe_cfg)
    sim.reset()
    runner = ProbeRunner(sim, scene, spec)
    print(f"  fixed-jaw sensor bodies : {runner.probes[0].body_names}")
    print(f"  moving-jaw sensor bodies: {runner.probes[1].body_names}")

    draw = draw_episode(args_cli.namespace, args_cli.attempt)
    if args_cli.pose is not None:
        x, y, yaw = args_cli.pose
        draw = dataclasses.replace(draw, object_x=x, object_y=y, object_yaw=yaw)

    print(f"\n[{args_cli.label}] attempt {args_cli.attempt} of {args_cli.namespace!r}")
    result = runner.run(draw)
    report(result["samples"])
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
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
