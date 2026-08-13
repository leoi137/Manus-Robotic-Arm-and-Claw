"""Drive the scripted grasp expert in Isaac Sim: one grasp, watched and measured.

Three modes, all headless and all GPU-serialised (one Isaac process at a time):

.. code-block:: bash

    # one randomized attempt, with the state machine narrated
    ~/isaaclab-env/bin/python scripts/demo_expert.py --headless --attempt 0

    # the same, recording the wrist POV with a state overlay
    ~/isaaclab-env/bin/python scripts/demo_expert.py --headless --attempt 0 --video

    # Step 7's tuning grid: 5 placements x 4 object yaws, one boot
    ~/isaaclab-env/bin/python scripts/demo_expert.py --headless --tuning

Both artefacts are named after the object -- ``<out-dir>/<object>_<attempt>.mp4``
and ``<out-dir>/<object>_demo.json`` (``<object>_tuning.json`` for ``--tuning``)
-- so filming the whole catalogue into one ``--out-dir`` leaves one video and one
summary per object rather than one of each, overwritten six times. ``--label``
still overrides the video basename outright.

Rendering is off unless ``--video`` is passed: success is measured from the
object's height, not from pixels, and skipping the render is a ~4x throughput
win. The wrist camera is in the scene either way, it is simply never asked for
a frame (Isaac Lab sensors render lazily, on ``.data`` access or ``update()``).

Everything printed is measured, not asserted: per-state convergence, the droop
bias the expert had to hold to get there (commanded minus measured), the jaw
angle the object stops the fingers at, and the success predicate's own view.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

# Make the src-layout package importable without installing it. This must
# happen before the manus imports below, but the sim-side ones are deliberately
# deferred until after AppLauncher has started Isaac Sim: manus.task_scene and
# manus.robot pull in isaaclab extensions that only exist on a live app.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--attempt", type=int, default=0, help="draw index into the demo namespace")
parser.add_argument("--attempts", type=int, default=1, help="how many consecutive draws to run")
parser.add_argument(
    "--attempt-list",
    default="",
    help="comma-separated draw indices to run instead of --attempt/--attempts",
)
parser.add_argument(
    "--namespace",
    default="expert_demo",
    help="draw namespace; pass expert_gate to replay a gate attempt exactly",
)
parser.add_argument("--object", default="cube_3cm", help="catalogue key of the object to grasp")
parser.add_argument("--tuning", action="store_true", help="run the 5 placements x 4 yaws grid")
parser.add_argument("--video", action="store_true", help="record the wrist POV to an mp4")
parser.add_argument(
    "--scan",
    action="store_true",
    help=(
        "run the SCAN search phase first: sweep the workspace until the object "
        "is inside the wrist camera's frustum, then grasp as usual"
    ),
)
parser.add_argument(
    "--tp-video",
    action="store_true",
    help=(
        "also record a fixed third-person view of the whole scene to "
        "<label>_tp.mp4 (implies --video)"
    ),
)
parser.add_argument(
    "--pose",
    type=float,
    nargs=3,
    metavar=("X", "Y", "YAW"),
    help="force one placement (metres, metres, radians) instead of the draw's",
)
parser.add_argument(
    "--label",
    default=None,
    help=(
        "basename for the recorded video; the default is <object>_<attempt>, so "
        "filming a catalogue into one --out-dir cannot overwrite across objects"
    ),
)
parser.add_argument(
    "--trace",
    type=int,
    default=0,
    help="print commanded-minus-measured and the object pose every N control steps",
)
parser.add_argument(
    "--out-dir",
    default=str(REPO_ROOT / "runs" / "expert_demo"),
    help="where videos and the json summary are written",
)
parser.add_argument(
    "--close-target",
    type=float,
    default=None,
    help="override the object's close_target_rad for this run (tuning aid)",
)
parser.add_argument(
    "--pad-offset",
    type=float,
    default=None,
    help="override kinematics.TCP_TO_PAD_CENTRE for this run (tuning aid)",
)
parser.add_argument(
    "--jaw-clearance",
    type=float,
    default=None,
    help="override expert.JAW_CLEARANCE for this run (tuning aid)",
)
parser.add_argument(
    "--tip-clearance",
    type=float,
    default=None,
    help=(
        "override the fingertip-to-table gap at the grasp, metres -- the knob "
        "that moves a short object's grasp height (tuning aid; 0.003-0.007 is "
        "the puck's feasible band)"
    ),
)
parser.add_argument(
    "--converge-tol",
    type=float,
    default=None,
    help="override the arm convergence tolerance in radians (tuning aid)",
)
parser.add_argument(
    "--seat-gap",
    type=float,
    default=None,
    help=(
        "override the gap the SEAT state aims the static pad at, metres -- 0 is "
        "kiss contact, and passing the jaw clearance itself disables the seat's "
        "stroke entirely (tuning aid; only bites on a seat_close object)"
    ),
)
parser.add_argument(
    "--no-seat",
    action="store_true",
    help="run a seat_close object without its SEAT state, for an A/B against it",
)
parser.add_argument(
    "--close-ramp",
    type=int,
    default=None,
    help="override the steps the jaws take to reach the close target (tuning aid)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The scene always carries the wrist camera, and Isaac Lab's Camera refuses to
# initialise unless the app renders sensors -- so require it rather than making
# the caller remember --enable_cameras.
args_cli.enable_cameras = True
if args_cli.tp_video:
    args_cli.video = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from manus import expert as expert_mod
from manus import kinematics, specs
from manus.expert import (
    CLOSE,
    ExpertConfig,
    GraspSuccessMonitor,
    ScriptedGraspExpert,
    classify_outcome,
)
from manus.objects import OBJECTS
from manus.randomize import draw_episode, placement_region
from manus.task_scene import apply_randomization, grasp_scene_cfg

PHYSICS_DT = 1.0 / 120.0
"""Simulation step, seconds -- the pipeline's locked physics rate."""

DECIMATION = 4
"""Physics steps per control step, i.e. the expert runs at 30 Hz."""

SETTLE_STEPS = 30
"""Physics steps held at home after a reset, so the object comes to rest."""

MAX_CONTROL_STEPS = 1200
"""Hard ceiling per attempt; the FSM's own budgets stop well short of it."""

VIDEO_FPS = 30
"""Playback rate: one frame per control step, so video is real time."""

TP_CAM_EYE: tuple[float, float, float] = (0.82, 0.0, 0.42)
"""Where the optional third-person camera stands, metres (world frame).

Dead in front of the base and above it, looking back down the region's axis of
symmetry. The view is chosen for the *sweep*: SCAN is a 55-80 deg swing of
shoulder_pan, which is an azimuth change about the base, so a camera on that
axis renders the whole search as left-right motion across the frame. An oblique
view (tried first, at (0.62, -0.42, 0.38)) foreshortens exactly that motion and
makes a 100-step sweep look like the arm standing still.
"""

TP_CAM_TARGET: tuple[float, float, float] = (0.06, 0.0, 0.08)
"""What it looks at: just in front of the base, a little above the table -- the
arm and the whole annulus it sweeps, rather than one placement."""

TUNING_PLACEMENTS: tuple[tuple[float, float], ...] = (
    (0.165, 0.0),  # mid radius, straight ahead
    (0.115, 0.0),  # inner edge
    (0.216, 0.0),  # outer edge
    (0.200, 100.0),  # far corner, +azimuth
    (0.200, -100.0),  # far corner, -azimuth
)
"""(radius [m], azimuth [deg]) about the pan axis: the Step 7 tuning grid.

Top-down radii, frozen: this is the grid the expert was tuned on. A side-mode
object is swept over the corresponding points of its own annulus instead --
:func:`tuning_placements`."""

TUNING_YAWS_DEG: tuple[float, ...] = (0.0, 22.5, 45.0, 67.5)
"""Object yaws spanning one full 90 deg cube symmetry period."""


def tuning_placements(spec) -> tuple[tuple[float, float], ...]:
    """:data:`TUNING_PLACEMENTS`, mapped onto `spec`'s own placement region.

    The grid is five points -- mid radius, both radial edges, both far corners --
    and it means the same five things in either region. Top-down objects get the
    literal Step 7 radii back (the map is the identity on
    :data:`~manus.kinematics.GRASP_REGION`, which is the region they were
    written in); a side object gets the same *fractions* of its own annulus, so
    ``--tuning`` sweeps a reachable grid rather than five infeasible plans.
    """
    region = placement_region(spec)
    low, high = kinematics.GRASP_REGION.radius
    target_low, target_high = region.radius
    span = region.azimuth_max_deg / kinematics.GRASP_REGION.azimuth_max_deg
    return tuple(
        (
            target_low + (target_high - target_low) * (radius - low) / (high - low),
            azimuth * span,
        )
        for radius, azimuth in TUNING_PLACEMENTS
    )


def polar_to_xy(radius: float, azimuth_deg: float) -> tuple[float, float]:
    """World (x, y) of a point at `radius` and `azimuth_deg` about the pan axis."""
    azimuth = math.radians(azimuth_deg)
    return (
        kinematics.PAN_AXIS_XY[0] + radius * math.cos(azimuth),
        kinematics.PAN_AXIS_XY[1] + radius * math.sin(azimuth),
    )


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn a few lines of status text into the top-left of an RGB frame."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        position = (8, 8 + 14 * index)
        # Cheap outline: the wrist view is bright, plain white text vanishes.
        for offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text(
                (position[0] + offset[0], position[1] + offset[1]), line, fill=(0, 0, 0)
            )
        draw.text(position, line, fill=(255, 255, 255))
    return np.asarray(image)


class AttemptRunner:
    """Runs grasp attempts against one live scene, reusing it across draws."""

    def __init__(self, sim, scene, spec, config=None) -> None:
        self.sim = sim
        self.scene = scene
        self.spec = spec
        self.config = config or ExpertConfig()
        self.robot = scene["robot"]
        self.object = scene["object"]
        self.camera = scene["wrist_cam"]
        # Optional fixed third-person camera; only present when --tp-video put
        # one in the scene cfg, so every other run pays nothing for it.
        self.tp_camera = scene.sensors.get("tp_cam")
        if self.tp_camera is not None:
            self.tp_camera.set_world_poses_from_view(
                eyes=torch.tensor([TP_CAM_EYE], dtype=torch.float32),
                targets=torch.tensor([TP_CAM_TARGET], dtype=torch.float32),
            )
        self.dt = sim.get_physics_dt()
        self.device = self.robot.data.joint_pos.torch.device
        assert self.robot.joint_names == list(specs.JOINT_NAMES), (
            f"joint order mismatch: {self.robot.joint_names} != {list(specs.JOINT_NAMES)}"
        )
        self.home = torch.tensor(
            [[specs.HOME_POSE[name] for name in specs.JOINT_NAMES]],
            dtype=torch.float32,
            device=self.device,
        )

    # -- plumbing ---------------------------------------------------------------

    def measured(self) -> np.ndarray:
        """Six measured joint positions (radians), in ``specs.JOINT_NAMES`` order."""
        return self.robot.data.joint_pos.torch[0].detach().cpu().numpy().astype(float)

    def object_pos(self) -> np.ndarray:
        """Object body-origin (x, y, z) in the robot's own frame, metres.

        Environment origin subtracted, which is the frame the expert plans in
        and the frame :class:`~manus.expert.GraspSuccessMonitor` compares
        against its FK of the TCP.
        """
        world = self.object.data.root_link_pos_w.torch[0]
        return (world - self.scene.env_origins[0]).detach().cpu().numpy().astype(float)

    def object_z(self) -> float:
        """Object body-origin height above the ground plane, metres."""
        return float(self.object_pos()[2])

    def object_tilt_deg(self) -> float:
        """Angle between the object's own +z and world +z, degrees.

        The measurement the cylinder's failure is *about* -- a one-sided push
        tips it before it slides, and a cylinder tilted even 3 deg is wider than
        the closing gap and gets levered out -- and the puck's too, since
        climbing the moving finger shows up here first. Read off the body
        quaternion rather than inferred from the height.
        """
        w, x, y, z = (float(v) for v in self.object.data.root_link_quat_w.torch[0])
        return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))

    def advance(self, render: bool) -> None:
        """One physics step, refreshing only the buffers this script reads.

        Deliberately not ``scene.update()``: that ticks the wrist camera too,
        and with ``update_period=0`` the camera re-renders on every tick, which
        is the single biggest cost in a non-video run.
        """
        self.sim.step(render=render)
        self.robot.update(self.dt)
        self.object.update(self.dt)

    def reset_episode(self, draw) -> None:
        """Home the arm, stamp the draw onto the scene, and let it settle."""
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

    # -- one attempt -------------------------------------------------------------

    def run(
        self,
        draw,
        *,
        record: bool = False,
        verbose: bool = True,
        name: str = "attempt",
        video_name: str | None = None,
    ) -> dict:
        """Run one attempt end to end and return its outcome dictionary.

        `name` is the attempt's identity (the draw namespace and index, which is
        what the summary records); `video_name` is the basename the mp4 is
        written under, which defaults to `name` and is object-qualified by the
        caller so a catalogue filmed into one ``--out-dir`` cannot collide.
        """
        self.reset_episode(draw)
        measured = self.measured()
        expert = ScriptedGraspExpert(self.spec, config=self.config)
        plan = expert.reset(draw, q_current=measured)
        monitor = GraspSuccessMonitor(self.spec)
        frames: list[np.ndarray] = []
        tp_frames: list[np.ndarray] = []
        rest_z = self.object_z()

        if verbose:
            radius, azimuth = placement_region(self.spec).polar(draw.object_x, draw.object_y)
            print(
                f"  placement r={radius:.3f} m az={math.degrees(azimuth):+.1f} deg "
                f"yaw={math.degrees(draw.object_yaw):+.1f} deg  "
                f"friction={draw.object_static_friction:.2f}  rest_z={rest_z * 1e3:.1f} mm"
            )
            print(
                f"  plan: mode={plan.grasp_mode}  "
                f"{'roll' if plan.grasp_mode == 'side' else 'yaw'}"
                f"={math.degrees(plan.grasp_yaw):+.1f} deg  "
                f"lift_rise={plan.lift_rise * 1e3:.0f} mm  close_target={plan.close_target:.3f} rad"
                + ("" if plan.ok else f"  INFEASIBLE: {plan.reason}")
            )

        seen = expert.state
        # Per-state object motion, which is what a seating failure is argued
        # from: how far the object moved *while a state was running*, measured
        # from where it stood when that state began.
        motion: dict[str, dict[str, float | list[float]]] = {}
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
            for substep in range(DECIMATION):
                self.advance(render=record and substep == DECIMATION - 1)
            measured = self.measured()
            position = self.object_pos()
            monitor.update(position, measured)

            record = motion.setdefault(
                expert.state,
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
                float(record["max_dxy_mm"]), 1e3 * float(math.hypot(delta[0], delta[1]))
            )
            record["max_rise_mm"] = max(float(record["max_rise_mm"]), 1e3 * float(delta[2]))
            record["max_tilt_deg"] = max(
                float(record["max_tilt_deg"]),
                abs(self.object_tilt_deg() - float(record["start_tilt_deg"])),
            )
            record["end"] = [float(v) for v in position]

            if args_cli.trace and expert.total_steps % args_cli.trace == 0:
                error = [targets[name] - value for name, value in zip(specs.JOINT_NAMES, measured)]
                position = self.object.data.root_link_pos_w.torch[0].cpu().numpy()
                print(
                    f"      t{expert.total_steps:4d} {expert.state:<8} "
                    f"cmd-meas mrad " + " ".join(f"{value * 1e3:+6.0f}" for value in error)
                    + f"  obj ({position[0]:+.3f},{position[1]:+.3f},{position[2]:.3f})"
                )

            if record:
                # Sensors refresh on update(), not on step(); without this the
                # camera would hand back the same frame all episode.
                self.camera.update(self.dt * DECIMATION)
                if self.tp_camera is not None:
                    self.tp_camera.update(self.dt * DECIMATION)
                    tp_frames.append(
                        annotate(
                            self.tp_camera.data.output["rgb"].torch[0, ..., :3]
                            .to(torch.uint8)
                            .cpu()
                            .numpy(),
                            [
                                f"{expert.state}  step {expert.state_step}",
                                f"object z {self.object_z() * 1e3:6.1f} mm",
                            ],
                        )
                    )
                frames.append(
                    annotate(
                        self.camera.data.output["rgb"].torch[0, ..., :3]
                        .to(torch.uint8)
                        .cpu()
                        .numpy(),
                        [
                            f"{expert.state}  step {expert.state_step}",
                            f"object z {self.object_z() * 1e3:6.1f} mm "
                            f"(bar {monitor.threshold_z * 1e3:.0f})",
                            f"gripper  {measured[-1]:.3f} rad",
                            f"hold {monitor.streak:2d}/{monitor.sustain}"
                            + ("  SUCCESS" if monitor.success else ""),
                        ],
                    )
                )
            if verbose and expert.state != seen:
                report = expert.reports[-1]
                print(
                    f"    {report.state:<8} {report.exit:<9} {report.steps:3d} steps  "
                    f"|q-target| {report.joint_error * 1e3:6.2f} mrad  "
                    + (
                        f"tcp {report.tcp_error * 1e3:5.2f} mm  "
                        if report.tcp_error is not None
                        else " " * 13
                    )
                    + f"droop {np.abs(report.bias).max() * 1e3:5.1f} mrad  "
                    f"grip {report.gripper:.3f}"
                )
                seen = expert.state

        outcome = classify_outcome(expert, monitor)
        result = {
            "attempt": None,
            "draw": draw.to_dict(),
            "outcome": outcome,
            "success": monitor.success,
            "rest_z": rest_z,
            "monitor": monitor.to_dict(),
            "telemetry": expert.telemetry(),
            "object_motion": motion,
        }
        if verbose:
            print("    object motion, per state (from where it stood when the state began):")
            for state, record in motion.items():
                print(
                    f"      {state:<9} dxy {float(record['max_dxy_mm']):6.2f} mm  "
                    f"rise {float(record['max_rise_mm']):7.2f} mm  "
                    f"tilt {float(record['max_tilt_deg']):6.2f} deg"
                )
            seat_exit = expert.telemetry().get("seat_exit")
            if seat_exit is not None:
                excess = expert.telemetry()["seat_excess"] or 0.0
                print(f"    seat: exit {seat_exit}, tracking excess {excess * 1e3:.2f} mrad")
            print(
                f"  -> {outcome.upper()}  peak z {monitor.peak_z * 1e3:.1f} mm "
                f"(bar {monitor.threshold_z * 1e3:.1f})  held {monitor.best_streak}/"
                f"{monitor.sustain}  steps {expert.total_steps}"
                + (f"  timeouts {expert.timeouts}" if expert.timeouts else "")
            )
        if record and frames:
            result["video"] = self.write_video(frames, video_name or name)
        if record and tp_frames:
            result["tp_video"] = self.write_video(tp_frames, video_name or name, suffix="_tp")
        return result

    def write_video(self, frames: list[np.ndarray], name: str, suffix: str = "") -> str:
        """Write the captured wrist frames to an mp4 and return its path.

        ``--label`` still wins outright, which is what pins the one-file-per-
        object preview names; `name` is the object-qualified default.
        """
        import imageio.v3 as iio

        out_dir = Path(args_cli.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # The suffix rides *outside* the --label override on purpose: without
        # it the third-person take would overwrite the wrist take whenever a
        # label is given, which is exactly when both are wanted.
        path = out_dir / f"{args_cli.label or name}{suffix}.mp4"
        iio.imwrite(path, np.stack(frames), fps=VIDEO_FPS)
        print(f"  video: {len(frames)} frames -> {path}")
        return str(path)


def droop_table(results: list[dict]) -> None:
    """Print the per-state convergence and droop summary over some attempts."""
    per_state: dict[str, list[dict]] = {}
    for result in results:
        for report in result["telemetry"]["states"]:
            per_state.setdefault(report["state"], []).append(report)
    print("\n  state     n   steps(med)  |q-target| mrad   tcp mm   droop mrad   exits")
    for state, reports in per_state.items():
        steps = np.median([report["steps"] for report in reports])
        joint = np.array([report["joint_error"] for report in reports]) * 1e3
        tcp = np.array(
            [report["tcp_error"] for report in reports if report["tcp_error"] is not None]
        )
        droop = np.array([max(np.abs(report["bias"])) for report in reports]) * 1e3
        exits: dict[str, int] = {}
        for report in reports:
            exits[report["exit"]] = exits.get(report["exit"], 0) + 1
        print(
            f"  {state:<8} {len(reports):3d}  {steps:9.0f}  "
            f"{joint.mean():7.2f} (max {joint.max():6.2f})  "
            f"{(tcp.mean() * 1e3 if tcp.size else 0.0):6.2f}  "
            f"{droop.mean():7.1f} (max {droop.max():6.1f})  {exits}"
        )


def main() -> int:
    """Run the requested attempts. Returns the process exit code."""
    spec = OBJECTS[args_cli.object]
    if args_cli.close_target is not None:
        spec = dataclasses.replace(spec, close_target_rad=args_cli.close_target)
        print(f"OVERRIDE close_target_rad = {spec.close_target_rad}")
    if args_cli.pad_offset is not None:
        # The expert reads the constant through the module, so rebinding it here
        # re-aims every plan made afterwards. Tuning aid only -- the committed
        # value lives in kinematics.py.
        kinematics.TCP_TO_PAD_CENTRE = args_cli.pad_offset
        expert_mod.TCP_TO_PAD_CENTRE = args_cli.pad_offset
        print(f"OVERRIDE TCP_TO_PAD_CENTRE = {args_cli.pad_offset}")
    if args_cli.jaw_clearance is not None:
        expert_mod.JAW_CLEARANCE = args_cli.jaw_clearance
        print(f"OVERRIDE JAW_CLEARANCE = {args_cli.jaw_clearance}")
    if args_cli.tip_clearance is not None:
        # Per spec rather than per module: the grasp height is an object
        # property, so the override rides with the object the whole way through
        # the plan (and into the spec any sweep records).
        spec = dataclasses.replace(spec, tip_clearance_m=args_cli.tip_clearance)
        print(
            f"OVERRIDE tip_clearance_m = {spec.tip_clearance_m} "
            f"-> grasp_height {expert_mod.grasp_height(spec) * 1e3:.2f} mm"
        )
    config = ExpertConfig()
    if args_cli.converge_tol is not None:
        config = dataclasses.replace(config, converge_tol=args_cli.converge_tol)
        print(f"OVERRIDE converge_tol = {config.converge_tol}")
    if args_cli.close_ramp is not None:
        config = dataclasses.replace(config, close_ramp=args_cli.close_ramp)
        print(f"OVERRIDE close_ramp = {config.close_ramp}")
    if args_cli.scan:
        config = dataclasses.replace(config, scan_phase=True)
        print(
            f"SCAN phase on: sweep at {config.scan_rate:.3f} rad/step, "
            f"entry ramp {config.scan_entry_ramp}, view margin {config.scan_view_margin}"
        )
    if args_cli.no_seat:
        # Per spec, like every other object property: the FSM reads seat_close
        # off the spec, so clearing it is exactly "the run this object had
        # before the seat existed" and nothing else changes.
        spec = dataclasses.replace(spec, seat_close=False)
        print("OVERRIDE seat_close = False (no SEAT state)")
    if args_cli.seat_gap is not None:
        config = dataclasses.replace(config, seat_gap=args_cli.seat_gap)
        print(
            f"OVERRIDE seat_gap = {config.seat_gap} -> stroke "
            f"{expert_mod.seat_stroke(config) * 1e3:.2f} mm over "
            f"{expert_mod.seat_ramp_steps(config)} steps"
        )

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device)
    )
    scene_cfg = grasp_scene_cfg(args_cli.object, num_envs=1, env_spacing=2.0)
    if args_cli.tp_video:
        # Attached to the *instance*, not the class: a third-person view is a
        # filming aid, and every dataset run must keep paying for exactly one
        # camera. InteractiveScene builds its entities from the cfg's __dict__,
        # so an instance attribute is all it takes.
        from isaaclab.sensors import CameraCfg

        scene_cfg.tp_cam = CameraCfg(
            prim_path="/World/tp_cam",
            update_period=0.0,
            width=640,
            height=480,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                horizontal_aperture=specs.WRIST_CAM_APERTURE,
                clipping_range=(0.05, 50.0),
            ),
        )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    runner = AttemptRunner(sim, scene, spec, config)

    # (attempt identity, video basename, draw). The video basename carries the
    # object so that filming several objects into one --out-dir writes several
    # files rather than overwriting one; the attempt identity stays the draw's,
    # because that is what the summary has to record to be replayable.
    jobs: list[tuple[str, str, object]] = []
    if args_cli.tuning:
        for radius, azimuth in tuning_placements(spec):
            x, y = polar_to_xy(radius, azimuth)
            for yaw_deg in TUNING_YAWS_DEG:
                # One draw per grid slot, so lighting/friction/colour vary the
                # way they will in the gate while the placement stays fixed.
                # The placement is overwritten below, so the draw's own region
                # does not matter -- only its lighting and friction are used.
                base = draw_episode("expert_tuning", len(jobs), spec)
                slot = f"r{radius:.3f}_az{azimuth:+.0f}_yaw{yaw_deg:+.1f}"
                jobs.append((
                    slot,
                    f"{args_cli.object}_{slot}",
                    dataclasses.replace(
                        base, object_x=x, object_y=y, object_yaw=math.radians(yaw_deg)
                    ),
                ))
    else:
        indices = (
            [int(part) for part in args_cli.attempt_list.split(",") if part.strip()]
            if args_cli.attempt_list
            else [args_cli.attempt + index for index in range(args_cli.attempts)]
        )
        for attempt in indices:
            draw = draw_episode(args_cli.namespace, attempt, spec)
            if args_cli.pose is not None:
                x, y, yaw = args_cli.pose
                draw = dataclasses.replace(draw, object_x=x, object_y=y, object_yaw=yaw)
            jobs.append((
                f"{args_cli.namespace}_{attempt:04d}",
                f"{args_cli.object}_{attempt:04d}",
                draw,
            ))

    results = []
    for name, video_name, draw in jobs:
        print(f"\n[{name}]")
        result = runner.run(
            draw, record=args_cli.video, verbose=True, name=name, video_name=video_name
        )
        result["attempt"] = name
        results.append(result)

    successes = sum(1 for result in results if result["success"])
    droop_table(results)
    modes: dict[str, int] = {}
    for result in results:
        modes[result["outcome"]] = modes.get(result["outcome"], 0) + 1
    close_grips = [
        report["gripper"]
        for result in results
        for report in result["telemetry"]["states"]
        if report["state"] == CLOSE
    ]
    if close_grips:
        print(
            f"\n  jaw angle at CLOSE exit: mean {np.mean(close_grips):.3f} rad, "
            f"range [{min(close_grips):.3f}, {max(close_grips):.3f}]"
        )
    print(f"\nRESULT: {successes}/{len(results)} succeeded   modes={modes}")

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Object-qualified for the same reason the videos are: a catalogue sweep
    # writes one summary per object instead of clobbering a single demo.json.
    summary = out_dir / (
        f"{args_cli.object}_tuning.json" if args_cli.tuning else f"{args_cli.object}_demo.json"
    )
    summary.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {summary}")
    return 0 if successes == len(results) else 1


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
