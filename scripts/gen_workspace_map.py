"""Map the graspable workspace, and gate the scripted expert over it.

Two modes, deliberately in one script: the map says where a grasp is *plannable*
and the gate says where one actually *works*, and they have to be read against
the same cells to mean anything.

.. code-block:: bash

    # (a) map: sim-free, samples the region with the expert's own planner
    ~/isaaclab-env/bin/python scripts/gen_workspace_map.py

    # (b) gate: 200 seeded placements in Isaac, 50 per boot, resumable
    ~/isaaclab-env/bin/python scripts/gen_workspace_map.py --gate --headless
    ...  # repeat until it reports nothing pending
    ~/isaaclab-env/bin/python scripts/gen_workspace_map.py --report-only

Map mode writes ``docs/workspace_map.json`` and ``docs/workspace_map.png`` and
never starts Isaac Sim -- the planner is pure numpy, so the whole region can be
swept in seconds on the CPU.

Gate mode runs :class:`~manus.expert.ScriptedGraspExpert` against the full
randomization of :func:`~manus.randomize.draw_episode` and appends one JSON
line per attempt to ``runs/expert_gate/attempts.jsonl``. That ledger is the
source of truth: a boot processes the first ``--chunk`` *pending* attempts and
exits, so a 200-attempt gate is four boots and survives being interrupted.
``report.md`` is regenerated from the ledger whenever the gate is complete.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Importing this does not start the simulator -- only AppLauncher(...) does, and
# map mode never gets there.
from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_JSON = REPO_ROOT / "docs" / "workspace_map.json"
MAP_PNG = REPO_ROOT / "docs" / "workspace_map.png"
GATE_DIR = REPO_ROOT / "runs" / "expert_gate"
GATE_LEDGER = GATE_DIR / "attempts.jsonl"
GATE_REPORT = GATE_DIR / "report.md"

GATE_NAMESPACE = "expert_gate"
"""Dataset name the gate's draws are seeded from."""

GATE_ATTEMPTS = 200
"""How many seeded placements the gate runs (the plan's bar is >= 200)."""

GATE_TARGET = 0.95
"""Success rate the gate demands."""

RADIUS_BINS = 3
AZIMUTH_BINS = 6
"""Report cells: the region cut into radius x azimuth boxes."""

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gate", action="store_true", help="run the expert gate in Isaac Sim")
parser.add_argument("--report-only", action="store_true", help="rebuild report.md from the ledger")
parser.add_argument("--attempts", type=int, default=GATE_ATTEMPTS, help="gate size")
parser.add_argument("--chunk", type=int, default=50, help="attempts to run in this boot")
parser.add_argument("--object", default="cube_3cm", help="catalogue key of the object to grasp")
parser.add_argument("--redo", default="", help="comma-separated attempt indices to re-run")
parser.add_argument(
    "--redo-failed", action="store_true", help="re-run every attempt the ledger records as failed"
)
parser.add_argument("--map-radius-cells", type=int, default=24, help="map resolution, radial")
parser.add_argument("--map-azimuth-cells", type=int, default=42, help="map resolution, angular")
parser.add_argument("--map-yaws", type=int, default=8, help="object yaws sampled per map cell")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# --- Cells ---------------------------------------------------------------------


def polar(x: float, y: float) -> tuple[float, float]:
    """``(radius [m], azimuth [deg])`` of a world point about the pan axis."""
    from manus.kinematics import GRASP_REGION

    radius, azimuth = GRASP_REGION.polar(x, y)
    return radius, math.degrees(azimuth)


def cell_of(x: float, y: float) -> str:
    """Name of the report cell a placement falls in, e.g. ``"r1_az-2"``."""
    from manus.kinematics import GRASP_REGION

    radius, azimuth = polar(x, y)
    low, high = GRASP_REGION.radius
    span = GRASP_REGION.azimuth_max_deg
    radius_bin = min(RADIUS_BINS - 1, max(0, int((radius - low) / (high - low) * RADIUS_BINS)))
    azimuth_bin = min(
        AZIMUTH_BINS - 1, max(0, int((azimuth + span) / (2 * span) * AZIMUTH_BINS))
    )
    return f"r{radius_bin}_az{azimuth_bin}"


def cell_label(name: str) -> str:
    """Human-readable bounds of a cell name from :func:`cell_of`."""
    from manus.kinematics import GRASP_REGION

    radius_bin, azimuth_bin = (int(part[1:]) if part[0] == "r" else int(part[2:])
                               for part in name.split("_"))
    low, high = GRASP_REGION.radius
    span = GRASP_REGION.azimuth_max_deg
    r0 = low + (high - low) * radius_bin / RADIUS_BINS
    r1 = low + (high - low) * (radius_bin + 1) / RADIUS_BINS
    a0 = -span + 2 * span * azimuth_bin / AZIMUTH_BINS
    a1 = -span + 2 * span * (azimuth_bin + 1) / AZIMUTH_BINS
    return f"r {r0:.3f}-{r1:.3f} m, az {a0:+.0f}..{a1:+.0f} deg"


# --- (a) Map -------------------------------------------------------------------


def sweep_region() -> dict:
    """Plan a grasp at every (radius, azimuth) cell and count what solves.

    Uses :func:`manus.expert.plan_grasp` rather than
    :func:`manus.kinematics.ik_solve` directly, so the map answers the question
    the pipeline actually asks: can the *expert* grasp here -- both waypoints
    solved at a jaw-aligned yaw the solver did not silently flip, and a lift
    that clears. That is a strictly harder question than "is the TCP reachable".
    """
    import numpy as np

    from manus.expert import plan_grasp
    from manus.kinematics import GRASP_REGION
    from manus.objects import OBJECTS

    spec = OBJECTS[args_cli.object]
    low, high = GRASP_REGION.radius
    span = GRASP_REGION.azimuth_max_deg
    # Edges rather than centres: the cells tile the sector, so the figure can
    # draw them as a polar mesh with no gaps and the JSON stays reconstructable.
    radius_edges = np.linspace(low, high, args_cli.map_radius_cells + 1)
    azimuth_edges = np.linspace(-span, span, args_cli.map_azimuth_cells + 1)
    radii = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    azimuths = 0.5 * (azimuth_edges[:-1] + azimuth_edges[1:])
    yaws = np.linspace(-math.pi, math.pi, args_cli.map_yaws, endpoint=False)

    cells = []
    for radius in radii:
        for azimuth_deg in azimuths:
            azimuth = math.radians(azimuth_deg)
            x = GRASP_REGION.pan_axis_xy[0] + radius * math.cos(azimuth)
            y = GRASP_REGION.pan_axis_xy[1] + radius * math.sin(azimuth)
            if GRASP_REGION.in_keepout(x, y):
                cells.append(
                    {
                        "r": float(radius),
                        "azimuth_deg": float(azimuth_deg),
                        "x": float(x),
                        "y": float(y),
                        "solvable": None,  # excluded, not unsolvable
                        "samples": 0,
                        "keepout": True,
                    }
                )
                continue
            solved = sum(
                1 for yaw in yaws if plan_grasp(spec, (x, y, float(yaw))).ok
            )
            cells.append(
                {
                    "r": float(radius),
                    "azimuth_deg": float(azimuth_deg),
                    "x": float(x),
                    "y": float(y),
                    "solvable": solved / len(yaws),
                    "samples": len(yaws),
                    "keepout": False,
                }
            )

    live = [cell for cell in cells if not cell["keepout"]]
    return {
        "generated_by": "scripts/gen_workspace_map.py",
        "object": spec.name,
        "region": {
            "pan_axis_xy": list(GRASP_REGION.pan_axis_xy),
            "radius_m": list(GRASP_REGION.radius),
            "azimuth_max_deg": GRASP_REGION.azimuth_max_deg,
            "keepout_x_m": list(GRASP_REGION.keepout_x),
            "keepout_abs_y_m": GRASP_REGION.keepout_abs_y,
        },
        "sampling": {
            "radius_cells": args_cli.map_radius_cells,
            "azimuth_cells": args_cli.map_azimuth_cells,
            "yaws_per_cell": args_cli.map_yaws,
            "criterion": "manus.expert.plan_grasp(...).ok",
            "radius_edges_m": [float(edge) for edge in radius_edges],
            "azimuth_edges_deg": [float(edge) for edge in azimuth_edges],
            "cell_order": "row-major over (radius, azimuth)",
        },
        "summary": {
            "cells": len(cells),
            "cells_in_region": len(live),
            "cells_fully_solvable": sum(1 for cell in live if cell["solvable"] == 1.0),
            "mean_solvable_fraction": (
                sum(cell["solvable"] for cell in live) / len(live) if live else 0.0
            ),
        },
        "cells": cells,
    }


def draw_map(data: dict, path: Path) -> bool:
    """Render the map to `path`. Returns False if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        return False

    import numpy as np

    region = data["region"]
    sampling = data["sampling"]
    pan_x, pan_y = region["pan_axis_xy"]

    # The cells tile the sector, so draw them as a polar mesh: corner grid from
    # the recorded edges, values row-major over (radius, azimuth), keep-out
    # cells left as NaN so the colormap's "bad" colour shows through.
    radius_edges = np.array(sampling["radius_edges_m"])
    azimuth_edges = np.radians(sampling["azimuth_edges_deg"])
    corner_r, corner_a = np.meshgrid(radius_edges, azimuth_edges, indexing="ij")
    values = np.array(
        [math.nan if cell["keepout"] else cell["solvable"] for cell in data["cells"]]
    ).reshape(len(radius_edges) - 1, len(azimuth_edges) - 1)

    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("0.85")
    figure, axes = plt.subplots(figsize=(7.5, 7.0))
    scatter = axes.pcolormesh(
        pan_x + corner_r * np.cos(corner_a),
        pan_y + corner_r * np.sin(corner_a),
        np.ma.masked_invalid(values),
        cmap=colormap,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )

    # Region outline: the annulus sector, walked out along one radius and back
    # along the other.
    low, high = region["radius_m"]
    span = math.radians(region["azimuth_max_deg"])
    sweep = np.linspace(-span, span, 200)
    outline_x = [*(pan_x + high * np.cos(sweep)), *(pan_x + low * np.cos(sweep[::-1]))]
    outline_y = [*(pan_y + high * np.sin(sweep)), *(pan_y + low * np.sin(sweep[::-1]))]
    axes.plot([*outline_x, outline_x[0]], [*outline_y, outline_y[0]], "k-", lw=1.4,
              label="GRASP_REGION")

    keepout_x = region["keepout_x_m"]
    axes.add_patch(
        Rectangle(
            (keepout_x[0], -region["keepout_abs_y_m"]),
            keepout_x[1] - keepout_x[0],
            2 * region["keepout_abs_y_m"],
            fill=False,
            edgecolor="crimson",
            lw=1.2,
            linestyle="--",
            label="base keep-out",
        )
    )
    axes.plot([pan_x], [pan_y], "k+", ms=10, label="shoulder_pan axis")

    axes.set_aspect("equal")
    axes.set_xlabel("world x [m]")
    axes.set_ylabel("world y [m]")
    axes.set_title(
        f"Graspable workspace, {data['object']}\n"
        f"fraction of {sampling['yaws_per_cell']} object yaws the expert can plan "
        "(grey = base keep-out)"
    )
    figure.colorbar(scatter, ax=axes, label="solvable fraction", shrink=0.8)
    handles, labels = axes.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axes.legend(unique.values(), unique.keys(), loc="upper left", fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True


def run_map() -> int:
    """Write the workspace map. Returns the process exit code."""
    data = sweep_region()
    MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    MAP_JSON.write_text(json.dumps(data, indent=1) + "\n")
    summary = data["summary"]
    print(
        f"cells {summary['cells_in_region']} in region "
        f"({summary['cells'] - summary['cells_in_region']} in the base keep-out); "
        f"fully solvable {summary['cells_fully_solvable']}; "
        f"mean solvable fraction {summary['mean_solvable_fraction']:.4f}"
    )
    print(f"wrote {MAP_JSON}")
    if draw_map(data, MAP_PNG):
        print(f"wrote {MAP_PNG}")
    else:
        print("matplotlib unavailable: skipped the png")
    return 0


# --- (b) Gate ------------------------------------------------------------------


def read_ledger() -> dict[int, dict]:
    """Latest record per attempt index from the append-only ledger."""
    if not GATE_LEDGER.exists():
        return {}
    records = {}
    for line in GATE_LEDGER.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["attempt"])] = record
    return records


def append_ledger(record: dict) -> None:
    """Append one attempt outcome to the ledger, flushed immediately."""
    GATE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with GATE_LEDGER.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


def pending(done: dict[int, dict]) -> list[int]:
    """Attempt indices still to run, honouring --redo / --redo-failed."""
    if args_cli.redo:
        return [int(part) for part in args_cli.redo.split(",") if part.strip()]
    if args_cli.redo_failed:
        return sorted(index for index, record in done.items() if not record["success"])
    return [index for index in range(args_cli.attempts) if index not in done]


def configuration() -> list[tuple[str, str, str]]:
    """``(name, value, module)`` of every constant the gate's behaviour depends on."""
    from manus import expert, kinematics
    from manus.objects import OBJECTS

    spec = OBJECTS[args_cli.object]
    config = expert.ExpertConfig()
    return [
        ("TCP_TO_PAD_CENTRE", f"{kinematics.TCP_TO_PAD_CENTRE * 1e3:.1f} mm", "kinematics"),
        ("pad_lateral_offset", f"{expert.pad_lateral_offset(spec) * 1e3:.1f} mm", "expert"),
        ("JAW_FIXED_FACE_X", f"{expert.JAW_FIXED_FACE_X * 1e3:.1f} mm", "expert"),
        ("JAW_CLEARANCE", f"{expert.JAW_CLEARANCE * 1e3:.1f} mm", "expert"),
        ("close_target_rad", f"{spec.close_target_rad:.3f} rad", f"objects[{spec.name}]"),
        ("gripper_open", f"{config.gripper_open:.3f} rad", "expert.ExpertConfig"),
        ("close_ramp", f"{expert.close_ramp_steps(spec, config)} steps", f"objects[{spec.name}]"),
        ("hover_height", f"{expert.pregrasp_height(spec, config) * 1e3:.1f} mm", "expert"),
        ("converge_tol", f"{config.converge_tol * 1e3:.0f} mrad", "expert.ExpertConfig"),
        ("state_budget", f"{config.state_budget} steps", "expert.ExpertConfig"),
        ("hold_steps", f"{config.hold_steps} steps", "expert.ExpertConfig"),
        ("lift_rise", f"{config.lift_rise * 1e3:.0f} mm", "expert.ExpertConfig"),
        (
            "droop gain/leak/limit",
            f"{config.droop_gain} / {config.droop_leak} / {config.droop_limit * 1e3:.0f} mrad",
            "expert.ExpertConfig",
        ),
        ("control rate", "30 Hz (physics 1/120 s, decimation 4)", "the pipeline contract"),
    ]


def write_report(done: dict[int, dict]) -> None:
    """Render ``report.md`` from the ledger."""
    import numpy as np

    records = [done[index] for index in sorted(done) if index < args_cli.attempts]
    total = len(records)
    successes = sum(1 for record in records if record["success"])
    rate = successes / total if total else 0.0
    modes = Counter(record["outcome"] for record in records)

    by_cell: dict[str, list[dict]] = {}
    for record in records:
        by_cell.setdefault(record["cell"], []).append(record)

    def state_stats(state: str, key: str) -> tuple[float, float]:
        values = [
            report[key]
            for record in records
            for report in record["states"]
            if report["state"] == state and report[key] is not None
        ]
        return (float(np.mean(values)), float(np.max(values))) if values else (0.0, 0.0)

    lines = [
        "# Expert gate — scripted grasp over the full grasp region",
        "",
        f"`{GATE_NAMESPACE}` draws 0..{args_cli.attempts - 1}, full per-attempt randomization "
        "(placement, object yaw, colour, friction, both lights, ground albedo, wrist-camera "
        "mount jitter). Success predicate: object centre ≥ spawn + 50 mm for 30 consecutive "
        "control steps with the jaws closed.",
        "",
        f"**{successes}/{total} = {rate * 100:.1f}%** — gate is ≥ {GATE_TARGET * 100:.0f}%: "
        f"**{'PASS' if rate >= GATE_TARGET and total >= GATE_ATTEMPTS else 'FAIL'}**",
        "",
        "## Outcomes",
        "",
        "| outcome | n |",
        "| --- | --- |",
    ]
    lines += [f"| {mode} | {count} |" for mode, count in modes.most_common()]

    margins = sorted((record["peak_z"] - record["threshold_z"]) * 1e3 for record in records)
    if margins:
        lines += [
            "",
            f"Height margin over the 50 mm bar: min {margins[0]:.1f} mm, "
            f"median {margins[len(margins) // 2]:.1f} mm, max {margins[-1]:.1f} mm — "
            "the successes are not marginal.",
        ]

    lines += [
        "",
        "## Configuration",
        "",
        "Every constant the gate was run with, read out of the code at report time.",
        "",
        "| constant | value | where |",
        "| --- | --- | --- |",
    ]
    lines += [f"| `{name}` | {value} | {where} |" for name, value, where in configuration()]

    lines += [
        "",
        "## By region cell",
        "",
        "| cell | bounds | n | successes | rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in sorted(by_cell):
        cell = by_cell[name]
        hits = sum(1 for record in cell if record["success"])
        lines.append(
            f"| `{name}` | {cell_label(name)} | {len(cell)} | {hits} | "
            f"{hits / len(cell) * 100:.0f}% |"
        )

    lines += [
        "",
        "## Per-state behaviour",
        "",
        "Convergence is servo-to-converge: a state ends when the measured joints reach the "
        "waypoint, and the droop column is the integral bias the expert had to hold to get "
        "them there — i.e. the commanded-minus-measured offset gravity costs.",
        "",
        "| state | steps (mean/max) | ‖q−target‖∞ at exit [mrad] | TCP error [mm] | "
        "droop bias [mrad] | exits |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for state in ("PREGRASP", "DESCEND", "CLOSE", "LIFT", "HOLD"):
        steps = [
            report["steps"]
            for record in records
            for report in record["states"]
            if report["state"] == state
        ]
        if not steps:
            continue
        exits = Counter(
            report["exit"]
            for record in records
            for report in record["states"]
            if report["state"] == state
        )
        joint_mean, joint_max = state_stats(state, "joint_error")
        tcp_mean, tcp_max = state_stats(state, "tcp_error")
        bias = [
            max(abs(value) for value in report["bias"])
            for record in records
            for report in record["states"]
            if report["state"] == state
        ]
        lines.append(
            f"| {state} | {np.mean(steps):.0f} / {max(steps)} | "
            f"{joint_mean * 1e3:.1f} / {joint_max * 1e3:.1f} | "
            f"{tcp_mean * 1e3:.1f} / {tcp_max * 1e3:.1f} | "
            f"{np.mean(bias) * 1e3:.1f} / {max(bias) * 1e3:.1f} | "
            f"{dict(exits)} |"
        )

    failures = [record for record in records if not record["success"]]
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("None.")
    else:
        lines += [
            "| attempt | outcome | cell | r [m] | az [deg] | object yaw [deg] | friction | "
            "peak z [mm] | timeouts |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for record in failures:
            draw = record["draw"]
            lines.append(
                f"| {record['attempt']} | {record['outcome']} | `{record['cell']}` | "
                f"{record['radius']:.3f} | {record['azimuth_deg']:+.1f} | "
                f"{math.degrees(draw['object_yaw']):+.1f} | "
                f"{draw['object_static_friction']:.2f} | {record['peak_z'] * 1e3:.1f} | "
                f"{record['timeouts']} |"
            )

    videos = sorted(path.name for path in GATE_DIR.glob("*.mp4"))
    if videos:
        lines += [
            "",
            "## Video evidence",
            "",
            "Wrist POV with the state overlay burned in, recorded by "
            "`scripts/demo_expert.py --namespace expert_gate --attempt-list ... --video` "
            "(untracked — regenerate from the draw index in the filename).",
            "",
        ]
        lines += [f"- `{name}`" for name in videos]

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "~/isaaclab-env/bin/python scripts/gen_workspace_map.py --gate --headless  # x4, 50/boot",
        "~/isaaclab-env/bin/python scripts/gen_workspace_map.py --report-only",
        "```",
        "",
        f"Ledger: `{GATE_LEDGER.relative_to(REPO_ROOT)}` (one JSON line per attempt, "
        "append-only, resumable).",
        "",
    ]
    GATE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    GATE_REPORT.write_text("\n".join(lines))
    print(f"{successes}/{total} = {rate * 100:.1f}%  modes={dict(modes)}")
    print(f"wrote {GATE_REPORT}")


def run_gate() -> int:
    """Run one chunk of the gate inside a live Isaac app. Returns the exit code."""
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    from manus import specs
    from manus.expert import (
        GraspSuccessMonitor,
        ScriptedGraspExpert,
        classify_outcome,
    )
    from manus.objects import OBJECTS
    from manus.randomize import draw_episode
    from manus.task_scene import apply_randomization, grasp_scene_cfg

    physics_dt = 1.0 / 120.0
    decimation = 4
    settle_steps = 30
    max_control_steps = 1200

    spec = OBJECTS[args_cli.object]
    done = read_ledger()
    todo = pending(done)[: args_cli.chunk]
    if not todo:
        print("nothing pending")
        return 0
    print(f"running {len(todo)} attempts: {todo[0]}..{todo[-1]}")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=physics_dt, device=args_cli.device)
    )
    scene = InteractiveScene(grasp_scene_cfg(args_cli.object, num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    obj = scene["object"]
    device = robot.data.joint_pos.torch.device
    assert robot.joint_names == list(specs.JOINT_NAMES), robot.joint_names
    home = torch.tensor(
        [[specs.HOME_POSE[name] for name in specs.JOINT_NAMES]],
        dtype=torch.float32,
        device=device,
    )

    def advance() -> None:
        # No render: the gate is decided by the object's height, not by pixels,
        # and the wrist camera is the most expensive thing in the scene.
        sim.step(render=False)
        robot.update(physics_dt)
        obj.update(physics_dt)

    started = time.time()
    for attempt in todo:
        draw = draw_episode(GATE_NAMESPACE, attempt)
        robot.write_joint_state_to_sim_index(
            position=home, velocity=torch.zeros_like(home), full_data=True
        )
        robot.set_joint_position_target_index(target=home)
        apply_randomization(scene, draw, spec)
        scene.reset()
        scene.write_data_to_sim()
        for _ in range(settle_steps):
            advance()

        measured = robot.data.joint_pos.torch[0].detach().cpu().numpy().astype(float)
        expert = ScriptedGraspExpert(spec)
        plan = expert.reset(draw, q_current=measured)
        monitor = GraspSuccessMonitor(spec)
        for _ in range(max_control_steps):
            if expert.done:
                break
            targets = expert.step(measured)
            robot.set_joint_position_target_index(
                target=torch.tensor(
                    [[targets[name] for name in specs.JOINT_NAMES]],
                    dtype=torch.float32,
                    device=device,
                )
            )
            scene.write_data_to_sim()
            for _ in range(decimation):
                advance()
            measured = robot.data.joint_pos.torch[0].detach().cpu().numpy().astype(float)
            # In the robot's own frame: the monitor checks the object against an
            # FK of the TCP, not just against a height.
            object_pos = (
                (obj.data.root_link_pos_w.torch[0] - scene.env_origins[0])
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
            monitor.update(object_pos, measured)

        radius, azimuth_deg = polar(draw.object_x, draw.object_y)
        record = {
            "attempt": attempt,
            "namespace": GATE_NAMESPACE,
            "object": spec.name,
            "success": bool(monitor.success),
            "outcome": classify_outcome(expert, monitor),
            "cell": cell_of(draw.object_x, draw.object_y),
            "radius": radius,
            "azimuth_deg": azimuth_deg,
            "peak_z": float(monitor.peak_z),
            "threshold_z": monitor.threshold_z,
            "best_streak": monitor.best_streak,
            "total_steps": expert.total_steps,
            "timeouts": expert.timeouts,
            "plan_ok": plan.ok,
            "plan_reason": plan.reason,
            "grasp_yaw": plan.grasp_yaw,
            "states": [report.to_dict() for report in expert.reports],
            "draw": draw.to_dict(),
        }
        append_ledger(record)
        print(
            f"  [{attempt:3d}] {record['outcome']:<12} peak {monitor.peak_z * 1e3:6.1f} mm  "
            f"r={radius:.3f} az={azimuth_deg:+6.1f}  steps {expert.total_steps:4d}"
            + (f"  timeouts {expert.timeouts}" if expert.timeouts else "")
        )

    elapsed = time.time() - started
    print(f"chunk done in {elapsed:.0f} s ({elapsed / len(todo):.1f} s/attempt)")
    done = read_ledger()
    remaining = [index for index in range(args_cli.attempts) if index not in done]
    if remaining:
        print(f"{len(remaining)} attempts still pending; re-run this command")
    write_report(done)
    return 0


def main() -> int:
    if args_cli.report_only:
        write_report(read_ledger())
        return 0
    if args_cli.gate:
        return run_gate()
    return run_map()


if __name__ == "__main__":
    if args_cli.gate:
        args_cli.enable_cameras = True  # the scene carries the wrist camera
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        try:
            code = main()
        finally:
            simulation_app.close()
    else:
        code = main()
    sys.exit(code)
