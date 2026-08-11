"""Patch the converted SO-101 USD so the two jaws collide on their real surface.

WHY
---
``assets/so101/usd/`` is **converter output** (Isaac Lab's ``convert_urdf.py``).
Its ``config.yaml`` records ``collision_type: Convex Hull``, and every collider
in ``payloads/instances.usda`` carries ``physics:approximation = "convexHull"``.
For the two jaws that is badly wrong: both parts are hooked/stepped, so their
convex hull spans from the fingertip up to the mounting body and cuts a wedge
across the grasp opening. Measured off the vendored STLs (Step 19, in the TCP
frame the expert's constants live in):

* ``wrist_roll_follower_so101_v1`` (the FIXED jaw) -- hull sits **6.6 mm**
  proud of the visual pad face at the fingertip step, rising to **8.6 mm** at
  the top of the band a 30 mm cube occupies. Hull volume is 2.58x the mesh's.
* ``moving_jaw_so101_v1`` -- hull sits **1.8-2.1 mm** proud. Volume 2.33x.

Cross-check that this is the real cause: the narrowest *visual* opening reaches
30 mm at jaw angle 0.195 rad, the narrowest *hull* opening only at 0.320 rad --
and the Step 8 gate measured the loaded jaws stalling at 0.27-0.35 rad. Physics
is clamping the cube on a surface several millimetres outside the one the
camera renders, which is exactly the gap the user sees on the fixed jaw.

WHAT THIS CHANGES
-----------------
Only the two jaw collision meshes, only in ``payloads/instances.usda``:

* ``physics:approximation``: ``convexHull`` -> ``sdf`` (or
  ``convexDecomposition`` with ``--approximation``),
* the matching PhysX cooking schema is applied and its tuning attributes are
  authored (``PhysxSDFMeshCollisionAPI`` /
  ``PhysxConvexDecompositionCollisionAPI``),
* optionally ``physxCollision:contactOffset`` / ``restOffset`` (off by default
  -- see "OFFSETS" below).

Every other collider (arm links, servo bodies) keeps ``convexHull``: nothing
else touches the object, self-collision is off, and hulls are cheaper.

WHY ``instances.usda`` AND NOT AN OVER IN THE ROOT LAYER
--------------------------------------------------------
``config.yaml`` has ``make_instanceable: true``, so ``payloads/base.usda``
marks each collider Xform ``instanceable = true``. USD forbids overriding
properties on prims inside an instance from an outer layer, so the edit has to
land on the instance *source*: the ``over`` blocks inside
``payloads/instances.usda``. Those prims are::

    /Instances/wrist_roll_follower_so101_v1_1/wrist_roll_follower_so101_v1
    /Instances/moving_jaw_so101_v1_1/moving_jaw_so101_v1

(the ``_1`` suffix is the converter's collision copy of each visual instance;
the un-suffixed sibling is the render mesh and is left alone).

SDF vs CONVEX DECOMPOSITION
---------------------------
Both jaws are dynamic bodies -- links of an articulation driven by position
targets, not kinematic actors -- so a plain triangle mesh is not an option:
PhysX rejects it outright ("attachShape: non-SDF triangle mesh ... not
supported for non-kinematic PxRigidDynamic instances"). The dynamic side of
the contact, the cube, is a box primitive and needs no change at all.

That leaves SDF (true surface, sampled on a grid) or convex decomposition
(N convex pieces). SDF is the default here because:

* both STLs are watertight and winding-consistent (checked with trimesh), the
  precondition PhysX warns about for SDF cooking;
* the largest AABB extent is 105.4 mm (fixed jaw) / 92.0 mm (moving jaw), so
  ``sdfResolution = 256`` gives 0.41 / 0.36 mm grid spacing -- an order of
  magnitude below the 6.6 mm error being removed, and below the 0.5-2 mm steps
  in the pad faces that the fix is meant to reproduce;
* a decomposition still leaves millimetres on the table: slicing the fixed jaw
  into 4-32 convex pieces (an axis-aligned stand-in for VHACD) drops the
  *median* pad bulge to ~0 but leaves a **5.0 mm maximum**, and contact is
  decided by the maximum, not the median. VHACD's own voxel grid
  (500 000 voxels over this part = 0.94 mm) sets a floor too.

Use ``--approximation convexDecomposition`` if Step 20 finds SDF unusable
(cook time, memory, or poor box-vs-SDF contact patches); the parameters that
path writes are tuned for this part (64 hulls, 2 M voxels, shrink-wrap on,
1% error) rather than left at PhysX defaults.

OFFSETS
-------
Nothing in the USD authors ``physxCollision:contactOffset`` or ``restOffset``,
and ``src/manus/objects.py`` sets only ``collision_enabled=True`` on its
``PhysxCollisionPropertiesCfg`` -- so both stay at the schema default
``-inf``, i.e. PhysX autocompute. That is *not* a contributor to the visible
gap: ``restOffset`` is what holds bodies apart at rest and its effective value
is 0, while ``contactOffset`` only decides how early contacts are *generated*.
So this script leaves them alone by default. ``--contact-offset`` /
``--rest-offset`` are provided for Step 20 in case the probe shows contacts
resting at a positive separation; the matching change for the cube belongs in
``manus.objects.ObjectSpec.make_spawn_cfg``'s
``PhysxCollisionPropertiesCfg``, not here.

REGENERATION IS DESTRUCTIVE
---------------------------
Re-running the URDF->USD conversion **overwrites** ``instances.usda`` and puts
``convexHull`` back: ``UrdfConverterCfg.collision_type`` only offers
Convex Hull / Convex Decomposition / Bounding Sphere / Bounding Cube -- there
is no SDF option, and ``scripts/tools/convert_urdf.py`` exposes no collision
flag at all. So this patch is a mandatory post-conversion step. It is recorded
in ``README.md`` next to the regenerate command; keep the two in sync.

USAGE
-----
CPU only -- pure ``pxr``, no Isaac Sim app, safe to run while the GPU is busy.

.. code-block:: bash

    PY=~/isaaclab-env/bin/python

    $PY scripts/fix_jaw_collision.py --dry-run     # show the diff, write nothing
    $PY scripts/fix_jaw_collision.py               # apply (idempotent)
    $PY scripts/fix_jaw_collision.py --revert      # restore the backup

The first apply copies ``instances.usda`` to ``instances.usda.orig-convexhull``
and never overwrites that backup, so ``--revert`` always returns to the
as-converted state.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

from pxr import Sdf, Usd

REPO_ROOT = Path(__file__).resolve().parents[1]

INSTANCES_USDA = (
    REPO_ROOT / "assets" / "so101" / "usd" / "so101_new_calib" / "payloads" / "instances.usda"
)
"""Layer holding the collider instance masters -- the only file this patches."""

BASE_USDA = (
    REPO_ROOT / "assets" / "so101" / "usd" / "so101_new_calib" / "payloads" / "base.usda"
)
"""Layer marking the collider Xforms ``instanceable``; only ``--deinstance`` touches it."""

BACKUP_SUFFIX = ".orig-convexhull"
"""Appended to a patched layer's name for the pristine converter output."""

JAW_COLLIDERS: dict[str, str] = {
    "fixed": "/Instances/wrist_roll_follower_so101_v1_1/wrist_roll_follower_so101_v1",
    "moving": "/Instances/moving_jaw_so101_v1_1/moving_jaw_so101_v1",
}
"""The two collision meshes to patch, by the jaw they belong to."""

JAW_INSTANCE_XFORMS: dict[str, str] = {
    "fixed": (
        "/so101_new_calib/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link"
        "/wrist_link/gripper_link/wrist_roll_follower_so101_v1_1"
    ),
    "moving": (
        "/so101_new_calib/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link"
        "/wrist_link/gripper_link/moving_jaw_so101_v1_link/moving_jaw_so101_v1_1"
    ),
}
"""Composed paths of the Xforms ``base.usda`` marks instanceable (``--deinstance``)."""

# Attribute names and types transcribed from the installed PhysX schema at
# omni.usd.schema.physx-110.1.13/plugins/PhysxSchema/resources/generatedSchema.usda
# (classes "PhysxSDFMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI",
# "PhysxCollisionAPI"). PhysxSchema itself is NOT importable outside the Kit
# runtime -- its .so links against Kit's USD build, not the pip one -- so the
# attributes are authored generically through Sdf, which produces byte-identical
# USD to what the schema writer would emit.
SDF_SCHEMA = "PhysxSDFMeshCollisionAPI"
DECOMPOSITION_SCHEMA = "PhysxConvexDecompositionCollisionAPI"
PHYSX_COLLISION_SCHEMA = "PhysxCollisionAPI"

SDF_DEFAULTS: dict[str, tuple[object, object]] = {
    # name -> (Sdf value type, schema default) -- only non-defaults get written
    "physxSDFMeshCollision:sdfResolution": (Sdf.ValueTypeNames.Int, 256),
    "physxSDFMeshCollision:sdfSubgridResolution": (Sdf.ValueTypeNames.Int, 6),
    "physxSDFMeshCollision:sdfNarrowBandThickness": (Sdf.ValueTypeNames.Float, 0.01),
    "physxSDFMeshCollision:sdfMargin": (Sdf.ValueTypeNames.Float, 0.01),
}

DECOMPOSITION_ATTRS: dict[str, tuple[object, object]] = {
    # Tuned for a ~100 mm printed part with a stepped pad, not left at PhysX
    # defaults (32 hulls / 500k voxels / no shrink-wrap / 10% error).
    "physxConvexDecompositionCollision:maxConvexHulls": (Sdf.ValueTypeNames.Int, 64),
    "physxConvexDecompositionCollision:hullVertexLimit": (Sdf.ValueTypeNames.Int, 64),
    "physxConvexDecompositionCollision:voxelResolution": (Sdf.ValueTypeNames.Int, 2_000_000),
    "physxConvexDecompositionCollision:errorPercentage": (Sdf.ValueTypeNames.Float, 1.0),
    "physxConvexDecompositionCollision:shrinkWrap": (Sdf.ValueTypeNames.Bool, True),
    "physxConvexDecompositionCollision:minThickness": (Sdf.ValueTypeNames.Float, 0.001),
}


class Change:
    """One authored difference, so ``--dry-run`` and the apply log agree."""

    def __init__(self, path: str, what: str, before: object, after: object) -> None:
        self.path = path
        self.what = what
        self.before = before
        self.after = after

    def __str__(self) -> str:
        return f"    {self.what:<52} {self.before!r} -> {self.after!r}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--approximation",
        choices=("sdf", "convexDecomposition", "convexHull"),
        default="sdf",
        help="collision approximation to author on the two jaws (default: sdf)",
    )
    parser.add_argument(
        "--sdf-resolution",
        type=int,
        default=256,
        help=(
            "physxSDFMeshCollision:sdfResolution. Grid spacing = largest AABB extent / "
            "resolution; the fixed jaw's extent is 105.4 mm, so 256 -> 0.41 mm"
        ),
    )
    parser.add_argument(
        "--sdf-subgrid-resolution",
        type=int,
        default=6,
        help="physxSDFMeshCollision:sdfSubgridResolution; 0 = dense SDF (much more memory)",
    )
    parser.add_argument(
        "--contact-offset",
        type=float,
        default=None,
        help=(
            "author physxCollision:contactOffset [m] on the two jaws. Default: leave "
            "unauthored (PhysX autocompute). Not a cause of the visible gap"
        ),
    )
    parser.add_argument(
        "--rest-offset",
        type=float,
        default=None,
        help="author physxCollision:restOffset [m] on the two jaws. Default: unauthored",
    )
    parser.add_argument(
        "--deinstance",
        action="store_true",
        help=(
            "also clear instanceable=true on the two collider Xforms in base.usda. "
            "Fallback only, for if PhysX refuses to cook an SDF inside an instance "
            "prototype -- costs one extra shape, changes nothing else"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="print changes, write nothing")
    parser.add_argument(
        "--revert",
        action="store_true",
        help=f"restore the *{BACKUP_SUFFIX} backups and exit",
    )
    return parser.parse_args(argv)


def backup_path(layer: Path) -> Path:
    return layer.with_name(layer.name + BACKUP_SUFFIX)


def ensure_backup(layer: Path, dry_run: bool) -> None:
    """Copy the pristine converter output aside, once and only once."""
    destination = backup_path(layer)
    if destination.exists():
        print(f"  backup already present: {destination.name}")
        return
    if dry_run:
        print(f"  would back up {layer.name} -> {destination.name}")
        return
    shutil.copy2(layer, destination)
    print(f"  backed up {layer.name} -> {destination.name}")


def revert() -> int:
    restored = 0
    for layer in (INSTANCES_USDA, BASE_USDA):
        source = backup_path(layer)
        if not source.exists():
            continue
        shutil.copy2(source, layer)
        print(f"  restored {layer.name} from {source.name}")
        restored += 1
    if not restored:
        print("  nothing to revert: no backups found")
        return 1
    return 0


def applied_schemas(spec: Sdf.PrimSpec) -> list[str]:
    """The prim's authored ``apiSchemas`` list, read straight off the layer.

    Not ``Usd.Prim.GetAppliedSchemas()``: the PhysX and Newton schema plugins
    are not registered in a bare ``pxr`` install, so USD silently drops the
    entries it cannot resolve. The raw list op always has all of them.
    """
    if spec is None or "apiSchemas" not in spec.ListInfoKeys():
        return []
    listop = spec.GetInfo("apiSchemas")
    return list(listop.explicitItems) + list(listop.prependedItems) + list(listop.appendedItems)


def set_applied_schemas(spec: Sdf.PrimSpec, schemas: list[str]) -> None:
    spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas))


def _same(before: object, after: object) -> bool:
    """Whether an authored value already equals the target.

    Floats need a tolerance: USD stores these attributes as 32-bit, so a
    written ``0.01`` reads back as ``0.009999999776482582`` and a naive ``==``
    would rewrite the layer on every run -- i.e. would not be idempotent.
    """
    if isinstance(before, float) and isinstance(after, float):
        return math.isclose(before, after, rel_tol=1e-6, abs_tol=1e-12)
    return before == after


def author_attribute(
    prim: Usd.Prim, name: str, type_name, value, changes: list[Change]
) -> None:
    """Idempotently write one uniform attribute, recording the change."""
    attribute = prim.GetAttribute(name)
    before = attribute.Get() if attribute and attribute.HasAuthoredValue() else None
    if _same(before, value):
        return
    if not attribute:
        attribute = prim.CreateAttribute(name, type_name, False, Sdf.VariabilityUniform)
    attribute.Set(value)
    changes.append(Change(str(prim.GetPath()), name, before, value))


def clear_attribute(prim: Usd.Prim, name: str, changes: list[Change]) -> None:
    """Remove an attribute this script may have written on an earlier run."""
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValue():
        return
    before = attribute.Get()
    prim.RemoveProperty(name)
    changes.append(Change(str(prim.GetPath()), name, before, "<removed>"))


def patch_jaw(stage: Usd.Stage, jaw: str, path: str, args: argparse.Namespace) -> list[Change]:
    """Author the chosen approximation (and its cooking schema) on one jaw."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise SystemExit(
            f"FATAL: {path} is not in {INSTANCES_USDA.name}. The USD layout changed "
            "(re-conversion?); re-derive the collider paths before patching."
        )
    if prim.GetTypeName() != "Mesh":
        raise SystemExit(f"FATAL: {path} is a {prim.GetTypeName()}, not a Mesh.")

    spec = stage.GetRootLayer().GetPrimAtPath(path)
    changes: list[Change] = []

    # 1. the approximation token itself
    approximation = prim.GetAttribute("physics:approximation")
    before = approximation.Get() if approximation else None
    if before != args.approximation:
        if not approximation:
            approximation = prim.CreateAttribute(
                "physics:approximation", Sdf.ValueTypeNames.Token, False, Sdf.VariabilityVarying
            )
        approximation.Set(args.approximation)
        changes.append(Change(path, "physics:approximation", before, args.approximation))

    # 2. the cooking schema + its tuning attributes, and the *other* family's
    #    leftovers cleared so re-running with a different --approximation is
    #    still idempotent rather than cumulative.
    schemas = applied_schemas(spec)
    wanted = {
        "sdf": SDF_SCHEMA,
        "convexDecomposition": DECOMPOSITION_SCHEMA,
        "convexHull": None,
    }[args.approximation]
    unwanted = {SDF_SCHEMA, DECOMPOSITION_SCHEMA} - {wanted}

    if args.approximation == "sdf":
        values = dict(SDF_DEFAULTS)
        values["physxSDFMeshCollision:sdfResolution"] = (
            Sdf.ValueTypeNames.Int,
            args.sdf_resolution,
        )
        values["physxSDFMeshCollision:sdfSubgridResolution"] = (
            Sdf.ValueTypeNames.Int,
            args.sdf_subgrid_resolution,
        )
    elif args.approximation == "convexDecomposition":
        values = dict(DECOMPOSITION_ATTRS)
    else:
        values = {}

    for name, (type_name, value) in values.items():
        author_attribute(prim, name, type_name, value, changes)
    for stale in unwanted:
        namespace = {
            SDF_SCHEMA: "physxSDFMeshCollision:",
            DECOMPOSITION_SCHEMA: "physxConvexDecompositionCollision:",
        }[stale]
        for name in list(SDF_DEFAULTS) + list(DECOMPOSITION_ATTRS):
            if name.startswith(namespace):
                clear_attribute(prim, name, changes)

    # 3. offsets, only if asked for
    for flag, name in (
        (args.contact_offset, "physxCollision:contactOffset"),
        (args.rest_offset, "physxCollision:restOffset"),
    ):
        if flag is not None:
            author_attribute(prim, name, Sdf.ValueTypeNames.Float, float(flag), changes)

    # 4. keep apiSchemas in step with what is authored
    desired = [name for name in schemas if name not in unwanted]
    if wanted is not None and wanted not in desired:
        desired.append(wanted)
    if args.contact_offset is not None or args.rest_offset is not None:
        if PHYSX_COLLISION_SCHEMA not in desired:
            desired.append(PHYSX_COLLISION_SCHEMA)
    if desired != schemas:
        set_applied_schemas(spec, desired)
        changes.append(Change(path, "apiSchemas", schemas, desired))

    if changes:
        print(f"  [{jaw}] {path}")
        for change in changes:
            print(change)
    else:
        print(f"  [{jaw}] {path}\n    already patched, nothing to do")
    return changes


def deinstance(args: argparse.Namespace) -> list[Change]:
    """Clear ``instanceable`` on the two collider Xforms in ``base.usda``."""
    ensure_backup(BASE_USDA, args.dry_run)
    stage = Usd.Stage.Open(str(BASE_USDA))
    changes: list[Change] = []
    for jaw, path in JAW_INSTANCE_XFORMS.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise SystemExit(f"FATAL: {path} is not in {BASE_USDA.name}.")
        if not prim.IsInstanceable():
            continue
        prim.SetInstanceable(False)
        changes.append(Change(path, "instanceable", True, False))
        print(f"  [{jaw}] {path}\n    instanceable true -> false")
    if changes and not args.dry_run:
        stage.GetRootLayer().Save()
    return changes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.revert:
        return revert()

    if not INSTANCES_USDA.exists():
        raise SystemExit(f"FATAL: {INSTANCES_USDA} does not exist.")

    print(f"layer: {INSTANCES_USDA}")
    print(f"target approximation: {args.approximation}")
    if args.approximation == "sdf":
        print(
            f"  sdfResolution={args.sdf_resolution} "
            f"(grid spacing = 105.4 mm / {args.sdf_resolution} = "
            f"{105.4 / args.sdf_resolution:.3f} mm on the fixed jaw, "
            f"92.0 / {args.sdf_resolution} = {92.0 / args.sdf_resolution:.3f} mm on the moving jaw)"
        )
    ensure_backup(INSTANCES_USDA, args.dry_run)

    stage = Usd.Stage.Open(str(INSTANCES_USDA))
    changes: list[Change] = []
    for jaw, path in JAW_COLLIDERS.items():
        changes.extend(patch_jaw(stage, jaw, path, args))

    if args.deinstance:
        changes.extend(deinstance(args))

    if not changes:
        print("\nno changes needed -- the USD already matches the requested state")
        return 0
    if args.dry_run:
        print(f"\nDRY RUN: {len(changes)} change(s) NOT written")
        return 0

    stage.GetRootLayer().Save()
    print(f"\nwrote {len(changes)} change(s) to {INSTANCES_USDA.name}")
    print(
        "\nNEXT:\n"
        "  1. re-run scripts/contact_probe.py --label after and compare with --label before\n"
        "     (fixed-jaw contacts should move from x ~= -6.6 mm to x ~= 0.0 mm in the TCP frame)\n"
        "  2. eyeball runs/contact_probe/after_hold.png -- the fixed-jaw gap should be gone\n"
        "  3. physics changed: re-measure the loaded jaw stall angle (predicted ~0.195 rad,\n"
        "     was 0.27-0.35) and re-tune close_target_rad / TCP_TO_PAD_CENTRE, then re-run\n"
        "     the >=200-attempt expert gate\n"
        "  4. this patch does NOT survive re-running convert_urdf.py -- re-apply it (README)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
