"""pydefect-complex CLI — systematic complex defect generation for pydefect.

Reads ``supercell_info.json`` from the current working directory and outputs
to ``defect/``, following pydefect's file conventions.

Usage::

    pydefect_complex -d N B -n 2
    pydefect_complex -d N B -n 3
    pydefect_complex --verbose -d P -n 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .log import configure_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pydefect_complex",
        description="Systematic complex defect generation for pydefect. "
                    "Reads supercell_info.json from CWD, outputs to defect/.",
    )

    parser.add_argument(
        "-d", "--dopants",
        nargs="*",
        default=None,
        metavar="EL",
        help="Dopant element symbols, e.g. ``-d N B``. Omit for intrinsic only.",
    )
    parser.add_argument(
        "-n", "--n-body",
        type=int,
        default=2,
        help="Maximum N-body order (default: 2). Generates all orders 2..n.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=3.0,
        help="Maximum defect-defect distance in Angstrom (default: 3.0).",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.3,
        help="Minimum defect-defect distance in Angstrom (default: 0.3).",
    )
    parser.add_argument(
        "--charges",
        nargs="*",
        type=int,
        default=[0],
        help="Charge states to generate (default: 0).",
    )
    parser.add_argument(
        "-g", "--geometries-only",
        action="store_true",
        help="Only enumerate geometries, save cache, and exit. No entries or output files.",
    )
    parser.add_argument(
        "--structures",
        action="store_true",
        help="Write per-defect POSCAR directories (default: only YAML + summary).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose DEBUG-level logging + pipeline tracking.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = CPU count, default).",
    )

    parsed = parser.parse_args(argv)

    if parsed.n_body < 2:
        parser.error("n-body must be >= 2")

    return parsed


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _load_supercell_info(path: str = "supercell_info.json"):
    """Load pydefect SupercellInfo from JSON."""
    from pydefect.input_maker.supercell_info import SupercellInfo

    p = Path(path)
    if not p.exists():
        logger.error("Required file not found: %s (run `pydefect supercell` first)", p.resolve())
        sys.exit(1)

    with open(p) as f:
        data = json.load(f)
    si = SupercellInfo.from_dict(data)
    logger.info(
        "SUPERCELL: %s (%d atoms, space group %s)",
        p.resolve(), len(si.structure), si.space_group,
    )
    return si


def _load_existing_complex_defect_in() -> tuple[dict[str, list[int]], set[str]]:
    """Read existing complex_defect_in.yaml from defect/ dir."""
    path = Path("defect") / "complex_defect_in.yaml"
    if not path.exists():
        return {}, set()
    data = yaml.safe_load(path.read_text()) or {}
    return data, set(data.keys())


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    configure_logging(
        verbose=args.verbose,
        log_file="pydefect_complex.log",
    )

    dopants = args.dopants if args.dopants is not None else []
    n = args.n_body

    # Log the full command as typed by the user
    cmd = "pydefect_complex " + " ".join(argv if argv is not None else sys.argv[1:])
    logger.info("COMMAND: %s", cmd)

    # ==========================================================
    # 1. Input
    # ==========================================================
    supercell_info = _load_supercell_info()
    logger.info("ARGS: n_body=%d dopants=%s max_distance=%.1f min_distance=%.2f charges=%s",
                n, dopants or "(intrinsic)", args.max_distance, args.min_distance, args.charges)

    # ==========================================================
    # 2. Defect set (from pydefect DefectSetMaker)
    # ==========================================================
    from .maker import ComplexDefectMaker
    n_workers = args.workers if args.workers > 0 else None
    maker = ComplexDefectMaker(
        supercell_info, dopants=dopants,
        max_distance=args.max_distance, min_distance=args.min_distance,
        charges=args.charges, verbose=args.verbose,
        skip_defects=args.geometries_only,
    )
    if not args.geometries_only:
        logger.info("DEFECTS: %d types: %s", len(maker.single_defects), maker.defect_names)

    # ==========================================================
    # 3. Existing entries (skip logic)
    # ==========================================================
    existing_yaml, existing_names = _load_existing_complex_defect_in()
    logger.info("EXISTING: %d entries in defect/complex_defect_in.yaml%s",
                len(existing_names),
                " (skip)" if existing_names else "")

    # ==========================================================
    # 4. Geometry cache (cross-process)
    # ==========================================================
    cached_orders = maker.load_geometry_cache("defect")
    missing: set[int] = set()
    for order in range(2, n + 1):
        if order not in cached_orders:
            missing.add(order)
    if not missing:
        logger.info("GEOMETRY: N=%d..%d fully cached, no enumeration needed", 2, n)
    else:
        logger.info("GEOMETRY: enumerate N=%s (cached: N=%s)", sorted(missing), sorted(cached_orders) if cached_orders else "none")

    # ==========================================================
    # 5. Ensure geometries are enumerated (needed before early exit)
    # ==========================================================
    maker.enumerator.enumerate(n)

    # ==========================================================
    # 6. Early exit: geometries only (no entries, no output)
    # ==========================================================
    if args.geometries_only:
        maker.save_geometry_cache("defect")
        for order, geoms in maker.enumerator.geometries.items():
            n_orient = sum(max(0, g.n_orientations) for g in geoms)
            logger.info("CACHE: N=%d %d geometries %d orientations",
                        order, len(geoms), n_orient)
        logger.info("GEOMETRIES ONLY: saved to defect/geometries_N*.yaml")
        return

    # ==========================================================
    # 7. Generate entries via Maker API (dedup happens internally)
    # ==========================================================
    final_entries = maker.generate_entries(
        n_or_geometries=n, charges=args.charges,
    )

    # ==========================================================
    # 8. Filter new entries (only those not already on disk)
    # ==========================================================
    new_entries = [e for e in final_entries if e.name not in existing_names]
    kept = len(existing_names)
    skipped = len(final_entries) - len(new_entries)
    logger.info("ENTRIES: %d after dedup, %d new, %d skipped (already exist)",
                len(final_entries), len(new_entries), skipped)

    if not new_entries and not existing_names:
        logger.warning("EMPTY: no entries generated — check supercell_info.json and dopants")
        return

    # ==========================================================
    # 9. Write output (Maker.write handles dedup-aware file I/O)
    # ==========================================================
    defect_dir = Path("defect").resolve()
    defect_dir.mkdir(parents=True, exist_ok=True)

    if new_entries and args.structures:
        maker.write(new_entries, str(defect_dir), merge=False)
        complex_defect_in_new = {
            e.name: e.complex_defect.charges for e in new_entries
        }
    elif new_entries:
        complex_defect_in_new = {
            e.name: e.complex_defect.charges for e in new_entries
        }
    else:
        complex_defect_in_new = {}

    from .io import write_summary, write_complex_defect_in_yaml
    write_summary(final_entries, str(defect_dir))
    maker._write_parameters(str(defect_dir))

    combined = dict(existing_yaml)
    combined.update(complex_defect_in_new)
    write_complex_defect_in_yaml(combined, str(defect_dir))
    logger.info("OUTPUT: %s (%d total, %d new, %d kept)",
                defect_dir / "complex_defect_in.yaml",
                len(combined), len(new_entries), kept)
    if not args.structures and new_entries:
        logger.info("STRUCTURES: skipped (use --structures to write POSCAR dirs)")

    # ==========================================================
    # 11. Save geometry cache for next run
    # ==========================================================
    maker.save_geometry_cache(str(defect_dir))
    for order, geoms in maker.enumerator.geometries.items():
        n_orient = sum(max(0, g.n_orientations) for g in geoms)
        logger.info("CACHE: N=%d %d geometries %d orientations",
                    order, len(geoms), n_orient)


if __name__ == "__main__":
    main()
