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
        default=5.0,
        help="Maximum defect-defect distance in Angstrom (default: 5.0).",
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
        "-v", "--verbose",
        action="store_true",
        help="Verbose DEBUG-level logging + pipeline tracking.",
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
        log_file="pydefect_complex.log" if args.verbose else None,
    )

    dopants = args.dopants if args.dopants is not None else []
    logger.info(
        "START pydefect-complex (n=%d, dopants=%s, max_distance=%.1f, min_distance=%.2f)",
        args.n_body, dopants or "(intrinsic only)",
        args.max_distance, args.min_distance,
    )

    # ------------------------------------------------------------------
    # 1. Load supercell info
    # ------------------------------------------------------------------
    supercell_info = _load_supercell_info()

    # ------------------------------------------------------------------
    # 2. Build maker (HostGraph, enumerator, single-defect list)
    # ------------------------------------------------------------------
    from .maker import ComplexDefectMaker

    maker = ComplexDefectMaker(
        supercell_info,
        dopants=dopants,
        max_distance=args.max_distance,
        min_distance=args.min_distance,
        charges=args.charges,
        verbose=args.verbose,
        track_pipeline=args.verbose,
    )
    logger.info("DEFECT TYPES: %s", maker.defect_names)

    # ------------------------------------------------------------------
    # 3. Read existing complex_defect_in.yaml for skip logic
    # ------------------------------------------------------------------
    existing_yaml, existing_names = _load_existing_complex_defect_in()
    if existing_names:
        logger.info("EXISTING: %d entries in defect/complex_defect_in.yaml", len(existing_names))

    # ------------------------------------------------------------------
    # 4. Generate entries for all orders 2..n
    #    (generates geometries + compositions + structures in one pass)
    # ------------------------------------------------------------------
    from .enumerate import generate_all_entries
    from .symmetry import deduplicate
    from .io import write_all, write_complex_defect_in_yaml, write_summary

    all_entries = generate_all_entries(
        maker.enumerator, supercell_info,
        maker.single_defects, N_max=args.n_body,
        charges=args.charges,
        show_progress=args.verbose,
    )
    logger.info("GENERATED: %d entries (all orders 2..%d)", len(all_entries), args.n_body)

    # ------------------------------------------------------------------
    # 5. Cross-composition geometric deduplication
    # ------------------------------------------------------------------
    final_entries = deduplicate(
        all_entries, maker.host_graph, args.max_distance,
    )
    logger.info("DEDUP: %d entries after geometric deduplication", len(final_entries))

    # ------------------------------------------------------------------
    # 6. Filter: skip entries already recorded in complex_defect_in.yaml
    # ------------------------------------------------------------------
    new_entries = [e for e in final_entries if e.name not in existing_names]
    skipped = len(final_entries) - len(new_entries)
    if skipped:
        logger.info("SKIP: %d entries already exist, not overwritten", skipped)

    if not new_entries and not existing_names:
        logger.warning("No entries generated — check supercell_info.json and dopants")
        return

    # ------------------------------------------------------------------
    # 7. Write output
    # ------------------------------------------------------------------
    defect_dir = Path("defect")
    defect_dir.mkdir(parents=True, exist_ok=True)

    if new_entries:
        complex_defect_in_new = write_all(
            new_entries, str(defect_dir), create_defect_json=True,
        )
        logger.info(
            "POSCAR: wrote %d new entries to %s/",
            len(new_entries), defect_dir,
        )
    else:
        complex_defect_in_new = {}

    # Summary: covers all entries from this run
    write_summary(final_entries, str(defect_dir))

    # Parameters: current run metadata
    maker._write_parameters(str(defect_dir))

    # Merge complex_defect_in.yaml (preserve existing, append new)
    combined = dict(existing_yaml)
    combined.update(complex_defect_in_new)
    write_complex_defect_in_yaml(combined, str(defect_dir))
    logger.info(
        "OUTPUT: %s (%d entries, %d new, %d kept)",
        defect_dir / "complex_defect_in.yaml",
        len(combined), len(new_entries), len(existing_names),
    )

    # ------------------------------------------------------------------
    # 8. Geometry cache status
    # ------------------------------------------------------------------
    for order, geoms in maker.enumerator.geometries.items():
        logger.info(
            "CACHE: N=%d geometry cache: %d unique configurations (%.0f orientations)",
            order, len(geoms),
            sum(max(0, g.n_orientations) for g in geoms),
        )


if __name__ == "__main__":
    main()
