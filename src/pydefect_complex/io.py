"""File I/O for complex defect structures.

Writes complex defect entries as pydefect-compatible directory structures
so they can be consumed by pydefect's post-processing pipeline:
  pydefect efnv -> pydefect des -> pydefect pe
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pydefect.input_maker.defect import SimpleDefect
    from .structure import ComplexDefectEntry
    from .core import ComplexDefect


def write_entry(
    entry: "ComplexDefectEntry",
    output_dir: str,
    charge: int,
    create_defect_json: bool = False,
) -> str:
    """Write a single ComplexDefectEntry as a pydefect-compatible directory.

    Creates: {output_dir}/{name}_{charge}/
               POSCAR
               prior_info.yaml
               defect_entry.json (optional)

    Returns the absolute path to the created directory.
    """
    dir_name = f"{entry.name}_{charge}"
    abs_path = Path(output_dir) / dir_name
    abs_path.mkdir(parents=True, exist_ok=True)

    # Write POSCAR
    entry.structure.to(fmt="poscar", filename=str(abs_path / "POSCAR"))

    # Write prior_info.yaml (charge state)
    with open(abs_path / "prior_info.yaml", "w") as f:
        yaml.dump({"charge": charge}, f, default_flow_style=False)

    # Write defect_entry.json for pydefect integration
    if create_defect_json:
        _write_defect_entry_json(abs_path, entry, charge)

    return str(abs_path)


def write_all(
    entries: list["ComplexDefectEntry"],
    output_dir: str,
    perfect_poscar_path: Optional[str] = None,
    create_defect_json: bool = True,
) -> dict[str, list[int]]:
    """Write all complex defect entries to a pydefect-compatible directory.

    Args:
        entries: List of ComplexDefectEntry to write.
        output_dir: Target directory (typically 'defect/').
        perfect_poscar_path: Path to perfect supercell POSCAR for
            defect_entry.json generation.
        create_defect_json: Whether to generate defect_entry.json files.

    Returns:
        A dict mapping defect names to their charge lists,
        suitable for writing as complex_defect_in.yaml.
    """
    complex_defect_in: dict[str, list[int]] = {}

    for entry in entries:
        cd = entry.complex_defect
        defect_name = entry.name  # e.g., "v_C_1+v_C_2.C_5"
        complex_defect_in[defect_name] = cd.charges

        for charge in cd.charges:
            write_entry(entry, output_dir, charge, create_defect_json)

    return complex_defect_in


def write_complex_defect_in_yaml(
    complex_defect_in: dict[str, list[int]],
    output_dir: str,
) -> str:
    """Write complex_defect_in.yaml for pydefect pipeline integration.

    This file has the same format as defect_in.yaml and can be merged
    with it to include complex defects in the standard pydefect workflow.
    """
    path = Path(output_dir) / "complex_defect_in.yaml"
    with open(path, "w") as f:
        yaml.dump(complex_defect_in, f, default_flow_style=None)
    return str(path)


def merge_defect_in(
    output_dir: str,
    defect_in_path: str = "defect_in.yaml",
) -> str:
    """Merge complex_defect_in.yaml into defect_in.yaml.

    Backs up the original defect_in.yaml as defect_in.yaml.bak.
    """
    output = Path(output_dir)
    complex_path = output / "complex_defect_in.yaml"
    defect_path = output / defect_in_path

    if not complex_path.exists():
        raise FileNotFoundError(
            f"{complex_path} not found. Run write_all() first."
        )

    complex_defects = yaml.safe_load(complex_path.read_text())

    if defect_path.exists():
        defect_in = yaml.safe_load(defect_path.read_text())
        # Backup
        backup_path = output / f"{defect_in_path}.bak"
        defect_path.rename(backup_path)
    else:
        defect_in = {}

    defect_in.update(complex_defects)

    with open(defect_path, "w") as f:
        yaml.dump(defect_in, f, default_flow_style=None)

    return str(defect_path)


# ---------------------------------------------------------------------------
# Human-readable defect summary
# ---------------------------------------------------------------------------


def _orient_str(n: int) -> str:
    if n < 0:
        return "—"
    return str(n)


def _dists_str(distances) -> str:
    if not distances:
        return "—"
    return " → ".join(f"{d:.2f}" for d in distances)


def write_summary(
    entries: list["ComplexDefectEntry"],
    output_dir: str,
    filename: str = "defect_summary.txt",
) -> str:
    """Write a human-readable summary of all complex defect entries.

    Lists each defect with its key properties: distances, symmetry,
    orientations, and composition.  Complements the machine-readable
    complex_defect_in.yaml.
    """
    from collections import Counter

    path = Path(output_dir) / filename

    lines = []
    lines.append("=" * 78)
    lines.append("Complex Defect Summary")
    lines.append("=" * 78)
    lines.append("")

    # Group by composition
    by_comp: dict[str, list] = {}
    for e in entries:
        by_comp.setdefault(e.complex_defect.name, []).append(e)

    for comp_name in sorted(by_comp):
        comp_entries = by_comp[comp_name]
        lines.append(f"[{comp_name}]  ({len(comp_entries)} configurations)")
        lines.append("-" * 78)
        lines.append(
            f"  {'Name':<28s} {'Formula':>10s} {'Atoms':>6s} "
            f"{'Dists (Å)':>20s} {'PG':>6s} {'SG':>12s} {'Orient':>6s}"
        )
        lines.append("  " + "-" * 76)

        for e in comp_entries:
            formula = str(e.structure.composition.formula) if e.structure else "?"
            n_atoms = len(e.structure) if e.structure else 0
            dists = _dists_str(e.distances)
            pg = e.point_group or "?"
            sg = e.space_group or "?"
            no = _orient_str(e.n_orientations)

            lines.append(
                f"  {e.name:<28s} {formula:>10s} {n_atoms:>6d} "
                f"{dists:>20s} {pg:>6s} {sg:>12s} {no:>6s}"
            )
        lines.append("")

    # Overall stats
    n_comp = len(by_comp)
    n_total = len(entries)
    pg_counts = Counter(e.point_group for e in entries if e.point_group)

    lines.append("=" * 78)
    lines.append(f"Total: {n_total} entries in {n_comp} compositions")
    if pg_counts:
        lines.append("Point group distribution:")
        for pg, count in pg_counts.most_common():
            lines.append(f"  {pg:>6s}: {count}")
    lines.append("=" * 78)

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return str(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_defect_entry_json(
    abs_path: Path,
    entry: "ComplexDefectEntry",
    charge: int,
):
    """Generate a minimal defect_entry.json for pydefect compatibility.

    The full defect_entry.json is normally created by pydefect_vasp de.
    This writes a placeholder that pydefect can recognize and later
    enrich with calculation results.
    """
    defect_entry = {
        "name": entry.name,
        "charge": charge,
        "full_name": f"{entry.name}_{charge}",
        "defect_center": None,
    }
    with open(abs_path / "defect_entry.json", "w") as f:
        json.dump(defect_entry, f, indent=2)