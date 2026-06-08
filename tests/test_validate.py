#!/usr/bin/env python3
"""Validation script for pydefect-complex generated structures.

Validates a ``defect/`` output directory:
  1. POSCAR files are parseable
  2. ``prior_info.yaml`` has a valid charge
  3. No atom overlap (min interatomic distance > 0.8 Å)

Usage as a CLI tool:
    python tests/test_validate.py /path/to/defect_output_dir

Usage as pytest:
    pytest tests/test_validate.py -v
"""

import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Validator (CLI + library)
# ---------------------------------------------------------------------------


def validate_output_dir(output_dir: str) -> dict:
    """Validate all POSCAR files in a pydefect-complex output directory.

    Returns dict with ``ok``, ``errors``, ``warnings`` keys.
    """
    import yaml
    from pymatgen.core import Structure

    output = Path(output_dir)
    errors = []
    warnings = []
    ok_count = 0

    for subdir in sorted(output.iterdir()):
        if not subdir.is_dir():
            continue

        poscar = subdir / "POSCAR"
        prior = subdir / "prior_info.yaml"

        if not poscar.exists():
            errors.append(f"{subdir.name}: missing POSCAR")
            continue

        try:
            structure = Structure.from_file(str(poscar))
        except Exception as e:
            errors.append(f"{subdir.name}: POSCAR parse error: {e}")
            continue

        # Check prior_info.yaml
        if prior.exists():
            try:
                info = yaml.safe_load(prior.read_text())
                charge = info.get("charge")
                if charge is None:
                    warnings.append(f"{subdir.name}: no charge in prior_info.yaml")
            except Exception:
                warnings.append(f"{subdir.name}: prior_info.yaml parse error")

        # Check minimum interatomic distance
        n = len(structure)
        lattice = structure.lattice
        min_dist = float("inf")
        min_pair = None
        for i in range(n):
            for j in range(i + 1, n):
                d = lattice.get_distance_and_image(
                    structure[i].frac_coords,
                    structure[j].frac_coords,
                )[0]
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)

        if min_dist < 0.8:
            warnings.append(
                f"{subdir.name}: atoms {min_pair} too close ({min_dist:.2f} Å)"
            )

        ok_count += 1

    return {
        "ok": ok_count,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Real integration tests (collected by pytest because filename starts with test_)
# ---------------------------------------------------------------------------

pytest.importorskip("pydefect")


def test_validate_output_dir_on_generated_defects(
    diamond_supercell_info, tmp_path,
):
    """Generate a real defect output, then validate it end-to-end.

    Exercises: ComplexDefectMaker → make_all_pairs → write → validate_output_dir.
    This is the only test that goes from a fresh supercell all the way through
    the public pipeline and confirms the output directory is structurally valid.
    """
    from pydefect_complex import ComplexDefectMaker

    out = tmp_path / "defect"
    maker = ComplexDefectMaker(
        diamond_supercell_info, max_distance=3.5,
    )
    maker.make_all_pairs()
    entries = maker.generate_entries(n_or_geometries=2)
    if not entries:
        pytest.skip("No entries generated with this cutoff")
    maker.write(entries, str(out))

    result = validate_output_dir(str(out))
    assert result["errors"] == [], f"Unexpected errors: {result['errors']}"
    assert result["ok"] == len(entries) * len(entries[0].complex_defect.charges)
    # Generated structures should be well-separated; no near-overlap warnings.
    assert result["warnings"] == [], f"Unexpected warnings: {result['warnings']}"


def test_validate_output_dir_flags_missing_poscar(tmp_path):
    """Validator should error on subdirs missing POSCAR."""
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "no_poscar_here").mkdir()

    result = validate_output_dir(str(bad))
    assert result["ok"] == 0
    assert any("missing POSCAR" in e for e in result["errors"])


def test_validate_output_dir_flags_unparseable_poscar(tmp_path):
    """Validator should error on subdirs with garbage POSCAR."""
    bad = tmp_path / "bad"
    bad.mkdir()
    sub = bad / "garbled"
    sub.mkdir()
    (sub / "POSCAR").write_text("not a poscar file at all\n")

    result = validate_output_dir(str(bad))
    assert result["ok"] == 0
    assert any("POSCAR parse error" in e for e in result["errors"])


def test_validate_output_dir_warns_no_charge(tmp_path):
    """Validator should warn when prior_info.yaml has no charge field."""
    sub = tmp_path / "no_charge"
    sub.mkdir()
    # Bare-minimum valid POSCAR (single H in a 10 Å cubic cell).
    (sub / "POSCAR").write_text(
        "H\n1.0\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nH\n1\nDirect\n0.5 0.5 0.5\n"
    )
    (sub / "prior_info.yaml").write_text("not_a_charge_key: 0\n")

    result = validate_output_dir(str(tmp_path))
    assert result["ok"] == 1
    assert any("no charge" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = validate_output_dir(sys.argv[1])
        print(f"Validated {result['ok']} directories")
        for e in result["errors"]:
            print(f"  ERROR: {e}")
        for w in result["warnings"]:
            print(f"  WARN: {w}")
        if not result["errors"]:
            print("All structures valid.")
    else:
        print("Usage: python test_validate.py /path/to/output_dir")
        print("Or:    pytest test_validate.py -v")
