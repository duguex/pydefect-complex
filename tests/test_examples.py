"""End-to-end smoke tests for the examples/ directory.

Each examples/<name>/ folder contains a real supercell_info.json that was
used to generate the included defect/ output. These tests re-run the CLI
on a tmp copy of the example input and assert that the pipeline still
produces a sensible output — catching regressions in:

- CLI argument parsing
- supercell_info.json reading
- geometry enumeration
- structure generation
- YAML output format

The tests are gated on each example existing on disk (the untracked
folders may be absent in minimal checkouts). Geometry-only mode (-g) is
used to keep the tests fast; the goal is end-to-end coverage, not
verifying a specific result.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def _example_dirs() -> list[str]:
    """Return names of example dirs that have a supercell_info.json."""
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in EXAMPLES_DIR.iterdir()
        if d.is_dir() and (d / "supercell_info.json").is_file()
    )


# Parametrize the smoke tests over every present example.
EXAMPLE_NAMES = _example_dirs()


@pytest.mark.skipif(
    not EXAMPLE_NAMES,
    reason="No examples/ subdirs with supercell_info.json present",
)
@pytest.mark.parametrize("example_name", EXAMPLE_NAMES)
def test_example_geometry_only_pipeline(example_name, tmp_path):
    """Run `pydefect_complex -g -n 2` on the example supercell_info.json.

    Asserts:
    - The CLI exits successfully.
    - ``defect/geometries_N2.yaml`` is written.
    - The geometry file is parseable and has the expected structure.
    """
    src = EXAMPLES_DIR / example_name / "supercell_info.json"
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(src, work / "supercell_info.json")

    proc = subprocess.run(
        [sys.executable, "-m", "pydefect_complex.cli", "-g", "-n", "2"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"pydefect_complex exited {proc.returncode} on {example_name}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    geom_path = work / "defect" / "geometries_N2.yaml"
    assert geom_path.is_file(), (
        f"Expected {geom_path} to be written; "
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    import yaml
    data = yaml.safe_load(geom_path.read_text())
    assert data["order"] == 2
    assert data["n_geometries"] >= 1
    assert "geometries" in data
    assert len(data["geometries"]) == data["n_geometries"]


@pytest.mark.skipif(
    "diamond" not in _example_dirs(),
    reason="examples/diamond/ not present",
)
def test_diamond_full_pipeline_with_structures(tmp_path):
    """Run a fuller pipeline (with --structures) on examples/diamond/.

    This exercises the full structure-generation + per-defect POSCAR
    writing path. Kept on the small diamond example to keep runtime
    bounded. Verifies:
    - CLI exit 0
    - defect/complex_defect_in.yaml has entries
    - At least one POSCAR subdir is created
    """
    src = EXAMPLES_DIR / "diamond" / "supercell_info.json"
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(src, work / "supercell_info.json")

    proc = subprocess.run(
        [
            sys.executable, "-m", "pydefect_complex.cli",
            "-d", "B", "-n", "2", "--max-distance", "3.0",
            "--structures",
        ],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"pydefect_complex exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    yaml_path = work / "defect" / "complex_defect_in.yaml"
    assert yaml_path.is_file()

    import yaml
    data = yaml.safe_load(yaml_path.read_text())
    assert data and len(data) > 0, "complex_defect_in.yaml should have entries"

    # At least one POSCAR subdirectory should be created.
    subdirs = [
        d for d in (work / "defect").iterdir()
        if d.is_dir() and (d / "POSCAR").is_file()
    ]
    assert len(subdirs) > 0, "Expected at least one POSCAR subdir with --structures"
    # Each subdir should also have a prior_info.yaml.
    for sd in subdirs:
        assert (sd / "prior_info.yaml").is_file(), (
            f"Missing prior_info.yaml in {sd}"
        )
