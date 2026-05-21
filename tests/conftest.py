"""Test fixtures for pydefect-complex.

Creates a diamond supercell using pydefect's Python API,
then exposes SupercellInfo and SimpleDefect objects for testing.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Diamond POSCAR (conventional cubic cell)
# ---------------------------------------------------------------------------

DIAMOND_POSCAR = """diamond
1.0
    3.56679000000000    0.00000000000000    0.00000000000000
    0.00000000000000    3.56679000000000    0.00000000000000
    0.00000000000000    0.00000000000000    3.56679000000000
C
8
Direct
    0.00000000000000    0.00000000000000    0.00000000000000
    0.00000000000000    0.50000000000000    0.50000000000000
    0.50000000000000    0.00000000000000    0.50000000000000
    0.50000000000000    0.50000000000000    0.00000000000000
    0.25000000000000    0.25000000000000    0.25000000000000
    0.25000000000000    0.75000000000000    0.75000000000000
    0.75000000000000    0.25000000000000    0.75000000000000
    0.75000000000000    0.75000000000000    0.25000000000000
"""


@pytest.fixture(scope="session")
def diamond_supercell_info():
    """Create a diamond supercell_info using pydefect's Python API.

    Returns a pydefect SupercellInfo object or None if unavailable.
    """
    try:
        from pydefect.input_maker.supercell_maker import SupercellMaker
        from pymatgen.core import Structure
    except ImportError:
        pytest.skip("pydefect not installed")

    tmpdir = tempfile.mkdtemp(prefix="pydefect_complex_test_")

    try:
        poscar_path = Path(tmpdir) / "POSCAR"
        poscar_path.write_text(DIAMOND_POSCAR)

        structure = Structure.from_file(str(poscar_path))
        maker = SupercellMaker(
            structure, max_num_atoms=300, min_num_atoms=100
        )
        return maker.supercell_info

    except Exception as e:
        pytest.skip(f"Could not create diamond supercell: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def diamond_single_defects(diamond_supercell_info):
    """Return a list of SimpleDefect objects for diamond."""
    from pydefect.input_maker.defect_set_maker import DefectSetMaker

    maker = DefectSetMaker(
        diamond_supercell_info,
        dopants=["N", "B"],
    )
    return list(maker.defect_set)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for file writing tests."""
    output = tmp_path / "defect"
    output.mkdir()
    return str(output)