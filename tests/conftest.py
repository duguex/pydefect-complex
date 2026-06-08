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
# POSCAR strings for each lattice type
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

# Hexagonal boron nitride (space group P6_3/mmc, #194)
HBN_POSCAR = """h-BN
1.0
    2.504000    0.000000    0.000000
   -1.252000    2.168000    0.000000
    0.000000    0.000000    6.661000
B   N
2   2
Direct
    0.333333    0.666667    0.250000
    0.666667    0.333333    0.750000
    0.333333    0.666667    0.750000
    0.666667    0.333333    0.250000
"""

# Rock-salt MgO (space group Fm-3m, #225, conventional cubic cell)
MGO_POSCAR = """MgO
1.0
    4.211000    0.000000    0.000000
    0.000000    4.211000    0.000000
    0.000000    0.000000    4.211000
Mg  O
4   4
Direct
    0.000000    0.000000    0.000000
    0.500000    0.500000    0.000000
    0.500000    0.000000    0.500000
    0.000000    0.500000    0.500000
    0.500000    0.500000    0.500000
    0.000000    0.000000    0.500000
    0.000000    0.500000    0.000000
    0.500000    0.000000    0.000000
"""

# Cubic perovskite BaTiO3 (space group Pm-3m, #221)
BATIO3_POSCAR = """BaTiO3
1.0
    4.000000    0.000000    0.000000
    0.000000    4.000000    0.000000
    0.000000    0.000000    4.000000
Ba  Ti  O
1   1   3
Direct
    0.000000    0.000000    0.000000
    0.500000    0.500000    0.500000
    0.500000    0.500000    0.000000
    0.500000    0.000000    0.500000
    0.000000    0.500000    0.500000
"""


# ---------------------------------------------------------------------------
# SupercellInfo factory
# ---------------------------------------------------------------------------


def _make_supercell_info(poscar: str, name: str):
    """Build a pydefect SupercellInfo for the given POSCAR string.

    Skips the test if pydefect cannot process the structure (e.g. the
    supercell maker rejects the lattice).  Returns the SupercellInfo
    object on success.
    """
    try:
        from pydefect.input_maker.supercell_maker import SupercellMaker
        from pymatgen.core import Structure
    except ImportError:
        pytest.skip("pydefect not installed")

    tmpdir = tempfile.mkdtemp(prefix=f"pydefect_complex_test_{name}_")
    try:
        poscar_path = Path(tmpdir) / "POSCAR"
        poscar_path.write_text(poscar)
        structure = Structure.from_file(str(poscar_path))
        maker = SupercellMaker(
            structure, max_num_atoms=300, min_num_atoms=80,
        )
        return maker.supercell_info
    except Exception as e:
        pytest.skip(f"Could not create {name} supercell: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Per-lattice fixtures (session-scoped, expensive to build)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def diamond_supercell_info():
    """Diamond (cubic, single element). C-diamond."""
    return _make_supercell_info(DIAMOND_POSCAR, "diamond")


@pytest.fixture(scope="session")
def hbn_supercell_info():
    """Hexagonal boron nitride (P6_3/mmc). Two elements, hexagonal."""
    return _make_supercell_info(HBN_POSCAR, "hbn")


@pytest.fixture(scope="session")
def mgo_supercell_info():
    """Rock-salt MgO (Fm-3m). Two elements, cubic, single distinct Wyckoff each."""
    return _make_supercell_info(MGO_POSCAR, "mgo")


@pytest.fixture(scope="session")
def batio3_supercell_info():
    """Cubic perovskite BaTiO3 (Pm-3m). Three elements, multiple distinct Wyckoffs."""
    return _make_supercell_info(BATIO3_POSCAR, "batio3")


# ---------------------------------------------------------------------------
# Per-lattice defect fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def diamond_single_defects(diamond_supercell_info):
    """Diamond + N, B dopants."""
    from pydefect.input_maker.defect_set_maker import DefectSetMaker

    return list(DefectSetMaker(
        diamond_supercell_info, dopants=["N", "B"],
    ).defect_set)


@pytest.fixture
def hbn_single_defects(hbn_supercell_info):
    """h-BN + C dopant (C substitutes N or B; interstitials possible)."""
    from pydefect.input_maker.defect_set_maker import DefectSetMaker

    return list(DefectSetMaker(
        hbn_supercell_info, dopants=["C"],
    ).defect_set)


@pytest.fixture
def mgo_single_defects(mgo_supercell_info):
    """MgO + Ca (substitutes Mg) and Al (substitutes O) dopants."""
    from pydefect.input_maker.defect_set_maker import DefectSetMaker

    return list(DefectSetMaker(
        mgo_supercell_info, dopants=["Ca", "Al"],
    ).defect_set)


@pytest.fixture
def batio3_single_defects(batio3_supercell_info):
    """BaTiO3 + Sr (substitutes Ba), Zr (substitutes Ti) dopants."""
    from pydefect.input_maker.defect_set_maker import DefectSetMaker

    return list(DefectSetMaker(
        batio3_supercell_info, dopants=["Sr", "Zr"],
    ).defect_set)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for file writing tests."""
    output = tmp_path / "defect"
    output.mkdir()
    return str(output)