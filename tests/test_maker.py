"""Integration tests for ComplexDefectMaker end-to-end workflow.

Requires pydefect to be installed.
Uses pydefect naming: vacancies are "Va_C1", substitutions are "N_C1".
"""

import os
import time
import pytest

pytest.importorskip("pydefect")

from pydefect_complex import ComplexDefectMaker, ComplexDefectEnumerator
from pydefect_complex.graph import HostGraph
from pydefect_complex.enumerate import assign_compositions


class TestMakerFromSupercellInfo:

    def test_from_supercell_info_creation(self, diamond_supercell_info):
        maker = ComplexDefectMaker(diamond_supercell_info)
        assert maker.supercell_info is diamond_supercell_info
        assert len(maker.single_defects) > 0
        assert len(maker.defect_names) > 0

    def test_defect_in_dict(self, diamond_supercell_info):
        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N"]
        )
        defect_in = maker.defect_in
        assert isinstance(defect_in, dict)
        for name, charges in defect_in.items():
            assert isinstance(name, str)
            assert isinstance(charges, (list, tuple))
            assert all(isinstance(c, int) for c in charges)

    def test_defect_pairs(self, diamond_supercell_info):
        maker = ComplexDefectMaker(diamond_supercell_info)
        pairs = maker.defect_pairs
        n = len(maker.single_defects)
        expected = n * (n + 1) // 2
        assert len(pairs) == expected

    def test_summary(self, diamond_supercell_info):
        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"]
        )
        s = maker.summary()
        assert s["n_single_defects"] > 0
        assert s["dopants"] == ["N", "B"]
        assert s["max_distance"] == 3.0

    def test_make_pair(self, diamond_supercell_info):
        maker = ComplexDefectMaker(diamond_supercell_info)
        names = maker.defect_names
        if len(names) < 2:
            pytest.skip("Need at least 2 defect types")

        entries = maker.make_pair(names[0], names[1], max_distance=5.0)
        for e in entries:
            assert names[0] in e.name
            assert names[1] in e.name
            assert e.distance > 0
            assert e.structure is not None

    def test_make_all_pairs(self, diamond_supercell_info):
        maker = ComplexDefectMaker(
            diamond_supercell_info, max_distance=3.5
        )
        geoms = maker.make_all_pairs()
        assert isinstance(geoms, list)
        assert all(g.n_defects == 2 for g in geoms)

    def test_write_output(self, diamond_supercell_info, tmp_output_dir):
        maker = ComplexDefectMaker(
            diamond_supercell_info, max_distance=3.5
        )
        maker.make_all_pairs()  # enumerate geometries
        entries = maker.generate_entries(n_or_geometries=2)
        if not entries:
            pytest.skip("No entries generated with this cutoff")

        yaml_path = maker.write(entries, tmp_output_dir)
        assert os.path.exists(yaml_path)

        first_entry = entries[0]
        first_dir = (
            f"{tmp_output_dir}/{first_entry.name}"
            f"_{first_entry.complex_defect.charges[0]}"
        )
        assert os.path.isdir(first_dir)
        assert os.path.isfile(f"{first_dir}/POSCAR")
        assert os.path.isfile(f"{first_dir}/prior_info.yaml")


class TestMakerRepr:
    def test_repr(self, diamond_supercell_info):
        maker = ComplexDefectMaker(diamond_supercell_info)
        r = repr(maker)
        assert "ComplexDefectMaker" in r
        assert "n_defects" in r


# ---------------------------------------------------------------------------
# PLAN-C: ComplexDefectEnumerator + N-body tests
# ---------------------------------------------------------------------------


class TestEnumerator:

    def test_enumerate_2(self, diamond_supercell_info):
        maker = ComplexDefectMaker(diamond_supercell_info)
        geo = maker.enumerate_geometries(N_max=2)
        assert 2 in geo
        assert len(geo[2]) >= 1, "Should find at least one N=2 geometry"
        for g in geo[2]:
            assert g.n_defects == 2
            assert len(g.edges) == 1

    def test_enumerate_3(self, diamond_supercell_info):
        maker = ComplexDefectMaker(diamond_supercell_info)
        geo = maker.enumerate_geometries(N_max=3)
        assert 3 in geo
        assert len(geo[3]) >= 1, "Should find at least one N=3 geometry"
        for g in geo[3]:
            assert g.n_defects == 3
            assert len(g.edges) >= 2

    def test_cache_reuse(self, diamond_supercell_info):
        maker = ComplexDefectMaker(diamond_supercell_info)
        geo2 = maker.enumerate_geometries(N_max=2)
        geo3 = maker.enumerate_geometries(N_max=3)
        # N=2 results should be the same cached objects
        assert geo2[2] is geo3[2]

    def test_assign_compositions(self, diamond_supercell_info):
        maker = ComplexDefectMaker(diamond_supercell_info)
        geo = maker.enumerate_geometries(N_max=2)
        pairs = assign_compositions(geo[2], maker.single_defects)
        assert len(pairs) > 0
        for G, cd in pairs:
            assert G.n_defects == cd.n_defects
            assert sorted(G.wyckoffs) == sorted(d.out_atom for d in cd.defects)

    def test_make_all_n_body_3(self, diamond_supercell_info):
        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=4.0,
        )
        maker.make_all_n_body(n=3)  # enumerate
        entries = maker.generate_entries(n_or_geometries=3)
        assert isinstance(entries, list)
        for e in entries:
            assert e.complex_defect.n_defects == 3
            assert e.structure is not None

    def test_charges_parameter(self, diamond_supercell_info):
        """Charges override applies to all generated entries."""
        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=4.0,
            charges=[-2, -1, 0, 1, 2],
        )
        maker.make_all_pairs()
        entries = maker.generate_entries(n_or_geometries=2)
        assert len(entries) > 0
        for e in entries:
            assert e.complex_defect.charges == [-2, -1, 0, 1, 2]

    def test_performance_n2(self, diamond_supercell_info):
        """N=2 enumeration completes in under 2 seconds."""
        maker = ComplexDefectMaker(diamond_supercell_info)
        t0 = time.time()
        maker.make_all_pairs()
        t1 = time.time()
        assert t1 - t0 < 2.0, f"N=2 took {t1 - t0:.2f}s"

    def test_performance_n3(self, diamond_supercell_info):
        """N=3 enumeration completes in under 20 seconds."""
        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=4.0,
        )
        t0 = time.time()
        maker.make_all_n_body(n=3)
        t1 = time.time()
        assert t1 - t0 < 20.0, f"N=3 took {t1 - t0:.2f}s"