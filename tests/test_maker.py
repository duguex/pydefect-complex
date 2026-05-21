"""Integration tests for ComplexDefectMaker end-to-end workflow.

Requires pydefect to be installed.
Uses pydefect naming: vacancies are "Va_C1", substitutions are "N_C1".
"""

import os
import pytest

pytest.importorskip("pydefect")

from pydefect_complex import ComplexDefectMaker


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
        assert s["max_distance"] == 5.0

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
        entries = maker.make_all_pairs()
        assert isinstance(entries, list)

    def test_write_output(self, diamond_supercell_info, tmp_output_dir):
        maker = ComplexDefectMaker(
            diamond_supercell_info, max_distance=3.5
        )
        entries = maker.make_all_pairs()
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