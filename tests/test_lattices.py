"""Smoke tests against non-diamond lattices.

Diamond (T_d) is the most symmetric and most tolerant test structure.
These tests catch bugs in:
- multi-element composition assignment (MgO has Mg and O sites;
  BaTiO3 has Ba, Ti, O on distinct Wyckoffs)
- non-cubic symmetry handling (h-BN is hexagonal)
- dopant substitution across different host species

The fixtures (hbn_supercell_info, mgo_supercell_info, batio3_supercell_info)
are session-scoped and skip if pydefect cannot build the supercell.
"""

import pytest

pytest.importorskip("pydefect")


# ---------------------------------------------------------------------------
# Generic smoke tests, parameterized over each lattice fixture.
# ---------------------------------------------------------------------------


def _run_full_pipeline(supercell_info, max_distance=3.5):
    """Helper: run make_all_pairs + generate_entries and return the entries."""
    from pydefect_complex import ComplexDefectMaker

    maker = ComplexDefectMaker(supercell_info, max_distance=max_distance)
    maker.make_all_pairs()
    return maker.generate_entries(n_or_geometries=2)


class TestGeometryEnumerationOnNonDiamond:
    """Every supported lattice should produce at least one 2-body geometry."""

    def test_hbn_produces_geometries(self, hbn_supercell_info):
        from pydefect_complex import ComplexDefectMaker

        maker = ComplexDefectMaker(hbn_supercell_info, max_distance=3.5)
        geoms = maker.enumerate_geometries(N_max=2)
        assert 2 in geoms and len(geoms[2]) > 0, (
            "h-BN should produce at least one 2-body geometry"
        )

    def test_mgo_produces_geometries(self, mgo_supercell_info):
        from pydefect_complex import ComplexDefectMaker

        maker = ComplexDefectMaker(mgo_supercell_info, max_distance=3.5)
        geoms = maker.enumerate_geometries(N_max=2)
        assert 2 in geoms and len(geoms[2]) > 0, (
            "MgO should produce at least one 2-body geometry"
        )

    def test_batio3_produces_geometries(self, batio3_supercell_info):
        from pydefect_complex import ComplexDefectMaker

        maker = ComplexDefectMaker(batio3_supercell_info, max_distance=3.5)
        geoms = maker.enumerate_geometries(N_max=2)
        assert 2 in geoms and len(geoms[2]) > 0, (
            "BaTiO3 should produce at least one 2-body geometry"
        )


class TestEntryGenerationOnNonDiamond:
    """Full pipeline (geometry + composition + structure) works on each lattice."""

    def test_hbn_full_pipeline(self, hbn_supercell_info):
        entries = _run_full_pipeline(hbn_supercell_info)
        if not entries:
            pytest.skip("No entries for h-BN with default cutoff")
        for e in entries:
            assert e.structure is not None
            assert e.complex_defect.n_defects == 2
            assert e.name  # .001-style suffix assigned by dedup

    def test_mgo_full_pipeline(self, mgo_supercell_info):
        entries = _run_full_pipeline(mgo_supercell_info)
        if not entries:
            pytest.skip("No entries for MgO with default cutoff")
        for e in entries:
            assert e.structure is not None
            assert e.complex_defect.n_defects == 2
            assert e.name

    def test_batio3_full_pipeline(self, batio3_supercell_info):
        entries = _run_full_pipeline(batio3_supercell_info)
        if not entries:
            pytest.skip("No entries for BaTiO3 with default cutoff")
        for e in entries:
            assert e.structure is not None
            assert e.complex_defect.n_defects == 2
            assert e.name


class TestDopantSubstitutionOnMultiElement:
    """Ca@Mg and Al@O in MgO; Sr@Ba and Zr@Ti in BaTiO3 should appear as defects."""

    def test_mgo_dopant_defects_present(self, mgo_supercell_info):
        from pydefect_complex import ComplexDefectMaker

        maker = ComplexDefectMaker(
            mgo_supercell_info, dopants=["Ca", "Al"], max_distance=3.5,
        )
        names = set(maker.defect_names)
        # Ca should substitute Mg; Al should substitute O. Pydefect's
        # DefectSetMaker uses the convention "ElementIn_ElementOut_Label",
        # so the names contain the host element symbol as a substring.
        assert any("Ca" in n and "Mg" in n for n in names), (
            f"Ca@Mg substitution missing from defect list: {names}"
        )
        assert any("Al" in n and "O" in n for n in names), (
            f"Al@O substitution missing from defect list: {names}"
        )

    def test_batio3_dopant_defects_present(self, batio3_supercell_info):
        from pydefect_complex import ComplexDefectMaker

        maker = ComplexDefectMaker(
            batio3_supercell_info, dopants=["Sr", "Zr"], max_distance=3.5,
        )
        names = set(maker.defect_names)
        assert any("Sr" in n and "Ba" in n for n in names), (
            f"Sr@Ba substitution missing: {names}"
        )
        assert any("Zr" in n and "Ti" in n for n in names), (
            f"Zr@Ti substitution missing: {names}"
        )


class TestWriteOutputOnNonDiamond:
    """Output pipeline works on each lattice."""

    def test_mgo_write_produces_yaml(self, mgo_supercell_info, tmp_output_dir):
        import yaml
        from pydefect_complex import ComplexDefectMaker

        entries = _run_full_pipeline(mgo_supercell_info)
        if not entries:
            pytest.skip("No entries for MgO with default cutoff")

        maker = ComplexDefectMaker(mgo_supercell_info, max_distance=3.5)
        maker.write(entries, tmp_output_dir)

        yaml_path = f"{tmp_output_dir}/complex_defect_in.yaml"
        import os
        assert os.path.exists(yaml_path)
        data = yaml.safe_load(open(yaml_path)) or {}
        assert len(data) > 0
        # Every entry should map to a non-empty charge list.
        for name, charges in data.items():
            assert isinstance(charges, list)
            assert len(charges) > 0


class TestAtomCountConsistency:
    """For a 2-defect entry: n_atoms = bulk - sum(out_elements) + sum(in_elements).

    Vacancies remove 1 atom; substitutions swap 1 atom; interstitials add 1.
    This property must hold for every generated entry regardless of lattice.
    """

    @pytest.mark.parametrize("supercell_name", [
        "diamond_supercell_info",
        "hbn_supercell_info",
        "mgo_supercell_info",
        "batio3_supercell_info",
    ])
    def test_n2_entries_atom_count(self, request, supercell_name):
        sc = request.getfixturevalue(supercell_name)
        bulk = len(sc.structure)

        entries = _run_full_pipeline(sc, max_distance=3.5)
        if not entries:
            pytest.skip(f"No entries for {supercell_name}")

        for e in entries:
            cd = e.complex_defect
            # Net atom change:
            #   vacancy     : in_atom is None                → -1 atom
            #   substitution: in_atom is not None, out is site label → net 0
            #   interstitial: in_atom is not None, out is "i*" → +1 atom
            n_removed = sum(
                1 for d in cd.defects
                if d.in_atom is None or not d.out_atom.startswith("i")
            )
            n_added = sum(1 for d in cd.defects if d.in_atom is not None)
            expected = bulk - n_removed + n_added
            assert len(e.structure) == expected, (
                f"{e.name}: expected {expected} atoms "
                f"(bulk={bulk}, removed={n_removed}, added={n_added}), "
                f"got {len(e.structure)}"
            )
