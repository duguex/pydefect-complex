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
# Unit tests for specific public methods (not covered transitively)
# ---------------------------------------------------------------------------


class TestSetDopants:
    def test_set_dopants_preserves_geometry_cache(
        self, diamond_supercell_info,
    ):
        """Switching dopants must NOT invalidate the geometry enumeration cache.

        This is the whole point of set_dopants() vs constructing a new maker.
        """
        from pydefect.input_maker.defect import SimpleDefect

        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N"], max_distance=3.5,
        )
        # Populate the geometry cache for N=2.
        maker.enumerate_geometries(N_max=2)
        assert 2 in maker.enumerator.geometries
        geoms_before = list(maker.enumerator.geometries[2])

        # Switch dopants.
        maker.set_dopants(dopants=["N", "B"])

        # Geometry cache must survive.
        assert 2 in maker.enumerator.geometries
        assert len(maker.enumerator.geometries[2]) == len(geoms_before)
        # And the same objects — no re-enumeration happened.
        assert maker.enumerator.geometries[2][0] is geoms_before[0]

    def test_set_dopants_updates_defect_list(
        self, diamond_supercell_info,
    ):
        """After set_dopants, single_defects reflects the new dopant set."""
        from pydefect.input_maker.defect import SimpleDefect

        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N"], max_distance=3.5,
        )
        before_names = set(maker.defect_names)

        maker.set_dopants(dopants=["N", "B"])
        after_names = set(maker.defect_names)

        # New dopant set must add at least one new defect type
        # (the B substitution), and the previous N dopant must remain.
        assert after_names > before_names
        assert "N" in str(before_names) or any("N" in n for n in before_names)
        # Entry cache must be cleared so the new chemistry is re-generated.
        assert maker.entry_cache == {}

    def test_set_dopants_to_intrinsic(self, diamond_supercell_info):
        """set_dopants([]) should produce only intrinsic (vacancy) defects."""
        from pydefect.input_maker.defect import SimpleDefect

        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=3.5,
        )
        assert any("N" in d.name or "B" in d.name for d in maker.single_defects)

        maker.set_dopants(dopants=[])
        # No dopant substitutions in the defect list anymore.
        assert not any(
            (d.in_atom is not None and not d.out_atom.startswith("i"))
            for d in maker.single_defects
        )


class TestWriteParameters:
    def test_write_parameters_includes_run_metadata(
        self, diamond_supercell_info, tmp_path,
    ):
        from pydefect_complex.maker import ComplexDefectMaker
        import yaml

        out = tmp_path / "params"
        out.mkdir()
        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=4.0,
            charges=[-1, 0, 1],
        )
        maker.make_all_pairs()  # populate entry_cache
        maker._write_parameters(str(out))

        path = out / "parameters.yaml"
        assert path.is_file()
        data = yaml.safe_load(path.read_text())
        assert data["pydefect_complex_version"]  # not empty
        assert "timestamp" in data
        assert data["parameters"]["max_distance_angstrom"] == 4.0
        assert data["parameters"]["dopants"] == ["N", "B"]
        assert data["parameters"]["charges"] == [-1, 0, 1]
        # Caches should be reflected.
        assert "2" in data["enumerator"]["n_geometries_cached"]
        assert data["entry_cache"]["orders_cached"] == [2]


class TestGeometryCacheRoundTrip:
    def test_save_then_load_round_trips_geometries(
        self, diamond_supercell_info, tmp_path,
    ):
        """Save → wipe enumerator cache → load must restore the same geometries."""
        from pydefect_complex.maker import ComplexDefectMaker

        out = tmp_path / "cache"
        out.mkdir()

        # First maker: enumerate and save.
        m1 = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=4.0,
        )
        m1.enumerate_geometries(N_max=3)
        geoms_before = {
            n: [g.to_dict() for g in gs]
            for n, gs in m1.enumerator.geometries.items()
        }
        m1.save_geometry_cache(str(out))

        # Second maker: identical parameters, load from cache.
        m2 = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=4.0,
        )
        loaded_orders = m2.load_geometry_cache(str(out))
        assert loaded_orders == {2, 3}

        # Compare the dict-form of every geometry — identity may differ
        # but content must match.
        for n, before_list in geoms_before.items():
            after_list = [g.to_dict() for g in m2.enumerator.geometries[n]]
            assert before_list == after_list

    def test_load_geometry_cache_skips_mismatched_max_distance(
        self, diamond_supercell_info, tmp_path,
    ):
        """Cache file with different max_distance must NOT be loaded."""
        from pydefect_complex.maker import ComplexDefectMaker

        out = tmp_path / "cache"
        out.mkdir()

        # Save with max_distance=4.0.
        m1 = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N"], max_distance=4.0,
        )
        m1.enumerate_geometries(N_max=2)
        m1.save_geometry_cache(str(out))

        # Try to load with max_distance=3.0 (mismatch).
        m2 = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N"], max_distance=3.0,
        )
        loaded = m2.load_geometry_cache(str(out))
        assert loaded == set(), "Mismatched cache should be skipped entirely"

    def test_load_geometry_cache_returns_empty_for_missing_dir(
        self, diamond_supercell_info, tmp_path,
    ):
        from pydefect_complex.maker import ComplexDefectMaker

        m = ComplexDefectMaker(diamond_supercell_info, max_distance=3.0)
        # Point at a directory that doesn't exist.
        assert m.load_geometry_cache(str(tmp_path / "nope")) == set()


class TestGenerateEntriesChargesOverride:
    def test_charges_at_construction_propagates_to_entries(
        self, diamond_supercell_info,
    ):
        """Charges passed to ComplexDefectMaker(charges=...) must flow through.

        This is the supported way to set charge states — it ensures the
        entry cache is populated with the desired charges on the first
        enumeration call.
        """
        from pydefect_complex.maker import ComplexDefectMaker

        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=4.0,
            charges=[-2, -1, 0, 1, 2],
        )
        maker.make_all_pairs()
        entries = maker.generate_entries(n_or_geometries=2)
        assert len(entries) > 0
        for e in entries:
            assert e.complex_defect.charges == [-2, -1, 0, 1, 2], (
                f"Expected constructor charges; got {e.complex_defect.charges}"
            )

    def test_charges_none_uses_maker_default(
        self, diamond_supercell_info,
    ):
        """When no charges constructor arg, default [0] is used."""
        from pydefect_complex.maker import ComplexDefectMaker

        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=4.0,
        )
        maker.make_all_pairs()
        entries = maker.generate_entries(n_or_geometries=2)
        assert len(entries) > 0
        for e in entries:
            assert e.complex_defect.charges == [0]

    def test_charges_override_stamps_on_cache_hit(
        self, diamond_supercell_info,
    ):
        """generate_entries(charges=...) overrides cached charges.

        Regression test: previously the entry-cache hit path returned
        cached entries with their original charges, silently dropping
        the override. See charges-override-cache-bug.md in memory.
        """
        from pydefect_complex.maker import ComplexDefectMaker

        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=4.0,
            charges=[0],  # default at construction
        )
        maker.make_all_pairs()  # populates entry_cache for N=2

        # Override charges AFTER the cache is populated.
        entries = maker.generate_entries(
            n_or_geometries=2, charges=[-2, -1, 0, 1, 2],
        )
        assert len(entries) > 0
        for e in entries:
            assert e.complex_defect.charges == [-2, -1, 0, 1, 2], (
                f"Override charges were dropped; got {e.complex_defect.charges}"
            )

    def test_charges_override_does_not_mutate_underlying_cache(
        self, diamond_supercell_info,
    ):
        """The override should not leak into the cache: a follow-up
        call with charges=None must return entries with the original
        constructor-level charges."""
        from pydefect_complex.maker import ComplexDefectMaker

        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=4.0,
            charges=[0],
        )
        maker.make_all_pairs()

        # First call: override.
        e1 = maker.generate_entries(
            n_or_geometries=2, charges=[-2, -1, 0, 1, 2],
        )
        for e in e1:
            assert e.complex_defect.charges == [-2, -1, 0, 1, 2]

        # Second call: no override — should return [0] again.
        e2 = maker.generate_entries(n_or_geometries=2)
        for e in e2:
            assert e.complex_defect.charges == [0], (
                f"Cache was mutated by previous override; got {e.complex_defect.charges}"
            )


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

    @pytest.mark.slow
    def test_performance_n4(self, diamond_supercell_info):
        """N=4 enumeration completes in under 300 seconds.

        N=4 quadruples on a 128-atom diamond cell is the upper end of
        what's tractable. The 300s budget accommodates CI environments
        where this test contends for CPU with other tests in the suite
        (e.g., the parallel-execution tests). The budget mainly catches
        catastrophic regressions (e.g., dedup becoming O(n^4) instead
        of O(n^2)); on a quiet 4-core machine this runs in ~80s.

        Skip with: ``pytest -m "not slow"``.
        """
        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N", "B"], max_distance=4.0,
        )
        t0 = time.time()
        geo = maker.enumerate_geometries(N_max=4)
        t1 = time.time()
        assert 4 in geo, "N=4 geometry enumeration returned no result"
        assert t1 - t0 < 300.0, f"N=4 took {t1 - t0:.2f}s"


class TestParallelExecution:
    """Smoke tests for the parallel-execution paths.

    The maker's ``n_workers`` flag gates two dispatch points:
      - ``ComplexDefectEnumerator._extend_order`` (geometry)
      - ``generate_all_entries`` (structure generation)
    These tests verify parallel runs don't corrupt the output.
    """

    def test_parallel_n_workers_2_matches_serial(self, diamond_supercell_info):
        """n_workers=2 should produce the same set of entries as n_workers=1."""
        serial = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N"], max_distance=4.0, n_workers=1,
        )
        serial.make_all_n_body(n=3)
        serial_entries = serial.generate_entries(n_or_geometries=3)

        parallel = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N"], max_distance=4.0, n_workers=2,
        )
        parallel.make_all_n_body(n=3)
        parallel_entries = parallel.generate_entries(n_or_geometries=3)

        # Compare by (name, structure formula) — structure objects may differ
        # in memory but should be identical by content.
        serial_keys = {(e.name, str(e.structure.composition.formula))
                       for e in serial_entries}
        parallel_keys = {(e.name, str(e.structure.composition.formula))
                         for e in parallel_entries}
        assert serial_keys == parallel_keys, (
            f"Serial/parallel disagree: "
            f"only_in_serial={serial_keys - parallel_keys}, "
            f"only_in_parallel={parallel_keys - serial_keys}"
        )