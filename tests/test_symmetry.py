"""Unit tests for symmetry.deduplicate — the cross-composition dedup core.

These tests use the real pydefect diamond supercell (via the session-scoped
``diamond_supercell_info`` fixture) and exercise ``deduplicate`` directly,
independent of ``ComplexDefectMaker.generate_entries``.
"""

import pytest

pytest.importorskip("pydefect")


@pytest.fixture
def diamond_pair_entries(diamond_supercell_info):
    """Return a list of (already-deduped) N=2 entries for diamond."""
    from pydefect_complex import ComplexDefectMaker

    maker = ComplexDefectMaker(
        diamond_supercell_info, max_distance=3.5,
    )
    maker.make_all_pairs()
    return maker.generate_entries(n_or_geometries=2)


class TestDeduplicateEmpty:
    def test_empty_input_returns_empty(self, diamond_supercell_info):
        from pydefect_complex.graph import HostGraph
        from pydefect_complex.symmetry import deduplicate

        hg = HostGraph.from_supercell_info(diamond_supercell_info)
        assert deduplicate([], hg, max_distance=3.0) == []


class TestDeduplicateCollapses:
    def test_duplicate_geometry_collapses(
        self, diamond_supercell_info, diamond_pair_entries,
    ):
        """Two entries with the same geometry (even with different comp names)
        should collapse to a single entry in the output."""
        from pydefect_complex.graph import HostGraph
        from pydefect_complex.symmetry import deduplicate

        if len(diamond_pair_entries) < 2:
            pytest.skip("Need >= 2 entries to test collapse")

        hg = HostGraph.from_supercell_info(diamond_supercell_info)
        # Take one entry and duplicate it (same graph, same complex_defect)
        original = diamond_pair_entries[0]
        duplicate = diamond_pair_entries[0]  # same object

        out = deduplicate([original, duplicate], hg, max_distance=3.5)
        assert len(out) == 1, "Identical entries should collapse to one"

    def test_indexing_starts_at_001(
        self, diamond_supercell_info, diamond_pair_entries,
    ):
        """Surviving entries get names ending in .{3-digit-index} starting at 001."""
        from pydefect_complex.graph import HostGraph
        from pydefect_complex.symmetry import deduplicate

        hg = HostGraph.from_supercell_info(diamond_supercell_info)
        out = deduplicate(diamond_pair_entries, hg, max_distance=3.5)
        if not out:
            pytest.skip("No entries to test")
        # Each surviving entry's name should end with .NNN (3-digit index).
        # The base name uses the compact-count format (e.g. "2Va_C1"),
        # so we check the suffix only.
        for e in out:
            suffix = e.name.rsplit(".", 1)[-1]
            assert len(suffix) == 3 and suffix.isdigit(), (
                f"Expected 3-digit index suffix, got {e.name!r}"
            )


class TestDeduplicateIndexing:
    def test_same_composition_inequivalent_geometries_get_distinct_indices(
        self, diamond_supercell_info,
    ):
        """Two inequivalent geometries of the same composition get .001 and .002."""
        from pydefect.input_maker.defect import SimpleDefect
        from pydefect_complex.core import ComplexDefect
        from pydefect_complex.graph import HostGraph, ComplexDefectGraph
        from pydefect_complex.symmetry import deduplicate
        from pymatgen.core import Structure, Lattice
        import numpy as np

        # Build a tiny cubic structure: 2 atoms in a 5 Å cube
        # (no symmetry between the two sites, so the two site-pairs
        # are inequivalent).
        lat = Lattice.cubic(5.0)
        s1 = Structure(lat, ["H", "H"], [[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]])
        hg = HostGraph(
            nodes=[
                __import__("pydefect_complex.graph", fromlist=["HostNode"])
                .HostNode(id=0, wyckoff="H1", element="H", frac_coord=np.array([0., 0., 0.])),
                __import__("pydefect_complex.graph", fromlist=["HostNode"])
                .HostNode(id=1, wyckoff="H2", element="H", frac_coord=np.array([0.4, 0.4, 0.4])),
            ],
            lattice=lat.matrix,
        )

        # Build two entries that share the same composition but have
        # *different* geometric graphs. We construct entries with the
        # minimum surface area that ``deduplicate`` reads:
        # complex_defect, defect_coords, distances, structure, graph.
        d1 = SimpleDefect(None, "H1", [0])
        d2 = SimpleDefect(None, "H2", [0])
        cd = ComplexDefect.from_pair(d1, d2)

        def make_entry(name, dist):
            coords = (np.array([0., 0., 0.]), np.array([0.4, 0.4, 0.4]))
            g = ComplexDefectGraph(
                host_node_ids=(0, 1),
                wyckoffs=("H1", "H2"),
                elements=("H", "H"),
                edges=[(0, 1, np.array([2.0 * dist, 0., 0.]))],
            )
            entry = cd  # placeholder, replaced below
            from pydefect_complex.structure import ComplexDefectEntry
            return ComplexDefectEntry(
                name=name,
                complex_defect=cd,
                site_path=("H1", "H2"),
                distances=(dist,),
                structure=s1,
                defect_coords=coords,
                graph=g,
            )

        e1 = make_entry("v_H1+v_H2_a", 1.0)
        e2 = make_entry("v_H1+v_H2_b", 2.0)  # different distance → inequivalent

        out = deduplicate([e1, e2], hg, max_distance=10.0)
        # If the two entries survive dedup, the surviving names should
        # end in .001 and .002 (deterministic per-composition indexing).
        # The base name is whatever ComplexDefect generates for
        # SimpleDefect(None, "H1"/"H2", ...): "Va_H1+Va_H2" (sorted by
        # out_atom reverse), so we just check the suffixes.
        assert len(out) == 2
        suffixes = sorted(e.name.rsplit(".", 1)[-1] for e in out)
        assert suffixes == ["001", "002"]


class TestVerifyDedup:
    def test_verify_dedup_clean_report(self, diamond_supercell_info, diamond_pair_entries):
        """verify_dedup returns ok=True on a properly deduped set."""
        from pydefect_complex.graph import HostGraph
        from pydefect_complex.symmetry import deduplicate, verify_dedup

        hg = HostGraph.from_supercell_info(diamond_supercell_info)
        out = deduplicate(diamond_pair_entries, hg, max_distance=3.5)
        report = verify_dedup(out, hg)
        assert report["ok"] is True
        assert report["n_entries"] == len(out)
        # All entries should map to a unique cluster
        assert report["false_positives"] == []
