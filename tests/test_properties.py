"""Hypothesis-based property tests for the core data structures.

These tests verify invariants that should hold for *any* well-formed
input, not just the specific cases the other test files cover. They run
cheap (no real diamond supercell, no spglib) so they can run on every
commit.

Properties covered:
- ComplexDefect sort is deterministic: order of input does not matter.
- ComplexDefect hash is consistent with equality.
- equivalent() is reflexive, symmetric.
- deduplicate() is idempotent and deterministic.
- deduplicate() never grows the input.
- fingerprint is stable under node reordering.
"""

from collections import Counter

import numpy as np
import pytest

pytest.importorskip("pydefect")
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Helpers: build minimal but valid data structures
# ---------------------------------------------------------------------------


def _make_host_graph(n_nodes: int = 8) -> "HostGraph":
    """A small cubic host graph with `n_nodes` H sites (0.0-0.7 fractional)."""
    from pydefect_complex.graph import HostGraph, HostNode

    lattice = np.eye(3) * 5.0  # 5 Å cubic
    nodes = []
    for i in range(n_nodes):
        # Spread sites on a 2x2x2 sub-grid of the unit cell.
        x = (i // 1) % 2 * 0.5
        y = (i // 2) % 2 * 0.5
        z = (i // 4) % 2 * 0.5
        nodes.append(HostNode(
            id=i, wyckoff=f"H{i+1}", element="H",
            frac_coord=np.array([x, y, z]),
        ))
    return HostGraph(nodes=nodes, lattice=lattice)


def _make_graph(node_ids, hg: "HostGraph") -> "ComplexDefectGraph":
    """Build a graph from a tuple of host node ids, with min-image edges."""
    from pydefect_complex.graph import ComplexDefectGraph, _edge_list

    coords = [tuple(hg.nodes[i].frac_coord) for i in node_ids]
    edges = _edge_list(coords, hg, max_distance=10.0)
    return ComplexDefectGraph(
        host_node_ids=tuple(node_ids),
        wyckoffs=tuple(hg.nodes[i].wyckoff for i in node_ids),
        elements=tuple(hg.nodes[i].element for i in node_ids),
        edges=edges,
    )


_NODE_ID_PAIRS = st.tuples(
    st.sampled_from([0, 1, 2, 3]),
    st.sampled_from([0, 1, 2, 3]),
).filter(lambda p: p[0] != p[1])


# ---------------------------------------------------------------------------
# ComplexDefect: sort is order-invariant, hash agrees with eq
# ---------------------------------------------------------------------------


class TestComplexDefectProperties:
    @given(
        d1_wyckoff=st.sampled_from(["C1", "C2", "N1", "Si1"]),
        d2_wyckoff=st.sampled_from(["C1", "C2", "N1", "Si1"]),
        d1_in=st.sampled_from([None, "N", "B", "P"]),
        d2_in=st.sampled_from([None, "N", "B", "P"]),
    )
    @settings(max_examples=50, deadline=None)
    def test_from_pair_order_invariant(
        self, d1_wyckoff, d2_wyckoff, d1_in, d2_in,
    ):
        from pydefect.input_maker.defect import SimpleDefect
        from pydefect_complex.core import ComplexDefect

        d1 = SimpleDefect(d1_in, d1_wyckoff, [0])
        d2 = SimpleDefect(d2_in, d2_wyckoff, [0])
        cd_a = ComplexDefect.from_pair(d1, d2)
        cd_b = ComplexDefect.from_pair(d2, d1)
        assert cd_a.name == cd_b.name
        assert cd_a == cd_b
        assert hash(cd_a) == hash(cd_b)

    @given(
        wyckoffs=st.lists(
            st.sampled_from(["C1", "C2", "N1", "Si1", "O1"]),
            min_size=1, max_size=5,
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_name_independent_of_input_order(self, wyckoffs):
        """Any permutation of the same multiset of wyckoffs should yield
        the same ComplexDefect name (since dedup is content-based)."""
        from pydefect.input_maker.defect import SimpleDefect
        from pydefect_complex.core import ComplexDefect

        defects = [SimpleDefect(None, w, [0]) for w in wyckoffs]
        cd1 = ComplexDefect.from_defects(defects)
        cd2 = ComplexDefect.from_defects(list(reversed(defects)))
        assert cd1.name == cd2.name


# ---------------------------------------------------------------------------
# equivalent(): reflexive, symmetric on the same graph
# ---------------------------------------------------------------------------


class TestEquivalentProperties:
    @given(ids=_NODE_ID_PAIRS)
    @settings(max_examples=30, deadline=None)
    def test_reflexive(self, ids):
        from pydefect_complex.graph import equivalent

        hg = _make_host_graph(n_nodes=4)
        g = _make_graph(ids, hg)
        assert equivalent(g, g)

    @given(
        ids_a=_NODE_ID_PAIRS,
        ids_b=_NODE_ID_PAIRS,
    )
    @settings(max_examples=30, deadline=None)
    def test_symmetric(self, ids_a, ids_b):
        from pydefect_complex.graph import equivalent

        hg = _make_host_graph(n_nodes=4)
        ga = _make_graph(ids_a, hg)
        gb = _make_graph(ids_b, hg)
        assert equivalent(ga, gb) == equivalent(gb, ga)


# ---------------------------------------------------------------------------
# deduplicate(): idempotent, deterministic, never grows
# ---------------------------------------------------------------------------


def _make_entry(name, cd, hg: "HostGraph", dist: float = 1.0):
    """Build a minimal ComplexDefectEntry for dedup testing."""
    from pydefect_complex.structure import ComplexDefectEntry
    from pymatgen.core import Structure, Lattice
    import numpy as np

    s = Structure(Lattice.cubic(5.0), ["H", "H"], [[0, 0, 0], [0.5, 0, 0]])
    coords = (np.array([0., 0., 0.]), np.array([0.5, 0., 0.]))
    g = _make_graph((0, 1), hg)
    return ComplexDefectEntry(
        name=name,
        complex_defect=cd,
        site_path=("H1", "H2"),
        distances=(dist,),
        structure=s,
        defect_coords=coords,
        graph=g,
    )


class TestDeduplicateProperties:
    @given(
        n_entries=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**16 - 1),
    )
    @settings(max_examples=20, deadline=None)
    def test_idempotent(self, n_entries, seed):
        """deduplicate(deduplicate(x)) is a fixed point."""
        from pydefect.input_maker.defect import SimpleDefect
        from pydefect_complex.core import ComplexDefect
        from pydefect_complex.symmetry import deduplicate

        hg = _make_host_graph(n_nodes=4)
        d1 = SimpleDefect(None, "H1", [0])
        d2 = SimpleDefect(None, "H2", [0])
        cd = ComplexDefect.from_pair(d1, d2)

        rng = np.random.default_rng(seed)
        entries = [
            _make_entry(f"e{i}", cd, hg, dist=float(rng.uniform(0.5, 2.0)))
            for i in range(n_entries)
        ]

        first = deduplicate(entries, hg, max_distance=10.0)
        second = deduplicate(first, hg, max_distance=10.0)
        # Compare as sets — output order is deterministic but the set of
        # (name, complex_defect.name) pairs must be identical.
        first_keys = {(e.name, e.complex_defect.name) for e in first}
        second_keys = {(e.name, e.complex_defect.name) for e in second}
        assert first_keys == second_keys

    @given(
        n_entries=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**16 - 1),
    )
    @settings(max_examples=20, deadline=None)
    def test_never_grows(self, n_entries, seed):
        from pydefect.input_maker.defect import SimpleDefect
        from pydefect_complex.core import ComplexDefect
        from pydefect_complex.symmetry import deduplicate

        hg = _make_host_graph(n_nodes=4)
        d1 = SimpleDefect(None, "H1", [0])
        d2 = SimpleDefect(None, "H2", [0])
        cd = ComplexDefect.from_pair(d1, d2)

        rng = np.random.default_rng(seed)
        entries = [
            _make_entry(f"e{i}", cd, hg, dist=float(rng.uniform(0.5, 2.0)))
            for i in range(n_entries)
        ]
        out = deduplicate(entries, hg, max_distance=10.0)
        assert len(out) <= len(entries)

    @given(seed=st.integers(min_value=0, max_value=2**16 - 1))
    @settings(max_examples=10, deadline=None)
    def test_deterministic(self, seed):
        """Same input twice → same output (set equality)."""
        from pydefect.input_maker.defect import SimpleDefect
        from pydefect_complex.core import ComplexDefect
        from pydefect_complex.symmetry import deduplicate

        hg = _make_host_graph(n_nodes=4)
        d1 = SimpleDefect(None, "H1", [0])
        d2 = SimpleDefect(None, "H2", [0])
        cd = ComplexDefect.from_pair(d1, d2)

        rng = np.random.default_rng(seed)
        entries = [
            _make_entry(f"e{i}", cd, hg, dist=float(rng.uniform(0.5, 2.0)))
            for i in range(3)
        ]
        a = deduplicate(entries, hg, max_distance=10.0)
        b = deduplicate(entries, hg, max_distance=10.0)
        assert [e.name for e in a] == [e.name for e in b]


# ---------------------------------------------------------------------------
# Fingerprint stability under node permutation
# ---------------------------------------------------------------------------


class TestFingerprintProperties:
    @given(ids=_NODE_ID_PAIRS)
    @settings(max_examples=30, deadline=None)
    def test_fingerprint_independent_of_node_order(self, ids):
        from pydefect_complex.graph import _edge_list, ComplexDefectGraph

        hg = _make_host_graph(n_nodes=4)
        # Same edge set, different node ordering.
        a = _make_graph(ids, hg)
        rev = tuple(reversed(ids))
        coords_rev = [tuple(hg.nodes[i].frac_coord) for i in rev]
        edges_rev = _edge_list(coords_rev, hg, max_distance=10.0)
        b = ComplexDefectGraph(
            host_node_ids=rev,
            wyckoffs=tuple(hg.nodes[i].wyckoff for i in rev),
            elements=tuple(hg.nodes[i].element for i in rev),
            edges=edges_rev,
        )
        assert a.fingerprint == b.fingerprint
