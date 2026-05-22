"""Graph-based complex defect enumeration via Apriori-style incremental geometry search.

  1. Enumerate geometrically unique N-node site configurations.
  2. Assign defect compositions by wyckoff label matching.
  3. For each (geometry, composition) pair, generate the defect structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import IStructure
    from pydefect.input_maker.supercell_info import SupercellInfo

from .core import ComplexDefect, _get_element, _is_interstitial
from .graph import HostGraph, ComplexDefectGraph, _edge_list, equivalent


# ---------------------------------------------------------------------------
# ComplexDefectEnumerator: Apriori-style incremental enumeration
# ---------------------------------------------------------------------------


class ComplexDefectEnumerator:
    """Enumerate geometrically unique N-body defect site configurations.

    Uses Apriori-style incremental building with online geometric dedup:

    1. N=2: anchor by wyckoff class → neighbor pairs → dedup
    2. k → k+1: extend unique k-geometries by external neighbors → dedup

    Caches results — calling enumerate(N_max=4) after enumerate(N_max=3)
    reuses previously computed geometries for orders 2 and 3.

    Attributes:
        geometries: {2: [G, ...], 3: [G, ...], ...} after enumeration.
    """

    def __init__(
        self,
        host_graph: HostGraph,
        max_distance: float = 5.0,
        min_distance: float = 0.3,
    ):
        self.host_graph = host_graph
        self.max_distance = max_distance
        self.min_distance = min_distance
        self._cache: dict[int, list[ComplexDefectGraph]] = {}

    @property
    def geometries(self) -> dict[int, list[ComplexDefectGraph]]:
        """Currently cached geometries, or empty dict if not yet enumerated."""
        return dict(self._cache)

    def enumerate(
        self, N_max: int, eps: float = 0.1,
    ) -> dict[int, list[ComplexDefectGraph]]:
        """Enumerate all geometrically unique N-node subgraphs for orders 2..N_max.

        Returns:
            {2: [ComplexDefectGraph, ...], 3: [...], ..., N_max: [...]}

        Raises:
            ValueError: if N_max < 2.
        """
        if N_max < 2:
            raise ValueError(f"N_max must be >= 2, got {N_max}")

        # Reuse cache if already computed
        max_cached = max(self._cache.keys()) if self._cache else 0
        if max_cached >= N_max:
            return {k: v for k, v in self._cache.items() if k <= N_max}

        # Bootstrap N=2 if needed
        if 2 not in self._cache:
            self._cache[2] = self._enumerate_2(eps)

        # Apriori: k → k+1
        start_k = max(2, max_cached)
        for k in range(start_k, N_max):
            if k + 1 not in self._cache:
                self._cache[k + 1] = self._extend_order(k, eps)

        return {k: v for k, v in self._cache.items() if k <= N_max}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enumerate_2(self, eps: float) -> list[ComplexDefectGraph]:
        """Generate all unique 2-node geometries from anchor+neighbor pairs.

        Anchors are chosen as one representative per (wyckoff, element) class.
        """
        geometries: list[ComplexDefectGraph] = []
        seen_anchor_classes: set[tuple[str, str]] = set()

        for anchor in self.host_graph.nodes:
            anchor_class = (anchor.wyckoff, anchor.element)
            if anchor_class in seen_anchor_classes:
                continue
            seen_anchor_classes.add(anchor_class)

            for nbr_id in self.host_graph.neighbors(anchor.id, self.max_distance):
                nbr = self.host_graph.nodes[nbr_id]

                d = self.host_graph.min_image_distance(
                    anchor.frac_coord, nbr.frac_coord,
                )
                if d < self.min_distance:
                    continue

                G = ComplexDefectGraph(
                    host_node_ids=(anchor.id, nbr_id),
                    wyckoffs=(anchor.wyckoff, nbr.wyckoff),
                    elements=(anchor.element, nbr.element),
                    edges=_edge_list(
                        [tuple(anchor.frac_coord), tuple(nbr.frac_coord)],
                        self.host_graph,
                        self.max_distance,
                    ),
                )

                if not any(equivalent(G, g, eps) for g in geometries):
                    geometries.append(G)

        return geometries

    def _extend_order(self, k: int, eps: float) -> list[ComplexDefectGraph]:
        """Extend unique k-geometries to (k+1)-geometries.

        For each unique G_k, finds all external neighbors, builds
        G_{k+1} = G_k ∪ {neighbor}, and keeps only geometrically
        unique results.
        """
        result: list[ComplexDefectGraph] = []

        for G_k in self._cache[k]:
            ext_neighbors = self.host_graph.neighbors_of_set(
                set(G_k.host_node_ids), self.max_distance,
            )

            for nbr_id in ext_neighbors:
                nbr = self.host_graph.nodes[nbr_id]

                # min_distance to ALL existing nodes
                too_close = any(
                    self.host_graph.min_image_distance(
                        nbr.frac_coord,
                        self.host_graph.nodes[hid].frac_coord,
                    )
                    < self.min_distance
                    for hid in G_k.host_node_ids
                )
                if too_close:
                    continue

                # Build extended graph
                new_ids = tuple(list(G_k.host_node_ids) + [nbr_id])
                coords = [
                    tuple(self.host_graph.nodes[i].frac_coord)
                    for i in new_ids
                ]
                edges = _edge_list(coords, self.host_graph, self.max_distance)

                G_next = ComplexDefectGraph(
                    host_node_ids=new_ids,
                    wyckoffs=tuple(list(G_k.wyckoffs) + [nbr.wyckoff]),
                    elements=tuple(list(G_k.elements) + [nbr.element]),
                    edges=edges,
                )

                if not any(equivalent(G_next, g, eps) for g in result):
                    result.append(G_next)

        return result


# ---------------------------------------------------------------------------
# Composition assignment: match geometries to defect compositions
# ---------------------------------------------------------------------------


def assign_compositions(
    geometries: list[ComplexDefectGraph],
    single_defects: list,
) -> list[tuple[ComplexDefectGraph, "ComplexDefect"]]:
    """Match geometries to compatible defect compositions by wyckoff label.

    For each geometry G, generates all N-defect combinations whose
    out_atom multiset matches G's wyckoff multiset, then pairs them.

    Filter rules (same as old structure.py):
      - Only the first defect can be interstitial.
      - Interstitial insert element must differ from next defect's host element.

    Args:
        geometries: List of unique ComplexDefectGraph objects.
        single_defects: List of pydefect SimpleDefect objects.

    Returns:
        List of (ComplexDefectGraph, ComplexDefect) pairs.
    """
    import itertools
    from .core import ComplexDefect as CD, _is_interstitial

    results = []
    by_n = {}
    for G in geometries:
        by_n.setdefault(G.n_defects, []).append(G)

    for n, geoms in sorted(by_n.items()):
        # For each N, try all combinations of N single defects
        for combo in itertools.combinations_with_replacement(single_defects, n):
            combo = list(combo)

            # Filter: only first can be interstitial
            if any(_is_interstitial(d.out_atom) for d in combo[1:]):
                continue

            # Filter: interstitial + element cycling (first layer only)
            if _is_interstitial(combo[0].out_atom):
                if (_get_element(combo[1].out_atom)
                        == combo[0].in_atom):
                    continue

            # Build the multiset of out_atom labels for this combo
            combo_wyckoffs = sorted(d.out_atom for d in combo)

            for G in geoms:
                # Multiset match: G.wyckoffs vs combo out_atoms
                if sorted(G.wyckoffs) == combo_wyckoffs:
                    # Defect ordering must match geometry node ordering
                    # For now: assign in sorted order (both are sorted by wyckoff)
                    # TODO: support defect permutation optimization
                    cd = CD.from_defects(combo)
                    results.append((G, cd))

    return results


def generate_all_entries(
    enumerator: "ComplexDefectEnumerator",
    supercell_info: "SupercellInfo",
    single_defects: list,
    N_max: int,
    eps: float = 0.1,
    orders: set[int] | None = None,
) -> list["ComplexDefectEntry"]:
    """Full pipeline: enumerate geometries, assign compositions, generate structures.

    Args:
        enumerator: Configured ComplexDefectEnumerator.
        supercell_info: pydefect SupercellInfo.
        single_defects: List of pydefect SimpleDefect objects.
        N_max: Maximum number of defect components.
        eps: Tolerance for geometric equivalence (Å).
        orders: If given, only generate entries for these specific orders.
                None means all orders 2..N_max.

    Returns:
        List of ComplexDefectEntry objects ready for writing.
    """
    from .structure import ComplexDefectEntry
    import numpy as np

    all_geometries = enumerator.enumerate(N_max, eps)

    target_orders = orders if orders is not None else set(all_geometries.keys())

    entries = []
    for n, geoms in all_geometries.items():
        if n not in target_orders:
            continue
        pairs = assign_compositions(geoms, single_defects)
        for G, cd in pairs:
            try:
                struct = generate_structure(
                    enumerator.host_graph, supercell_info, G, cd,
                )
            except (ValueError, IndexError) as e:
                # Skip geometries that can't be structurally realized
                continue

            # Build defect_coords from host node positions
            defect_coords = tuple(
                tuple(float(x) for x in enumerator.host_graph.nodes[nid].frac_coord)
                for nid in G.host_node_ids
            )

            # Compute chain distances along enumeration order
            chain_dists = []
            for k in range(1, cd.n_defects):
                c_prev = enumerator.host_graph.nodes[G.host_node_ids[k - 1]].frac_coord
                c_curr = enumerator.host_graph.nodes[G.host_node_ids[k]].frac_coord
                d = enumerator.host_graph.min_image_distance(c_prev, c_curr)
                chain_dists.append(d)

            entries.append(ComplexDefectEntry(
                name=cd.name,
                complex_defect=cd,
                site_path=tuple(d.out_atom for d in cd.defects),
                distances=tuple(chain_dists),
                structure=struct,
                defect_coords=defect_coords,
                graph=G,
            ))

    return entries


# ---------------------------------------------------------------------------
# Structure generation from graph + composition (unchanged)
# ---------------------------------------------------------------------------


def generate_structure(
    host_graph: HostGraph,
    supercell_info: "SupercellInfo",
    graph: ComplexDefectGraph,
    complex_defect: ComplexDefect,
) -> "IStructure":
    """Generate defect structure from a geometry graph + defect composition.

    Applies each SimpleDefect at its corresponding host node position.
    """
    from pydefect.input_maker.defect_entries_maker import (
        copy_to_structure,
        to_istructure,
        add_atom_to_structure,
    )

    structure = copy_to_structure(supercell_info.structure)
    defects = complex_defect.defects

    for i, node_id in enumerate(graph.host_node_ids):
        d = defects[i]
        node = host_graph.nodes[node_id]

        if _is_interstitial(d.out_atom):
            # Interstitial: add atom at the interstitial site
            coords = node.frac_coord
            if d.in_atom is not None:
                add_atom_to_structure(structure, d.in_atom, coords)
        else:
            # Vacancy or substitution: remove host atom
            # Find the atom index in the current structure
            site = supercell_info.sites[d.out_atom]
            rep_idx = site.equivalent_atoms[0] if i == 0 else None

            if i == 0:
                # First defect: use the representative site
                coords = structure.pop(rep_idx).frac_coords
            else:
                # Subsequent: remove atom at the graph node's position
                # Find by matching fractional coordinates
                target = node.frac_coord
                found_idx = None
                for s_idx, s in enumerate(structure):
                    fc = s.frac_coords
                    d_vec = fc - target
                    d_vec -= np.round(d_vec)
                    if np.linalg.norm(np.dot(d_vec, host_graph.lattice)) < 0.01:
                        found_idx = s_idx
                        break
                if found_idx is None:
                    raise ValueError(f"Cannot find site at {target}")
                coords = structure.pop(found_idx).frac_coords

            if d.in_atom is not None:
                add_atom_to_structure(structure, d.in_atom, coords)

    return to_istructure(structure)