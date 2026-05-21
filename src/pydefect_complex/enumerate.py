"""Graph-based complex defect enumeration.

Replaces the recursive structure-generation algorithm with
direct HostGraph enumeration:

  1. Enumerate all N-node site combinations satisfying distance constraints.
  2. Build geometry graphs, deduplicate.
  3. For each (geometry, composition) pair, generate the defect structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import IStructure
    from pydefect.input_maker.supercell_info import SupercellInfo
    from pydefect.input_maker.defect import SimpleDefect

from .core import ComplexDefect, _get_element, _is_interstitial
from .graph import HostGraph, ComplexDefectGraph, _edge_list, equivalent


# ---------------------------------------------------------------------------
# Site enumeration
# ---------------------------------------------------------------------------


def _min_image_dist(
    fc_a: np.ndarray, fc_b: np.ndarray, lattice: np.ndarray,
) -> float:
    d = fc_a - fc_b
    d -= np.round(d)
    return float(np.linalg.norm(np.dot(d, lattice)))


def enumerate_sites(
    host_graph: HostGraph,
    n: int,
    wyckoff_constraints: list[str],
    max_distance: float,
    min_distance: float = 0.3,
) -> list[ComplexDefectGraph]:
    """Enumerate all N-node subgraphs using symmetry reduction.

    Uses StructureSymmetrizer at each layer to enumerate only
    symmetry-inequivalent sites, mapping back to host graph nodes
    via KDTree lookup.
    """
    if len(wyckoff_constraints) != n:
        raise ValueError(
            f"wyckoff_constraints length {len(wyckoff_constraints)} != n={n}"
        )

    from pydefect.input_maker.defect_entries_maker import copy_to_structure, to_istructure
    from vise.util.structure_symmetrizer import StructureSymmetrizer

    supercell = host_graph._get_structure()

    # Anchor: first defect's out_atom → representative site
    anchor_w = wyckoff_constraints[0]
    results = []

    # Pre-filter: nodes matching each wyckoff constraint
    node_by_wyckoff = {}
    for node in host_graph.nodes:
        node_by_wyckoff.setdefault(node.wyckoff, []).append(node)

    # Anchor from representative site
    anchor_nodes = node_by_wyckoff.get(anchor_w, [])
    if not anchor_nodes:
        return []

    anchor = anchor_nodes[0]  # representative (lowest index)

    # Start recursion with symmetrizer on pristine structure
    no_defect_sym = StructureSymmetrizer(supercell)
    pristine = copy_to_structure(supercell)
    lattice = host_graph.lattice

    _recurse_sym(
        host_graph, pristine, lattice,
        wyckoff_constraints, no_defect_sym,
        max_distance, min_distance,
        path_ids=[anchor.id],
        path_coords=[anchor.frac_coord],
        layer_idx=1,
        output=results,
    )
    return results


def _recurse_sym(
    host_graph, current_structure, lattice,
    wyckoff_constraints, pristine_sym,
    max_distance, min_distance,
    path_ids, path_coords, layer_idx, output,
):
    """Symmetry-reduced DFS."""
    if layer_idx >= len(wyckoff_constraints):
        # Build graph
        node_ids = tuple(path_ids)
        coords = [tuple(float(x) for x in c) for c in path_coords]
        edges = _edge_list(coords, host_graph, max_distance)
        nodes = host_graph.nodes
        output.append(ComplexDefectGraph(
            host_node_ids=node_ids,
            wyckoffs=tuple(nodes[i].wyckoff for i in node_ids),
            elements=tuple(nodes[i].element for i in node_ids),
            edges=edges,
        ))
        return

    from vise.util.structure_symmetrizer import StructureSymmetrizer
    sym = StructureSymmetrizer(current_structure)

    target_w = wyckoff_constraints[layer_idx]
    prev_coord = path_coords[-1]

    for site_name in sym.sites:
        if site_name != target_w:
            continue

        site = sym.sites[site_name]
        idx = site.equivalent_atoms[0]
        site_coord = current_structure[idx].frac_coords

        # Overlap check
        too_close = False
        for pc in path_coords:
            d = site_coord - pc
            d -= np.round(d)
            if float(np.linalg.norm(np.dot(d, lattice))) < min_distance:
                too_close = True
                break
        if too_close:
            continue

        # Chain distance
        d = site_coord - prev_coord
        d -= np.round(d)
        dist = float(np.linalg.norm(np.dot(d, lattice)))
        if not (min_distance < dist <= max_distance):
            continue

        # Map to host graph node via KDTree
        nid = host_graph.find_node(site_coord)

        if nid in path_ids:
            continue

        # Remove this site from structure for symmetry analysis of next layer
        from pydefect.input_maker.defect_entries_maker import copy_to_structure, to_istructure
        new_struct = copy_to_structure(current_structure)
        # Find the atom at this position in new_struct
        found = None
        for s_i, s in enumerate(new_struct):
            dv = s.frac_coords - site_coord
            dv -= np.round(dv)
            if np.linalg.norm(np.dot(dv, lattice)) < 0.01:
                found = s_i
                break
        if found is not None:
            new_struct.pop(found)

        _recurse_sym(
            host_graph, to_istructure(new_struct), lattice,
            wyckoff_constraints, pristine_sym,
            max_distance, min_distance,
            path_ids + [nid],
            path_coords + [site_coord],
            layer_idx + 1,
            output,
        )


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