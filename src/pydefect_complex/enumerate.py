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
from .graph import HostGraph, ComplexDefectGraph, _edge_list


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
# Structure generation from graph + composition
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