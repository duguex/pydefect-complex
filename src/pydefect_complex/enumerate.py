"""Graph-based complex defect enumeration via Apriori-style incremental geometry search.

  1. Enumerate geometrically unique N-node site configurations.
  2. Assign defect compositions by wyckoff label matching.
  3. For each (geometry, composition) pair, generate the defect structure.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import IStructure
    from pydefect.input_maker.supercell_info import SupercellInfo

from .core import ComplexDefect, _get_element, _is_interstitial
from .graph import HostGraph, ComplexDefectGraph, _edge_list, equivalent
from .log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional progress bar helper
# ---------------------------------------------------------------------------


def _maybe_tqdm(iterable, **kwargs):
    """Wrap *iterable* in tqdm if installed, otherwise return as-is."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


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
        pristine_structure=None,
        n_workers: int | None = None,
    ):
        self.host_graph = host_graph
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.pristine_structure = pristine_structure
        self._cache: dict[int, list[ComplexDefectGraph]] = {}
        self.n_workers = n_workers if n_workers is not None else max(1, os.cpu_count() or 1)

    @property
    def geometries(self) -> dict[int, list[ComplexDefectGraph]]:
        """Currently cached geometries, or empty dict if not yet enumerated."""
        return dict(self._cache)

    def enumerate(
        self, N_max: int, eps: float = 0.1,
    ) -> dict[int, list[ComplexDefectGraph]]:
        """Enumerate all geometrically unique N-node subgraphs for orders 2..N_max.

        Args:
            N_max: Maximum number of nodes (>= 2).
            eps: Geometric equivalence tolerance (Å).

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
            logger.debug("ENUMERATE: cache hit for N_max=%d (max_cached=%d)", N_max, max_cached)
            return {k: v for k, v in self._cache.items() if k <= N_max}

        # Bootstrap N=2 if needed
        if 2 not in self._cache:
            self._cache[2] = self._enumerate_2(eps)
            self._compute_orientations(self._cache[2])

        # Apriori: k → k+1
        start_k = max(2, max_cached)
        orders_iter = range(start_k, N_max)
        if start_k < N_max:
            orders_iter = _maybe_tqdm(
                orders_iter,
                desc="Enumerating geometries",
                unit="order",
            )
        for k in orders_iter:
            if k + 1 not in self._cache:
                self._cache[k + 1] = self._extend_order(k, eps)
                self._compute_orientations(self._cache[k + 1])

        return {k: v for k, v in self._cache.items() if k <= N_max}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_orientations(self, geometries: list[ComplexDefectGraph]):
        """Compute orientation counts for all geometries in-place.

        Uses the host crystal's space group — geometry only, no chemistry.
        """
        if self.pristine_structure is None:
            return
        from .structure import _count_orientations_from_coords
        hg = self.host_graph
        sym_ops = hg._get_sym_ops()
        for G in geometries:
            if G.n_orientations >= 0:
                continue
            fc = [tuple(hg.nodes[nid].frac_coord) for nid in G.host_node_ids]
            n_orient, pg = _count_orientations_from_coords(
                fc, self.pristine_structure, sym_ops)
            G.n_orientations = n_orient
            G.point_group = pg

        if geometries and any(G.point_group for G in geometries):
            from collections import Counter
            pg_counts = Counter(G.point_group for G in geometries if G.point_group)
            pg_summary = ", ".join(f"{pg}={n}" for pg, n in pg_counts.most_common(8))
            n_orient = sum(max(0, G.n_orientations) for G in geometries)
            logger.info(
                "ORIENTATIONS: %d geometries, %d total orientations, "
                "point groups: %s",
                len(geometries), n_orient, pg_summary,
            )

    def _enumerate_2(self, eps: float) -> list[ComplexDefectGraph]:
        """Generate all unique 2-node geometries from anchor+neighbor pairs.

        Anchors are chosen as one representative per (wyckoff, element) class.
        """
        geometries: list[ComplexDefectGraph] = []
        seen_anchor_classes: set[tuple[str, str]] = set()

        anchor_iterator = _maybe_tqdm(
            self.host_graph.nodes,
            desc="N=2 anchors",
            unit=" anchor",
            leave=False,
        )
        for anchor in anchor_iterator:
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

        logger.info(
            "ENUMERATE N=2: %d unique geometries from %d anchor classes "
            "(max_dist=%.1f min_dist=%.1f eps=%.2f)",
            len(geometries), len(seen_anchor_classes),
            self.max_distance, self.min_distance, eps,
        )
        return geometries

    def _extend_order(self, k: int, eps: float) -> list[ComplexDefectGraph]:
        """Extend unique k-geometries to (k+1)-geometries.

        Dispatches to serial or parallel implementation based on
        ``self.n_workers``.
        """
        if self.n_workers <= 1:
            return self._extend_order_serial(k, eps)
        return self._extend_order_parallel(k, eps)

    def _extend_order_serial(self, k: int, eps: float) -> list[ComplexDefectGraph]:
        """Serial extension: k → k+1."""
        result: list[ComplexDefectGraph] = []

        iterator = _maybe_tqdm(
            self._cache[k],
            desc=f"N={k} → {k+1}",
            unit=f" N={k} geom",
            leave=False,
        )
        for G_k in iterator:
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

        logger.info(
            "ENUMERATE N=%d: %d unique geometries from %d N=%d geometries",
            k + 1, len(result), len(self._cache[k]), k,
        )
        return result

    def _extend_order_parallel(self, k: int, eps: float) -> list[ComplexDefectGraph]:
        """Parallel chunk-then-merge extension: k → k+1."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        geometries = list(self._cache[k])
        n_workers = min(self.n_workers, len(geometries))
        if n_workers <= 1:
            return self._extend_order_serial(k, eps)

        chunks = [geometries[i::n_workers] for i in range(n_workers)]
        chunks = [c for c in chunks if c]

        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = [
                exe.submit(
                    _extend_chunk_worker,
                    self.host_graph,
                    chunk,
                    self.max_distance,
                    self.min_distance,
                    eps,
                )
                for chunk in chunks
            ]

            chunk_results: list[ComplexDefectGraph] = []
            iterator = _maybe_tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"N={k}→{k+1} chunks",
                leave=False,
            )
            for f in iterator:
                chunk_results.extend(f.result())

        # Merge: cross-dedup across chunks
        result: list[ComplexDefectGraph] = []
        iterator = _maybe_tqdm(
            chunk_results,
            desc=f"N={k+1} merge",
            leave=False,
        )
        for g in iterator:
            if not any(equivalent(g, r, eps) for r in result):
                result.append(g)

        logger.info(
            "ENUMERATE N=%d: %d unique geometries (%d workers, "
            "%d pre-merge candidates)",
            k + 1, len(result), n_workers, len(chunk_results),
        )
        return result


# ---------------------------------------------------------------------------
# Parallel worker: extend a chunk of N=k geometries (local dedup)
# ---------------------------------------------------------------------------


def _extend_chunk_worker(
    host_graph: HostGraph,
    chunk: list[ComplexDefectGraph],
    max_distance: float,
    min_distance: float,
    eps: float,
) -> list[ComplexDefectGraph]:
    """Worker for :meth:`ComplexDefectEnumerator._extend_order_parallel`.

    For each N=k geometry in *chunk*, finds external neighbors,
    builds candidate N=(k+1) graphs, and deduplicates locally.
    """
    result: list[ComplexDefectGraph] = []
    for G_k in chunk:
        ext_neighbors = host_graph.neighbors_of_set(
            set(G_k.host_node_ids), max_distance,
        )
        for nbr_id in ext_neighbors:
            nbr = host_graph.nodes[nbr_id]

            too_close = any(
                host_graph.min_image_distance(
                    nbr.frac_coord,
                    host_graph.nodes[hid].frac_coord,
                )
                < min_distance
                for hid in G_k.host_node_ids
            )
            if too_close:
                continue

            new_ids = tuple(list(G_k.host_node_ids) + [nbr_id])
            coords = [
                tuple(host_graph.nodes[i].frac_coord) for i in new_ids
            ]
            edges = _edge_list(coords, host_graph, max_distance)

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
# Parallel entry generation worker
# ---------------------------------------------------------------------------


def _entry_batch_worker(
    host_graph: HostGraph,
    structure: "IStructure",
    sites: dict,
    batch: list[tuple[ComplexDefectGraph, "ComplexDefect"]],
    charges: list[int] | None,
) -> list["ComplexDefectEntry"]:
    """Generate ComplexDefectEntry objects for a batch of (G, cd) pairs."""
    from copy import copy
    from .structure import ComplexDefectEntry

    results: list[ComplexDefectEntry] = []
    for G, cd in batch:
        cd_local = copy(cd)
        if charges is not None:
            cd_local.charges = list(charges)
        try:
            struct = _generate_structure(
                host_graph, structure, sites, G, cd_local,
            )
        except (ValueError, IndexError):
            continue

        defect_coords = tuple(
            tuple(float(x) for x in host_graph.nodes[nid].frac_coord)
            for nid in G.host_node_ids
        )

        chain_dists = []
        for k in range(1, cd_local.n_defects):
            cp = host_graph.nodes[G.host_node_ids[k - 1]].frac_coord
            cc = host_graph.nodes[G.host_node_ids[k]].frac_coord
            d = host_graph.min_image_distance(cp, cc)
            chain_dists.append(d)

        results.append(ComplexDefectEntry(
            name=cd_local.name,
            complex_defect=cd_local,
            site_path=tuple(d.out_atom for d in cd_local.defects),
            distances=tuple(chain_dists),
            structure=struct,
            defect_coords=defect_coords,
            graph=G,
        ))
    return results


def _generate_entries_parallel(
    pairs: list[tuple[ComplexDefectGraph, "ComplexDefect"]],
    host_graph: HostGraph,
    structure: "IStructure",
    sites: dict,
    charges: list[int] | None,
    n_workers: int,
) -> list["ComplexDefectEntry"]:
    """Generate entries in parallel using ProcessPoolExecutor."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = min(n_workers, len(pairs))
    chunks = [pairs[i::n_workers] for i in range(n_workers)]
    chunks = [c for c in chunks if c]

    entries: list[ComplexDefectEntry] = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = [
            exe.submit(
                _entry_batch_worker,
                host_graph, structure, sites, chunk, charges,
            )
            for chunk in chunks
        ]

        iterator = _maybe_tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating structures",
            unit=" batch",
        )
        for f in iterator:
            entries.extend(f.result())

    logger.info(
        "ENTRIES PARALLEL: generated %d entries (%d workers)",
        len(entries), n_workers,
    )
    return entries


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
                    cd = CD.from_defects(combo)
                    results.append((G, cd))

    n_orders = len(by_n)
    logger.info(
        "COMPOSITIONS: %d (geometry, composition) pairs from "
        "%d geometries and %d single-defect types across %d orders",
        len(results), len(geometries), len(single_defects), n_orders,
    )
    return results


def generate_all_entries(
    enumerator: "ComplexDefectEnumerator",
    supercell_info: "SupercellInfo",
    single_defects: list,
    N_max: int,
    eps: float = 0.1,
    orders: set[int] | None = None,
    charges: list[int] | None = None,
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
        charges: Charge states to assign to every generated entry.
                 None uses ComplexDefect default (neutral only).

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
        n_before = len(entries)
        n_pairs = 0
        pairs = assign_compositions(geoms, single_defects)
        n_pairs = len(pairs)
        n_workers = enumerator.n_workers
        if n_workers > 1 and len(pairs) >= n_workers:
            entries += _generate_entries_parallel(
                pairs, enumerator.host_graph, supercell_info.structure,
                supercell_info.sites, charges, n_workers,
            )
        else:
            for G, cd in pairs:
                if charges is not None:
                    cd.charges = list(charges)
                try:
                    struct = _generate_structure(
                        enumerator.host_graph, supercell_info.structure,
                        supercell_info.sites, G, cd,
                    )
                except (ValueError, IndexError):
                    continue

                defect_coords = tuple(
                    tuple(float(x) for x in enumerator.host_graph.nodes[nid].frac_coord)
                    for nid in G.host_node_ids
                )

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
                    pristine_structure_cache=supercell_info.structure,
                ))

        logger.info(
            "ENTRIES N=%d: %d entries from %d geometries and %d composition pairs",
            n, len(entries) - n_before, len(geoms), n_pairs,
        )

    return entries


# ---------------------------------------------------------------------------
# Structure generation from graph + composition (unchanged)
# ---------------------------------------------------------------------------


def _generate_structure(
    host_graph: HostGraph,
    structure: "IStructure",
    sites: dict,
    graph: ComplexDefectGraph,
    complex_defect: ComplexDefect,
) -> "IStructure":
    """Internal: generate defect structure from raw structure + sites dict.

    ``sites`` is ``supercell_info.sites`` (a dict of pydefect Site objects).
    """
    from pydefect.input_maker.defect_entries_maker import (
        copy_to_structure,
        to_istructure,
        add_atom_to_structure,
    )

    structure = copy_to_structure(structure)
    defects = complex_defect.defects

    for i, node_id in enumerate(graph.host_node_ids):
        d = defects[i]
        node = host_graph.nodes[node_id]

        if _is_interstitial(d.out_atom):
            coords = node.frac_coord
            if d.in_atom is not None:
                add_atom_to_structure(structure, d.in_atom, coords)
        else:
            site = sites[d.out_atom]
            rep_idx = site.equivalent_atoms[0] if i == 0 else None

            if i == 0:
                coords = structure.pop(rep_idx).frac_coords
            else:
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


def generate_structure(
    host_graph: HostGraph,
    supercell_info: "SupercellInfo",
    graph: ComplexDefectGraph,
    complex_defect: ComplexDefect,
) -> "IStructure":
    """Generate defect structure from a geometry graph + defect composition.

    Applies each SimpleDefect at its corresponding host node position.
    """
    return _generate_structure(
        host_graph, supercell_info.structure, supercell_info.sites,
        graph, complex_defect,
    )