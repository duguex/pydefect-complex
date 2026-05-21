"""Graph-based deduplication of complex defect entries.

Deduplication is geometry-first:
  1. Build geometry graphs for all entries (type-agnostic).
  2. Cluster by geometric equivalence across all compositions.
  3. Assign names: {composition}.{index} where index is the
     position of this geometry within the composition's set.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from .graph import equivalent, HostGraph, ComplexDefectGraph

if TYPE_CHECKING:
    from .structure import ComplexDefectEntry

logger = logging.getLogger(__name__)


def deduplicate(
    entries: list["ComplexDefectEntry"],
    host_graph: HostGraph,
    max_distance: float,
    eps: float = 0.1,
) -> list["ComplexDefectEntry"]:
    """Remove geometrically equivalent entries across all compositions.

    1. Build type-agnostic graphs for all entries.
    2. Cluster entries whose graphs are geometrically equivalent.
    3. Within each composition, assign sequential indices based on
       which geometric clusters that composition appears in.

    Args:
        entries: All generated ComplexDefectEntry objects.
        host_graph: Pre-built HostGraph of the pristine supercell.
        max_distance: Max distance (Å) for graph edges.
        eps: Max residual (Å) for graph edge matching.

    Returns:
        Deduplicated entries with assigned names.
    """
    if not entries:
        return []

    # Build graphs for all entries
    for entry in entries:
        if entry.graph is None:
            entry.graph = ComplexDefectGraph.from_entry(
                entry, host_graph, max_distance,
            )

    # --- Step 1: geometric clustering (type-agnostic) ---
    clusters: list[list["ComplexDefectEntry"]] = [[entries[0]]]
    entry_cluster_id: dict[int, int] = {id(entries[0]): 0}

    for e in entries[1:]:
        found = False
        for cid, cluster in enumerate(clusters):
            if equivalent(e.graph, cluster[0].graph, eps):
                cluster.append(e)
                entry_cluster_id[id(e)] = cid
                found = True
                break
        if not found:
            entry_cluster_id[id(e)] = len(clusters)
            clusters.append([e])

    n_geom = len(clusters)
    logger.info(
        "Geometry dedup: %d entries → %d unique geometries",
        len(entries), n_geom,
    )

    # --- Step 2: per-composition indexing ---
    # Collect which compositions appear in which clusters.
    # cluster_compositions[cid] = set of composition names in that cluster
    cluster_compositions = defaultdict(set)
    for cid, cluster in enumerate(clusters):
        for e in cluster:
            cluster_compositions[cid].add(e.complex_defect.name)

    # Per composition, assign index based on which clusters it appears in.
    comp_geom_indices: dict[str, dict[int, int]] = defaultdict(dict)
    for cid, comps in cluster_compositions.items():
        for comp_name in comps:
            comp_geom_indices[comp_name][cid] = len(comp_geom_indices[comp_name]) + 1

    # --- Step 3: keep one entry per (composition, geometry cluster) ---
    result = []
    seen: set[tuple[str, int]] = set()
    for e in entries:
        cid = entry_cluster_id[id(e)]
        comp_name = e.complex_defect.name
        key = (comp_name, cid)
        if key in seen:
            continue
        seen.add(key)

        idx = comp_geom_indices[comp_name][cid]
        e.name = f"{comp_name}.{idx:03d}"
        result.append(e)

    logger.info("Final: %d entries after cross-composition geometric dedup", len(result))
    return result


def deduplicate_with_distance_priority(
    entries: list["ComplexDefectEntry"],
    host_graph: HostGraph,
    max_distance: float,
    eps: float = 0.1,
) -> list["ComplexDefectEntry"]:
    """Like deduplicate, but keeps the entry with shortest total edge length."""
    if not entries:
        return []

    for entry in entries:
        if entry.graph is None:
            entry.graph = ComplexDefectGraph.from_entry(
                entry, host_graph, entry.complex_defect, max_distance,
            )

    clusters: list[list["ComplexDefectEntry"]] = [[entries[0]]]
    for e in entries[1:]:
        found = False
        for cluster in clusters:
            if equivalent(e.graph, cluster[0].graph, eps):
                cluster.append(e)
                found = True
                break
        if not found:
            clusters.append([e])

    # Keep shortest per cluster
    kept = []
    for cluster in clusters:
        best = min(cluster, key=lambda e: sum(
            float(np.linalg.norm(v)) for _, _, v in e.graph.edges
        ))
        kept.append(best)

    # Per-composition indexing
    comp_indices: dict[str, int] = defaultdict(int)
    for e in kept:
        comp_indices[e.complex_defect.name] += 1
        e.name = f"{e.complex_defect.name}.{comp_indices[e.complex_defect.name]:03d}"

    return kept


def stats(entries: list["ComplexDefectEntry"]) -> dict:
    """Return summary statistics."""
    if not entries:
        return {"total": 0}

    defect_types = set()
    all_distances = []
    n_body_counts = {}
    for e in entries:
        defect_types.add(e.complex_defect.name)
        all_distances.extend(e.distances)
        n = e.complex_defect.n_defects
        n_body_counts[n] = n_body_counts.get(n, 0) + 1

    return {
        "total": len(entries),
        "unique_defect_types": len(defect_types),
        "n_body_distribution": n_body_counts,
        "min_distance": min(all_distances) if all_distances else 0,
        "max_distance": max(all_distances) if all_distances else 0,
        "mean_distance": sum(all_distances) / len(all_distances) if all_distances else 0,
    }