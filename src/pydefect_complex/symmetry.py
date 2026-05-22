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


