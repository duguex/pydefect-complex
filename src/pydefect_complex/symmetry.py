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


def _distance_fingerprint(
    entry: "ComplexDefectEntry",
    host_graph: "HostGraph",
    decimals: int = 2,
) -> tuple[float, ...]:
    """Sorted tuple of all pairwise min-image distances between defect sites.

    Used as a fast fingerprint to verify Kabsch graph-isomorphism dedup.
    Two geometries with different fingerprints are guaranteed inequivalent;
    matching fingerprints are necessary but not sufficient for equivalence.
    """
    fc = [np.array(c) for c in entry.defect_coords]
    dists = []
    for i in range(len(fc)):
        for j in range(i + 1, len(fc)):
            d = host_graph.min_image_distance(fc[i], fc[j])
            dists.append(round(float(d), decimals))
    return tuple(sorted(dists))


def verify_dedup(
    entries: list["ComplexDefectEntry"],
    host_graph: "HostGraph",
    eps: float = 0.1,
) -> dict:
    """Verify geometric deduplication correctness using distance fingerprints.

    Does NOT change any entries or the dedup result. Returns a report dict:

        {
            "n_entries": total entries after dedup,
            "fingerprint_groups": how many distinct fingerprints,
            "false_positives": fingerprints that span multiple clusters
                              (entries with same fingerprint but in different
                              geometric clusters — may indicate under-dedup),
            "false_negatives": fingerprints where one cluster has entries with
                              different fingerprints (may indicate over-dedup),
            "ok": True if no anomalies found,
        }

    A "false positive" (same fingerprint, different cluster) means the
    Kabsch dedup may be over-strict (entries that look identical by distance
    were not merged).  A "false negative" (same cluster, different fingerprint)
    means the fingerprint is too coarse to distinguish inequivalent geometries.
    """
    import logging
    logger = logging.getLogger(__name__)

    if not entries:
        return {"n_entries": 0, "ok": True}

    # Build geometric clusters (same as deduplicate step 1)
    clusters: list[list["ComplexDefectEntry"]] = [[entries[0]]]
    entry_cluster_id: dict[int, int] = {id(entries[0]): 0}

    from .graph import equivalent, ComplexDefectGraph

    for e in entries[1:]:
        if e.graph is None:
            e.graph = ComplexDefectGraph.from_entry(e, host_graph, 4.0)
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

    # Compute fingerprints
    fp_to_clusters: dict[tuple, set[int]] = {}
    cluster_to_fps: dict[int, set[tuple]] = {}

    for cid, cluster in enumerate(clusters):
        cluster_to_fps[cid] = set()
        for e in cluster:
            fp = _distance_fingerprint(e, host_graph)
            fp_to_clusters.setdefault(fp, set()).add(cid)
            cluster_to_fps[cid].add(fp)

    false_positives = []  # same FP, different clusters
    false_negatives = []  # same cluster, different FPs

    for fp, cids in fp_to_clusters.items():
        if len(cids) > 1:
            false_positives.append({
                "fingerprint": fp,
                "cluster_ids": sorted(cids),
                "n_entries": sum(len(clusters[c]) for c in cids),
            })

    for cid, fps in cluster_to_fps.items():
        if len(fps) > 1:
            false_negatives.append({
                "cluster_id": cid,
                "fingerprints": [list(f) for f in sorted(fps)],
                "n_entries": len(clusters[cid]),
            })

    ok = len(false_positives) == 0
    report = {
        "n_entries": len(entries),
        "n_clusters": len(clusters),
        "fingerprint_groups": len(fp_to_clusters),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ok": ok,
    }

    if false_positives:
        logger.warning(
            "Dedup verification: %d fingerprint(s) span multiple clusters "
            "(possible under-dedup)", len(false_positives))
    if false_negatives:
        logger.info(
            "Dedup verification: %d cluster(s) have multiple fingerprints "
            "(expected — distance fingerprint is coarser than Kabsch)",
            len(false_negatives))

    return report


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


