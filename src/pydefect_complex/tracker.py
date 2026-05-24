"""Pipeline intermediate result tracker.

Writes numbered stage outputs to a ``_cache/`` subdirectory for
offline inspection.  Each pipeline stage produces one YAML or JSON file.

Usage::

    tracker = PipelineTracker("defect", enabled=True)
    tracker.write_parameters(maker.summary())
    tracker.write_geometries(geometries, {"max_distance": 4.0, ...})
    tracker.write_composition_mapping(pairs)
    tracker.write_dedup_clusters(n_before, clusters, cluster_comps, indices)
    tracker.write_dedup_verification(report)
    tracker.write_entries_metadata(entries)
"""

from __future__ import annotations

import datetime
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .graph import ComplexDefectGraph
    from .structure import ComplexDefectEntry

from .log import get_logger

logger = get_logger(__name__)


class PipelineTracker:
    """Write intermediate pipeline results to ``_cache/``.

    Each method dumps one stage's output as a standalone YAML/JSON file.
    Files are numbered in pipeline order so they sort naturally.
    """

    def __init__(
        self,
        output_dir: str,
        enabled: bool = False,
    ):
        self._cache_dir = Path(output_dir) / "_cache"
        self._enabled = enabled
        if enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _write_yaml(self, filename: str, data: dict) -> str:
        path = self._cache_dir / filename
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=None, sort_keys=False)
        return str(path)

    def _write_json(self, filename: str, data: dict) -> str:
        path = self._cache_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(path)

    # ------------------------------------------------------------------
    # Stage outputs
    # ------------------------------------------------------------------

    def write_parameters(self, maker_summary: dict) -> Optional[str]:
        """Write run metadata.

        File: ``_cache/00_parameters.yaml``
        """
        if not self._enabled:
            return None
        try:
            from importlib.metadata import version
            pkg_version = version("pydefect-complex")
        except Exception:
            pkg_version = "0.1.0-dev"
        data = {
            "pipeline_stage": "00_parameters",
            "timestamp": datetime.datetime.now().isoformat(),
            "pydefect_complex_version": pkg_version,
            **maker_summary,
        }
        path = self._write_yaml("00_parameters.yaml", data)
        logger.info("TRACKER: wrote %s", Path(path).name)
        return path

    def write_geometries(
        self,
        geometries: dict[int, list["ComplexDefectGraph"]],
        parameters: dict,
    ) -> Optional[str]:
        """Write geometry enumeration results.

        File: ``_cache/01_geometries.yaml``
        """
        if not self._enabled or not geometries:
            return None
        import numpy as np

        geom_list = []
        for n in sorted(geometries):
            for i, g in enumerate(geometries[n]):
                dists = sorted(
                    float(np.linalg.norm(v)) for _, _, v in g.edges
                )
                geom_list.append({
                    "order": n,
                    "geometry_index": i,
                    "host_node_ids": list(g.host_node_ids),
                    "wyckoffs": list(g.wyckoffs),
                    "elements": list(g.elements),
                    "n_edges": len(g.edges),
                    "distances": [round(d, 3) for d in dists],
                    "n_orientations": g.n_orientations,
                    "point_group": g.point_group,
                })
        data = {
            "pipeline_stage": "01_geometry_enumerate",
            "parameters": parameters,
            "orders": {n: len(gs) for n, gs in sorted(geometries.items())},
            "n_total": len(geom_list),
            "geometries_by_order": geom_list,
        }
        path = self._write_yaml("01_geometries.yaml", data)
        logger.info("TRACKER: wrote %s (%d geometries)", Path(path).name, len(geom_list))
        return path

    def write_composition_mapping(
        self,
        pairs: list[tuple],
    ) -> Optional[str]:
        """Write composition-to-geometry mapping.

        File: ``_cache/02_composition_mapping.yaml``
        """
        if not self._enabled or not pairs:
            return None
        comps: dict[str, list[dict]] = defaultdict(list)
        for G, cd in pairs:
            comps[cd.name].append({
                "host_node_ids": list(G.host_node_ids),
                "wyckoffs": list(G.wyckoffs),
                "elements": list(G.elements),
                "n_edges": len(G.edges),
                "n_orientations": G.n_orientations,
                "point_group": G.point_group,
            })
        data = {
            "pipeline_stage": "02_composition_mapping",
            "n_pairs": len(pairs),
            "compositions": dict(comps),
        }
        path = self._write_yaml("02_composition_mapping.yaml", data)
        logger.info(
            "TRACKER: wrote %s (%d pairs, %d compositions)",
            Path(path).name, len(pairs), len(comps),
        )
        return path

    def write_dedup_clusters(
        self,
        entries_before: int,
        clusters: list[list],
        cluster_compositions: dict,
        comp_geom_indices: dict,
    ) -> Optional[str]:
        """Write dedup cluster details.

        File: ``_cache/03_dedup_clusters.yaml``
        """
        if not self._enabled:
            return None
        cluster_data = []
        for cid, cluster in enumerate(clusters):
            comps_in = sorted(set(e.complex_defect.name for e in cluster))
            cluster_data.append({
                "cluster_id": cid,
                "n_entries": len(cluster),
                "compositions": comps_in,
                "example_name": cluster[0].name if cluster else None,
            })
        data = {
            "pipeline_stage": "03_dedup_clusters",
            "entries_before": entries_before,
            "n_clusters": len(clusters),
            "clusters": cluster_data,
            "comp_geom_indices": {
                comp: dict(indices)
                for comp, indices in comp_geom_indices.items()
            },
        }
        path = self._write_yaml("03_dedup_clusters.yaml", data)
        logger.info("TRACKER: wrote %s (%d clusters)", Path(path).name, len(clusters))
        return path

    def write_dedup_verification(self, report: dict) -> Optional[str]:
        """Write dedup verification results.

        File: ``_cache/04_dedup_verification.yaml``
        """
        if not self._enabled or not report:
            return None
        data: dict = {"pipeline_stage": "04_dedup_verification", **report}
        path = self._write_yaml("04_dedup_verification.yaml", data)
        if not report.get("ok", True):
            logger.warning(
                "TRACKER: wrote %s — %d false positive fingerprint(s)",
                Path(path).name, len(report.get("false_positives", [])),
            )
        else:
            logger.info("TRACKER: wrote %s (ok=True)", Path(path).name)
        return path

    def write_entries_metadata(
        self, entries: list["ComplexDefectEntry"],
    ) -> Optional[str]:
        """Write entry metadata (no pymatgen structures, only scalar fields).

        File: ``_cache/05_entries.json``
        """
        if not self._enabled or not entries:
            return None
        entry_list = []
        for e in entries:
            entry_list.append({
                "name": e.name,
                "composition": e.complex_defect.name,
                "n_defects": e.complex_defect.n_defects,
                "charges": e.complex_defect.charges,
                "distances": [round(float(d), 4) for d in e.distances],
                "defect_coords": [
                    [round(float(c), 6) for c in coord]
                    for coord in e.defect_coords
                ],
                "formula": str(e.structure.composition.formula),
                "n_atoms": len(e.structure),
                "n_orientations": e.n_orientations,
                "point_group": e.point_group,
                "space_group": e.space_group,
            })
        data = {
            "pipeline_stage": "05_entries",
            "n_entries": len(entry_list),
            "entries": entry_list,
        }
        path = self._write_json("05_entries.json", data)
        logger.info("TRACKER: wrote %s (%d entries)", Path(path).name, len(entry_list))
        return path