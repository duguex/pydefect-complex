"""ComplexDefectMaker — main entry point for complex defect generation.

Usage:
    from pydefect_complex import ComplexDefectMaker

    maker = ComplexDefectMaker.from_supercell_info(
        "defect/supercell_info.json",
        dopants=["N", "B"],
        max_distance=4.0,
    )
    entries = maker.make_all_pairs()
    maker.write(entries, "defect")
"""

from __future__ import annotations

import itertools
import json
import logging
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pydefect.input_maker.supercell_info import SupercellInfo

from .core import ComplexDefect
from .structure import ComplexDefectEntry
from .graph import HostGraph, ComplexDefectGraph
from .enumerate import (
    ComplexDefectEnumerator,
    assign_compositions,
    generate_all_entries,
)
from .symmetry import deduplicate
from .io import write_all, write_complex_defect_in_yaml, merge_defect_in, write_summary

logger = logging.getLogger(__name__)


class ComplexDefectMaker:
    """Generate complex defect structures via geometry-first enumeration.

    Uses Apriori-style incremental enumeration (PLAN-C):
      1. Enumerate geometrically unique N-node site configurations.
      2. (optional) Assign defect compositions by wyckoff label matching.
      3. (optional) Generate structures and deduplicate.

    Default workflow returns geometries only — no chemistry.  Composition
    assignment and structure generation are explicit separate steps.

    Public API:
        maker.make_all_pairs()            → list[ComplexDefectGraph] (N=2)
        maker.make_all_n_body(n=3)        → list[ComplexDefectGraph] (N-body)
        maker.generate_entries(n)         → list[ComplexDefectEntry]
        maker.generate_entries(n, dopants=["N","B"])
        maker.set_dopants(["X", "Y"])     → swap default dopants for entries
        maker.make_pair(d1, d2)           → entries for one specific pair
        maker.write(entries, dir)         → pydefect-compatible output
        maker.show_geometries(N_max)      → print geometry summary
    """

    def __init__(
        self,
        supercell_info: "SupercellInfo",
        dopants: Optional[list[str]] = None,
        max_distance: float = 5.0,
        min_distance: float = 0.3,
    ):
        self.supercell_info = supercell_info
        self.dopants = dopants or []
        self.max_distance = max_distance
        self.min_distance = min_distance

        from pydefect.input_maker.defect_set_maker import DefectSetMaker
        maker = DefectSetMaker(supercell_info, dopants=self.dopants)
        self._single_defects = list(maker.defect_set)
        self._defect_map = {d.name: d for d in self._single_defects}
        self._defect_in = {d.name: d.charges for d in self._single_defects}

        self.host_graph = HostGraph.from_supercell_info(supercell_info)
        self.enumerator = ComplexDefectEnumerator(
            self.host_graph,
            max_distance=max_distance,
            min_distance=min_distance,
            pristine_structure=supercell_info.structure,
        )

    # --- Class methods ---

    @classmethod
    def from_supercell_info(
        cls, path: str, dopants=None, max_distance=5.0, min_distance=0.3,
    ) -> "ComplexDefectMaker":
        from pydefect.input_maker.supercell_info import SupercellInfo
        with open(path) as f:
            data = json.load(f)
        return cls(SupercellInfo.from_dict(data), dopants, max_distance, min_distance)

    # --- Properties ---

    @property
    def single_defects(self) -> list:
        return self._single_defects

    @property
    def defect_in(self) -> dict[str, list[int]]:
        return dict(self._defect_in)

    @property
    def defect_names(self) -> list[str]:
        return list(self._defect_map.keys())

    @property
    def defect_pairs(self) -> list[tuple]:
        return list(itertools.combinations_with_replacement(self._single_defects, 2))

    def set_dopants(self, dopants: Optional[list[str]] = None):
        """Replace dopants without resetting geometry enumeration cache.

        Rebuilds the single-defect list from pydefect's DefectSetMaker
        while keeping the existing HostGraph and enumerator (with any
        already-cached geometries). Use this to re-generate complexes
        with different dopants without re-enumerating geometries.

        Args:
            dopants: New dopant list (empty list = intrinsic only).
        """
        from pydefect.input_maker.defect_set_maker import DefectSetMaker

        self.dopants = dopants or []
        maker = DefectSetMaker(self.supercell_info, dopants=self.dopants)
        self._single_defects = list(maker.defect_set)
        self._defect_map = {d.name: d for d in self._single_defects}
        self._defect_in = {d.name: d.charges for d in self._single_defects}

    # --- Geometry enumeration (public) ---

    def enumerate_geometries(
        self, N_max: int, eps: float = 0.1,
    ) -> dict[int, list[ComplexDefectGraph]]:
        """Enumerate geometrically unique N-node subgraphs.

        Returns {2: [G, ...], 3: [G, ...], ..., N_max: [...]}.
        Cached — repeated calls with higher N_max reuse prior results.
        """
        return self.enumerator.enumerate(N_max, eps)

    # --- Geometry enumeration (default: no chemistry) ---

    def make_all_n_body(
        self, n: int = 2,
        max_distance: float | None = None,
        min_distance: float | None = None,
    ) -> list[ComplexDefectGraph]:
        """Enumerate geometrically unique N-body site configurations.

        Default workflow — returns geometries only, no defect chemistry.
        Use generate_entries() afterwards to assign compositions and
        generate structures.

        Returns list[ComplexDefectGraph] for order n.
        """
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")

        max_d = max_distance if max_distance is not None else self.max_distance
        min_d = min_distance if min_distance is not None else self.min_distance

        if max_d != self.enumerator.max_distance or min_d != self.enumerator.min_distance:
            self.enumerator = ComplexDefectEnumerator(
                self.host_graph, max_distance=max_d, min_distance=min_d,
                pristine_structure=self.supercell_info.structure,
            )
            self.max_distance = max_d
            self.min_distance = min_d

        self.enumerator.enumerate(n)
        return list(self.enumerator.geometries.get(n, []))

    def make_all_pairs(
        self, max_distance=None, min_distance=None,
    ) -> list[ComplexDefectGraph]:
        """Enumerate all N=2 geometries (no chemistry)."""
        return self.make_all_n_body(2, max_distance, min_distance)

    # --- Explicit composition assignment + entry generation -----------

    def generate_entries(
        self,
        n_or_geometries: int | list[ComplexDefectGraph] = 2,
        dopants: list[str] | None = None,
        max_distance: float | None = None,
        min_distance: float | None = None,
        deduplicate_symmetry: bool = True,
    ) -> list[ComplexDefectEntry]:
        """Assign defect compositions and generate structures.

        Args:
            n_or_geometries: Either N (int) to generate all N-body entries,
                             or a list of ComplexDefectGraph to generate
                             entries for specific geometries.
            dopants: Override default dopants for this call.
            max_distance, min_distance: Override distance cutoffs.
            deduplicate_symmetry: Whether to cross-composition dedup.

        Returns list[ComplexDefectEntry].
        """
        if dopants is not None:
            self.set_dopants(dopants)

        max_d = max_distance if max_distance is not None else self.max_distance
        min_d = min_distance if min_distance is not None else self.min_distance

        if max_d != self.enumerator.max_distance or min_d != self.enumerator.min_distance:
            self.enumerator = ComplexDefectEnumerator(
                self.host_graph, max_distance=max_d, min_distance=min_d,
                pristine_structure=self.supercell_info.structure,
            )
            self.max_distance = max_d
            self.min_distance = min_d

        if isinstance(n_or_geometries, int):
            n = n_or_geometries
            # Enumerate if not cached
            self.enumerator.enumerate(n)
            geometries = self.enumerator.geometries.get(n, [])
        else:
            geometries = n_or_geometries
            n = max(g.n_defects for g in geometries) if geometries else 2
            self.enumerator.enumerate(n)

        entries = generate_all_entries(
            self.enumerator, self.supercell_info,
            self._single_defects, N_max=n,
        )

        logger.info("Total entries before dedup: %d", len(entries))

        if deduplicate_symmetry and entries:
            entries = deduplicate(entries, self.host_graph, max_d)
            logger.info("After dedup: %d entries", len(entries))

        # Filter to requested geometries/order
        if isinstance(n_or_geometries, list):
            geom_node_sets = {tuple(sorted(g.host_node_ids)) for g in n_or_geometries}
            entries = [e for e in entries
                       if e.graph is not None
                       and tuple(sorted(e.graph.host_node_ids)) in geom_node_sets]
        else:
            entries = [e for e in entries if e.complex_defect.n_defects == n_or_geometries]

        return entries

    def make_complex(
        self, defect_names: list[str],
        max_distance=None, min_distance=None,
    ) -> list[ComplexDefectEntry]:
        """Generate entries for a specific N-defect complex."""
        if len(defect_names) < 2:
            raise ValueError("Need at least 2 defect names")

        defects = [self._defect_map[n] for n in defect_names]
        cd = ComplexDefect.from_defects(defects)

        max_d = max_distance if max_distance is not None else self.max_distance
        min_d = min_distance if min_distance is not None else self.min_distance

        if max_d != self.enumerator.max_distance or min_d != self.enumerator.min_distance:
            self.enumerator = ComplexDefectEnumerator(
                self.host_graph, max_distance=max_d, min_distance=min_d,
                pristine_structure=self.supercell_info.structure,
            )

        all_entries = generate_all_entries(
            self.enumerator, self.supercell_info,
            self._single_defects, N_max=cd.n_defects,
        )
        entries = [e for e in all_entries if e.complex_defect.name == cd.name]
        if entries:
            entries = deduplicate(entries, self.host_graph, max_d)
        return entries

    def make_pair(self, d1, d2, max_distance=None, min_distance=None):
        return self.make_complex([d1, d2], max_distance, min_distance)

    def show_geometries(self, N_max: int = 2):
        """Print a human-readable geometry summary (no chemistry)."""
        geo = self.enumerate_geometries(N_max)
        for n in sorted(geo):
            geoms = geo[n]
            print(f"\n=== N={n} 几何构型 ({len(geoms)} 个) ===")
            for i, g in enumerate(geoms):
                dists = sorted([float(np.linalg.norm(v)) for _, _, v in g.edges])
                ds = ", ".join(f"{d:.2f}" for d in dists)
                print(f"  G{i:3d}: edges={len(g.edges)} dists=[{ds}] "
                      f"n_orient={g.n_orientations} pg={g.point_group} "
                      f"wyckoffs={g.wyckoffs}")

    def save_geometry_cache(self, output_dir: str = "."):
        """Write geometry cache to JSON files for later reuse."""
        import json
        from pathlib import Path
        out = Path(output_dir)
        for n, geoms in self.enumerator.geometries.items():
            path = out / f"cache_geometry_N{n}.json"
            path.write_text(json.dumps(
                [g.to_dict() for g in geoms], indent=2, ensure_ascii=False))
            logger.info("Saved %d geometries to %s", len(geoms), path)

    @staticmethod
    def load_geometries(path: str) -> list[ComplexDefectGraph]:
        """Load geometries from a cache JSON file."""
        return ComplexDefectGraph.load_json(path)

    # --- Output ---

    def write(self, entries, output_dir=".", merge=False) -> str:
        complex_defect_in = write_all(entries, output_dir, create_defect_json=True)
        yaml_path = write_complex_defect_in_yaml(complex_defect_in, output_dir)
        summary_path = write_summary(entries, output_dir)
        logger.info("Summary written to %s", summary_path)
        if merge:
            merge_defect_in(output_dir)
        return yaml_path

    # --- Info ---

    def summary(self) -> dict:
        return {
            "n_single_defects": len(self._single_defects),
            "n_pairs": len(self.defect_pairs),
            "dopants": self.dopants,
            "max_distance": self.max_distance,
            "min_distance": self.min_distance,
            "defect_names": self.defect_names,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ComplexDefectMaker("
            f"n_defects={s['n_single_defects']}, "
            f"n_pairs={s['n_pairs']}, "
            f"dopants={s['dopants']}, "
            f"max_dist={s['max_distance']} Å)"
        )