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
from .io import write_all, write_complex_defect_in_yaml, merge_defect_in

logger = logging.getLogger(__name__)


class ComplexDefectMaker:
    """Generate complex defect structures via geometry-first enumeration.

    Uses Apriori-style incremental enumeration (PLAN-C):
      1. Enumerate geometrically unique N-node site configurations.
      2. Assign defect compositions by wyckoff label matching.
      3. Generate structures and deduplicate.

    Both geometry enumeration and entry generation are cached by order.
    Calling make_all_n_body(n=4) after make_all_n_body(n=2) only computes
    N=3 and N=4 entries — lower orders are reused from cache.

    Public API:
        maker.make_pair(d1, d2)        → generate one specific pair
        maker.make_complex([d1, d2])   → generate one specific complex
        maker.make_all_pairs()         → all N=2 complexes
        maker.make_all_n_body(n=3)     → all N-body complexes
        maker.enumerate_geometries(N_max) → raw geometry enumeration
        maker.set_dopants(["X", "Y"])  → swap dopants (clears entry cache only)
        maker.write(entries, dir)      → write pydefect-compatible output
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
        )
        self._entry_cache: dict[int, list[ComplexDefectEntry]] = {}

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
        self._entry_cache.clear()  # invalidate: different dopants → different defect list
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

    # --- Defect generation ---

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

    def make_all_n_body(
        self, n=2, max_distance=None, min_distance=None, deduplicate_symmetry=True,
    ) -> list[ComplexDefectEntry]:
        """Generate all N-body complex defect combinations.

        Caches entries by order — calling with larger n only computes
        new orders, reusing cached results for lower orders.

        Uses the Apriori pipeline:
          1. Enumerate unique geometries for order n.
          2. Assign all compatible defect compositions.
          3. Deduplicate across compositions.
        """
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")

        max_d = max_distance if max_distance is not None else self.max_distance
        min_d = min_distance if min_distance is not None else self.min_distance

        # Distance param change invalidates everything
        if max_d != self.enumerator.max_distance or min_d != self.enumerator.min_distance:
            self.enumerator = ComplexDefectEnumerator(
                self.host_graph, max_distance=max_d, min_distance=min_d,
            )
            self.max_distance = max_d
            self.min_distance = min_d
            self._entry_cache.clear()

        # Determine which orders need computing
        missing = {
            k for k in range(2, n + 1)
            if k not in self._entry_cache
        }

        if missing:
            logger.info("Computing orders: %s (cached: %s)", sorted(missing),
                        sorted(self._entry_cache.keys()))
            entries = generate_all_entries(
                self.enumerator, self.supercell_info,
                self._single_defects, N_max=n,
                orders=missing,
            )

            logger.info("Total entries before dedup: %d", len(entries))

            if deduplicate_symmetry and entries:
                entries = deduplicate(entries, self.host_graph, max_d)
                logger.info("After dedup: %d entries", len(entries))

            # Partition by n_defects and cache each order
            by_order: dict[int, list[ComplexDefectEntry]] = {}
            for e in entries:
                by_order.setdefault(e.complex_defect.n_defects, []).append(e)
            for k, ents in by_order.items():
                self._entry_cache[k] = ents
                logger.info("Cached %d entries for N=%d", len(ents), k)
        else:
            logger.info("All orders 2..%d already cached", n)

        return list(self._entry_cache.get(n, []))

    def make_all_pairs(self, max_distance=None, min_distance=None, deduplicate_symmetry=True):
        return self.make_all_n_body(2, max_distance, min_distance, deduplicate_symmetry)

    # --- Output ---

    def write(self, entries, output_dir=".", merge=False) -> str:
        complex_defect_in = write_all(entries, output_dir, create_defect_json=True)
        yaml_path = write_complex_defect_in_yaml(complex_defect_in, output_dir)
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