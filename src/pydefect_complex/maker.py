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

from .core import ComplexDefect, _get_element
from .structure import ComplexDefectEntry
from .graph import HostGraph, ComplexDefectGraph
from .enumerate import enumerate_sites, generate_structure
from .symmetry import deduplicate
from .io import write_all, write_complex_defect_in_yaml, merge_defect_in

logger = logging.getLogger(__name__)


class ComplexDefectMaker:
    """Generate complex defect structures via geometry-first enumeration.

    Phase 1: Enumerate all N-node geometric configurations from HostGraph.
    Phase 2: For each defect composition, assign defect types to compatible
             geometric configurations and generate structures.
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

    # --- Generation ---

    def make_complex(
        self, defect_names: list[str],
        max_distance=None, min_distance=None,
    ) -> list[ComplexDefectEntry]:
        """Generate entries for a specific N-defect complex."""
        if len(defect_names) < 2:
            raise ValueError("Need at least 2 defect names")

        defects = [self._defect_map[n] for n in defect_names]
        cd = ComplexDefect.from_defects(defects)
        return self._generate_for_composition(cd, max_distance, min_distance)

    def make_pair(self, d1, d2, max_distance=None, min_distance=None):
        return self.make_complex([d1, d2], max_distance, min_distance)

    def make_all_n_body(
        self, n=2, max_distance=None, min_distance=None, deduplicate_symmetry=True,
    ) -> list[ComplexDefectEntry]:
        """Generate all N-body complex defect combinations."""
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")

        max_d = max_distance if max_distance is not None else self.max_distance
        min_d = min_distance if min_distance is not None else self.min_distance

        all_entries = []
        for combo in itertools.combinations_with_replacement(self._single_defects, n):
            cd = ComplexDefect.from_defects(list(combo))
            entries = self._generate_for_composition(cd, max_d, min_d)
            if entries:
                logger.info("%s: %d entries", cd.name, len(entries))
            all_entries.extend(entries)

        logger.info("Total entries before dedup: %d", len(all_entries))

        if deduplicate_symmetry:
            all_entries = deduplicate(all_entries, self.host_graph, max_d)
            logger.info("After dedup: %d entries", len(all_entries))

        return all_entries

    def make_all_pairs(self, max_distance=None, min_distance=None, deduplicate_symmetry=True):
        return self.make_all_n_body(2, max_distance, min_distance, deduplicate_symmetry)

    # --- Internal ---

    def _generate_for_composition(
        self, cd: ComplexDefect, max_d: float, min_d: float,
    ) -> list[ComplexDefectEntry]:
        """Phase 1+2: enumerate geometries, assign defect types, build structures."""
        # Wyckoff constraints from the defect types
        wyckoff_constraints = [d.out_atom for d in cd.defects]
        # First defect's out_atom fixes the anchor
        # (e.g. "C1" means we start from a C1 site)
        # For subsequent defects with same out_atom, they can be at any
        # symmetry-inequivalent site of the same wyckoff type
        # → all match the same wyckoff label from HostGraph

        geoms = enumerate_sites(
            self.host_graph,
            n=cd.n_defects,
            wyckoff_constraints=wyckoff_constraints,
            max_distance=max_d,
            min_distance=min_d,
        )
        if not geoms:
            return []

        entries = []
        for geom in geoms:
            # Build structure from geometry + defect composition
            struct = generate_structure(
                self.host_graph, self.supercell_info, geom, cd,
            )
            # Store defect coords for pydefect compatibility
            defect_coords = tuple(
                tuple(float(x) for x in self.host_graph.nodes[nid].frac_coord)
                for nid in geom.host_node_ids
            )
            # Compute chain distances from edges
            edge_map = {}
            for i, j, v in geom.edges:
                edge_map[(i, j)] = float(np.linalg.norm(v))
            chain_dists = []
            for k in range(1, cd.n_defects):
                d = edge_map.get((k - 1, k), edge_map.get((k, k - 1), 0.0))
                chain_dists.append(d)

            entries.append(ComplexDefectEntry(
                name=cd.name,
                complex_defect=cd,
                site_path=tuple(wyckoff_constraints),
                distances=tuple(chain_dists),
                structure=struct,
                defect_coords=defect_coords,
                graph=geom,
            ))

        return entries

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