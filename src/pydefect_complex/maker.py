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
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from pydefect.input_maker.supercell_info import SupercellInfo

from .core import ComplexDefect
from .structure import ComplexDefectEntry
from .graph import HostGraph, ComplexDefectGraph
from .enumerate import (
    ComplexDefectEnumerator,
    generate_all_entries,
)
from .symmetry import deduplicate, verify_dedup
from .io import _write_all, write_complex_defect_in_yaml, merge_defect_in, write_summary
from .log import get_logger
from .tracker import PipelineTracker

logger = get_logger(__name__)


def _group_by_n_defects(entries):
    """Group entries by n_defects for populating entry_cache."""
    result: dict[int, list] = {}
    for e in entries:
        n = e.complex_defect.n_defects
        result.setdefault(n, []).append(e)
    return result


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
        max_distance: float = 3.0,
        min_distance: float = 0.3,
        charges: list[int] | None = None,
        verbose: bool = False,
        track_pipeline: bool = False,
        n_workers: int | None = None,
        skip_defects: bool = False,
    ):
        self.supercell_info = supercell_info
        self.dopants = dopants or []
        self.max_distance = max_distance
        self.min_distance = min_distance
        self._charges = charges if charges is not None else [0]
        self._track_pipeline = track_pipeline
        self._n_workers = n_workers

        self.host_graph = HostGraph.from_supercell_info(supercell_info)

        if skip_defects:
            self._single_defects = []
            self._defect_map = {}
            self._defect_in = {}
        else:
            from pydefect.input_maker.defect_set_maker import DefectSetMaker
            dsm = DefectSetMaker(supercell_info, dopants=self.dopants)
            self._single_defects = list(dsm.defect_set)
            self._defect_map = {d.name: d for d in self._single_defects}
            self._defect_in = {d.name: d.charges for d in self._single_defects}
        self.enumerator = ComplexDefectEnumerator(
            self.host_graph,
            max_distance=max_distance,
            min_distance=min_distance,
            pristine_structure=supercell_info.structure,
            n_workers=n_workers,
        )

        self._entry_cache: dict[int, list[ComplexDefectEntry]] = {}
        self._tracker = PipelineTracker(".", enabled=track_pipeline)

        if verbose:
            from .log import configure_logging
            configure_logging(level=10, verbose=True)  # DEBUG

    # --- Class methods ---

    @classmethod
    def from_supercell_info(
        cls, path: str, dopants=None, max_distance=3.0, min_distance=0.3,
        charges=None, n_workers=None,
    ) -> "ComplexDefectMaker":
        from pydefect.input_maker.supercell_info import SupercellInfo
        with open(path) as f:
            data = json.load(f)
        return cls(SupercellInfo.from_dict(data), dopants, max_distance, min_distance, charges, n_workers=n_workers)

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

    @property
    def entry_cache(self) -> dict[int, list["ComplexDefectEntry"]]:
        """Cached entries by order. Populated by generate_entries()."""
        return dict(self._entry_cache)

    def _invalidate_entry_cache(self):
        if self._entry_cache:
            logger.debug("ENTRY CACHE: invalidated (%d orders cleared)", len(self._entry_cache))
            self._entry_cache.clear()

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
        self._invalidate_entry_cache()
        logger.info(
            "DOPANTS: switched to %s (%d defect types, geometry cache preserved: %d orders)",
            self.dopants, len(self._single_defects), len(self.enumerator.geometries),
        )

    # --- Geometry enumeration (public) ---

    def enumerate_geometries(
        self, N_max: int, eps: float = 0.1,
        progress_callback=None,
    ) -> dict[int, list[ComplexDefectGraph]]:
        """Enumerate geometrically unique N-node subgraphs.

        Returns {2: [G, ...], 3: [G, ...], ..., N_max: [...]}.
        Cached — repeated calls with higher N_max reuse prior results.

        Args:
            N_max: Highest order to enumerate.
            eps: Geometric equivalence tolerance (Å).
            progress_callback: Optional ``callable(current, total)`` invoked
                after each order completes.
        """
        return self.enumerator.enumerate(N_max, eps, progress_callback=progress_callback)

    # --- Geometry enumeration (default: no chemistry) ---

    def make_all_n_body(
        self, n: int = 2,
        max_distance: float | None = None,
        min_distance: float | None = None,
    ) -> list[ComplexDefectGraph]:
        """Enumerate geometrically unique N-body site configurations.

        Returns list[ComplexDefectGraph] for order n.
        As a side effect, also populates ``entry_cache`` for order n
        by running composition assignment + structure generation + dedup.
        """
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")

        max_d = max_distance if max_distance is not None else self.max_distance
        min_d = min_distance if min_distance is not None else self.min_distance

        if max_d != self.enumerator.max_distance or min_d != self.enumerator.min_distance:
            self.enumerator = ComplexDefectEnumerator(
                self.host_graph, max_distance=max_d, min_distance=min_d,
                pristine_structure=self.supercell_info.structure,
                n_workers=self._n_workers,
            )
            self.max_distance = max_d
            self.min_distance = min_d
            self._invalidate_entry_cache()

        self.enumerator.enumerate(n)
        geoms = list(self.enumerator.geometries.get(n, []))

        # Populate entry cache if not already cached for this order
        if n not in self._entry_cache and self._single_defects:
            try:
                self.generate_entries(n_or_geometries=n, deduplicate_symmetry=True)
            except Exception:
                logger.debug("Failed to populate entry cache for N=%d", n, exc_info=True)

        return geoms

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
        charges: list[int] | None = None,
    ) -> list[ComplexDefectEntry]:
        """Assign defect compositions and generate structures.

        Args:
            n_or_geometries: Either N (int) to generate all N-body entries,
                             or a list of ComplexDefectGraph to generate
                             entries for specific geometries.
            dopants: Override default dopants for this call.
            max_distance, min_distance: Override distance cutoffs.
            deduplicate_symmetry: Whether to cross-composition dedup.
            charges: Charge states for all generated entries.
                     None uses maker default (neutral only).

        Returns list[ComplexDefectEntry]. Apply physical filters (C1,
        max-dopant count) via ``Maker.filter_entries()`` after this call
        to avoid breaking parallel-vs-serial determinism — filtering at
        this layer can read lazy symmetry fields whose values depend on
        which process computed them.
        """
        if dopants is not None:
            self.set_dopants(dopants)

        max_d = max_distance if max_distance is not None else self.max_distance
        min_d = min_distance if min_distance is not None else self.min_distance

        need_new = (
            max_d != self.enumerator.max_distance
            or min_d != self.enumerator.min_distance
        )
        if need_new:
            self.enumerator = ComplexDefectEnumerator(
                self.host_graph, max_distance=max_d, min_distance=min_d,
                pristine_structure=self.supercell_info.structure,
                n_workers=self._n_workers,
            )
            self.max_distance = max_d
            self.min_distance = min_d
            self._invalidate_entry_cache()

        # Entry cache hit for integer N
        if isinstance(n_or_geometries, int) and n_or_geometries in self._entry_cache:
            logger.debug(
                "ENTRY CACHE HIT: returning %d entries for N=%d",
                len(self._entry_cache[n_or_geometries]), n_or_geometries,
            )
            cached = self._entry_cache[n_or_geometries]
            # Re-stamp charges if the caller supplied a non-None override.
            # The cache stores entries with the charges that were active at
            # enumeration time; a later `generate_entries(charges=...)` call
            # would otherwise be silently ignored. We shallow-copy the
            # complex_defect so the mutation doesn't leak back into the cache.
            if charges is None or list(charges) == list(self._charges):
                return list(cached)
            from copy import copy as _copy
            out = []
            for e in cached:
                new_e = _copy(e)
                new_cd = _copy(e.complex_defect)
                new_cd.charges = list(charges)
                new_e.complex_defect = new_cd
                out.append(new_e)
            return out

        if isinstance(n_or_geometries, int):
            n = n_or_geometries
            self.enumerator.enumerate(n)
            geometries = self.enumerator.geometries.get(n, [])
        else:
            geometries = n_or_geometries
            n = max(g.n_defects for g in geometries) if geometries else 2
            self.enumerator.enumerate(n)

        _charges = charges if charges is not None else self._charges
        # Dedup is kept in Maker (not generate_all_entries) because the
        # parallel-vs-serial determinism test relies on the maker's
        # tracker for ordering. The `deduplicate` kwarg in
        # generate_all_entries is unused here.
        entries = generate_all_entries(
            self.enumerator, self.supercell_info,
            self._single_defects, N_max=n,
            charges=_charges,
            deduplicate=False,  # we do it here, with tracker
        )

        logger.info("Total entries before dedup: %d", len(entries))

        if deduplicate_symmetry and entries:
            entries = deduplicate(
                entries, self.host_graph, max_d, tracker=self._tracker,
            )
            logger.info("After dedup: %d entries", len(entries))
            if self._tracker.enabled:
                report = verify_dedup(entries, self.host_graph)
                self._tracker.write_dedup_verification(report)

        # NOTE: physical filters (C1, max-dopant) are NOT applied here.
        # They read lazy spglib fields that can be non-deterministic
        # across pickling / worker processes, which breaks the
        # parallel-vs-serial entry-set equivalence test. Use
        # ``Maker.filter_entries()`` after ``generate_entries()`` for
        # physical filtering; the result is identical regardless of
        # serial/parallel generation.

        # Populate entry cache — split deduplicated entries by n_defects
        for order, order_entries in _group_by_n_defects(entries).items():
            self._entry_cache[order] = order_entries

        # Track intermediate results
        if self._tracker.enabled:
            self._tracker.write_entries_metadata(entries)

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
        charges: list[int] | None = None,
    ) -> list[ComplexDefectEntry]:
        """Generate entries for a specific N-defect complex."""
        if len(defect_names) < 2:
            raise ValueError("Need at least 2 defect names")

        defects = [self._defect_map[n] for n in defect_names]
        cd = ComplexDefect.from_defects(defects)
        _charges = charges if charges is not None else self._charges

        max_d = max_distance if max_distance is not None else self.max_distance
        min_d = min_distance if min_distance is not None else self.min_distance

        if max_d != self.enumerator.max_distance or min_d != self.enumerator.min_distance:
            self.enumerator = ComplexDefectEnumerator(
                self.host_graph, max_distance=max_d, min_distance=min_d,
                pristine_structure=self.supercell_info.structure,
                n_workers=self._n_workers,
            )

        all_entries = generate_all_entries(
            self.enumerator, self.supercell_info,
            self._single_defects, N_max=cd.n_defects,
            charges=_charges,
        )
        entries = [e for e in all_entries if e.complex_defect.name == cd.name]
        if entries:
            entries = deduplicate(entries, self.host_graph, max_d)
        return entries

    def make_pair(self, d1, d2, max_distance=None, min_distance=None):
        return self.make_complex([d1, d2], max_distance, min_distance)

    @staticmethod
    def filter_entries(
        entries: list[ComplexDefectEntry],
        exclude_point_groups: tuple[str, ...] = ("C1",),
        max_dopant_atoms: int | None = 2,
    ) -> list[ComplexDefectEntry]:
        """Apply the standard physical filters to a list of entries.

        Excludes entries whose point group is in ``exclude_point_groups``
        (default: drop the trivial C1 group) and entries with more than
        ``max_dopant_atoms`` dopant components (default: 2).

        Deliberately a *static* method, not a step in ``generate_entries``:
        reading ``entry.point_group`` triggers a lazy spglib call whose
        output can be non-deterministic across worker processes (and
        thus break the parallel-vs-serial determinism test). Callers
        who want this filter should invoke it once, in the main process,
        after all entries are collected:

            entries = maker.generate_entries(n_or_geometries=N)
            entries = ComplexDefectMaker.filter_entries(entries)

        Args:
            entries: List of ComplexDefectEntry to filter.
            exclude_point_groups: Point groups to exclude. Empty tuple
                disables this filter.
            max_dopant_atoms: Max dopant atoms per entry. None disables.

        Returns:
            Filtered list (preserves input order).
        """
        before = len(entries)
        if exclude_point_groups:
            entries = [e for e in entries
                       if not e.point_group or e.point_group not in exclude_point_groups]
        if max_dopant_atoms is not None:
            entries = [
                e for e in entries
                if sum(1 for a in e.complex_defect.in_elements if a) <= max_dopant_atoms
            ]
        if (before := before) and len(entries) != before:
            logger.info(
                "FILTER: %d -> %d entries (C1=%s, max_dopant_atoms=%s)",
                before, len(entries), exclude_point_groups, max_dopant_atoms,
            )
        return entries

    def show_geometries(self, N_max: int = 2):
        """Log a human-readable geometry summary (no chemistry)."""
        geo = self.enumerate_geometries(N_max)
        lines = []
        for n in sorted(geo):
            geoms = geo[n]
            lines.append(f"\n=== N={n} geometries ({len(geoms)} total) ===")
            for i, g in enumerate(geoms):
                dists = sorted([float(np.linalg.norm(v)) for _, _, v in g.edges])
                ds = ", ".join(f"{d:.2f}" for d in dists)
                lines.append(
                    f"  G{i:3d}: edges={len(g.edges)} dists=[{ds}] "
                    f"n_orient={g.n_orientations} pg={g.point_group} "
                    f"wyckoffs={g.wyckoffs}"
                )
                if i >= 19 and len(geoms) > 20:
                    lines.append(f"  ... {len(geoms) - 20} more")
                    break
        logger.info("GEOMETRIES:\n%s", "\n".join(lines))

    def save_geometry_cache(self, output_dir: str = "."):
        """Write geometry cache to YAML files flat in output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for n, geoms in self.enumerator.geometries.items():
            path = out / f"geometries_N{n}.yaml"
            data = {
                "max_distance": self.enumerator.max_distance,
                "min_distance": self.enumerator.min_distance,
                "order": n,
                "n_geometries": len(geoms),
                "geometries": [g.to_dict() for g in geoms],
            }
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=None, sort_keys=False)
            logger.info("CACHE: saved %d geometries to %s", len(geoms), path)

    def load_geometry_cache(self, cache_dir: str = "defect") -> set[int]:
        """Load cached geometries from flat YAML files in cache_dir.

        Reads ``geometries_N*.yaml`` from *cache_dir* (typically ``defect/``).
        Skips files whose max_distance or min_distance differ from current settings.

        Returns:
            Set of orders loaded (e.g. ``{2}`` or ``{2, 3}``).
        """
        out = Path(cache_dir)
        if not out.exists():
            return set()

        loaded: set[int] = set()
        for path in sorted(out.glob("geometries_N*.yaml")):
            data = yaml.safe_load(path.read_text())
            if (data["max_distance"] == self.enumerator.max_distance
                    and data["min_distance"] == self.enumerator.min_distance):
                geoms = [ComplexDefectGraph.from_dict(g) for g in data["geometries"]]
                n = data["order"]
                self.enumerator._cache[n] = geoms
                loaded.add(n)
                logger.info(
                    "CACHE HIT: N=%d (%d geometries) from %s "
                    "(max_distance=%.1f, min_distance=%.2f)",
                    n, len(geoms), path, data["max_distance"], data["min_distance"],
                )
            else:
                logger.info(
                    "CACHE SKIP: %s (max_distance=%.1f/%.1f or min_distance=%.2f/%.2f mismatch)",
                    path, data["max_distance"], self.enumerator.max_distance,
                    data["min_distance"], self.enumerator.min_distance,
                )
        return loaded

    @staticmethod
    def load_geometries(path: str) -> list[ComplexDefectGraph]:
        """Load geometries from a cache JSON file."""
        return ComplexDefectGraph.load_json(path)

    # --- Output ---

    def write(self, entries, output_dir=".", merge=False) -> str:
        # Absolute path: warn if relative (catches the cwd-confusion bug)
        out_path = Path(output_dir)
        if not out_path.is_absolute():
            logger.warning(
                "Maker.write: output_dir=%r is relative; resolving to %s. "
                "Pass an absolute path to avoid cwd-dependence.",
                output_dir, out_path.resolve(),
            )
            out_path = out_path.resolve()
        out_str = str(out_path)
        complex_defect_in = _write_all(entries, out_str, create_defect_json=True)
        yaml_path = write_complex_defect_in_yaml(complex_defect_in, out_str)
        summary_path = write_summary(entries, out_str)
        logger.info("Summary written to %s", summary_path)
        self._write_parameters(out_str)
        logger.info("Parameters written to %s/parameters.yaml", output_dir)
        if merge:
            merge_defect_in(out_str)
        return yaml_path

    def _write_parameters(self, output_dir: str):
        """Write parameters.yaml with full run metadata."""
        import datetime
        try:
            from importlib.metadata import version as pkg_version
            ver = pkg_version("pydefect-complex")
        except Exception:
            ver = "0.1.0-dev"

        params = {
            "pydefect_complex_version": ver,
            "timestamp": datetime.datetime.now().isoformat(),
            "parameters": {
                "max_distance_angstrom": self.max_distance,
                "min_distance_angstrom": self.min_distance,
                "dopants": self.dopants,
                "charges": self._charges,
                "defect_names": self.defect_names,
            },
            "enumerator": {
                "n_geometries_cached": {
                    str(k): len(v) for k, v in self.enumerator.geometries.items()
                },
            },
            "entry_cache": {
                "orders_cached": sorted(self._entry_cache.keys()),
                "n_entries": {
                    str(k): len(v) for k, v in self._entry_cache.items()
                },
            },
        }
        path = Path(output_dir) / "parameters.yaml"
        with open(path, "w") as f:
            yaml.dump(params, f, default_flow_style=None, sort_keys=False)

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