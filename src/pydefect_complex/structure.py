"""Structure generation engine for complex defects.

Core algorithm: recursive layer-by-layer defect application with
symmetry-aware site enumeration and distance filtering.

  1. Apply defect_1 to pristine supercell
  2. Symmetrize the defected structure
  3. Enumerate symmetry-inequivalent sites for defect_2
  4. Filter by element match + distance threshold
  5. Apply defect_2, recurse for defect_3...defect_N
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from pymatgen.core import IStructure
    from pydefect.input_maker.supercell_info import SupercellInfo
    from pydefect.input_maker.defect import SimpleDefect
    from vise.util.structure_symmetrizer import StructureSymmetrizer

from .core import ComplexDefect, _get_element, _is_interstitial


@dataclass
class ComplexDefectEntry:
    """A generated complex defect structure with metadata.

    Attributes:
        name: Full defect name (set after dedup, format: "{compact_name}.{index}").
        complex_defect: The ComplexDefect that produced this entry.
        site_path: Tuple of site names for each defect layer.
        distances: Tuple of distances between successive defect centers (Å).
        structure: Final IStructure of the defect supercell.
        defect_coords: N defect center fractional coords, in defect application order.
        graph: ComplexDefectGraph (set after generation).
    """

    name: str
    complex_defect: ComplexDefect
    site_path: tuple[str, ...]
    distances: tuple[float, ...]
    structure: "IStructure"
    defect_coords: tuple = ()  # tuple of (x,y,z) fractional coords, application order
    graph: object = None  # ComplexDefectGraph, set after generation

    @property
    def distance(self) -> float:
        """Shortcut for 2-body distance."""
        return self.distances[0] if self.distances else 0.0


# ---------------------------------------------------------------------------
# Structure operations
# ---------------------------------------------------------------------------

def _apply_single_defect(
    supercell_info: "SupercellInfo",
    defect: "SimpleDefect",
) -> tuple["IStructure", np.ndarray]:
    """Apply a single defect to the pristine supercell.

    Returns:
        (defect_structure, defect_center_frac_coords)
    """
    from pydefect.input_maker.defect_entries_maker import (
        copy_to_structure,
        to_istructure,
        add_atom_to_structure,
    )

    structure = copy_to_structure(supercell_info.structure)

    if _is_interstitial(defect.out_atom):
        index = int(defect.out_atom[1:]) - 1
        coords = supercell_info.interstitials[index].frac_coords.copy()
    else:
        site = supercell_info.sites[defect.out_atom]
        index = site.equivalent_atoms[0]
        coords = structure.pop(index).frac_coords

    if defect.in_atom is not None:
        add_atom_to_structure(structure, defect.in_atom, coords)

    return to_istructure(structure), coords


def _apply_defect_to_defected(
    symmetrizer: "StructureSymmetrizer",
    in_atom: Optional[str],
    out_atom_site: str,
) -> tuple["IStructure", np.ndarray]:
    """Apply a defect to an already-defected structure."""
    from pydefect.input_maker.defect_entries_maker import (
        copy_to_structure,
        to_istructure,
        add_atom_to_structure,
    )

    structure = copy_to_structure(symmetrizer.structure)
    site = symmetrizer.sites[out_atom_site]
    index = site.equivalent_atoms[0]
    coords = structure.pop(index).frac_coords

    if in_atom is not None:
        add_atom_to_structure(structure, in_atom, coords)

    return to_istructure(structure), coords


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def _frac_min_image_distance(
    a: np.ndarray,
    b: np.ndarray,
    lattice_matrix: np.ndarray,
) -> float:
    """Minimum-image distance (Å) between two fractional coordinates."""
    diff = a - b
    diff -= np.round(diff)
    return float(np.linalg.norm(diff @ lattice_matrix))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_entries(
    supercell_info: "SupercellInfo",
    complex_defect: ComplexDefect,
    max_distance: float = 5.0,
    min_distance: float = 0.3,
) -> list[ComplexDefectEntry]:
    """Generate all valid complex defect structures for N >= 2.

    Uses recursive layer-by-layer defect application. The first defect
    is applied to the pristine supercell; each subsequent defect is
    applied to the already-defected structure, with sites enumerated
    via symmetry analysis of the current structure.

    Distance constraints:
      - Each new defect center must be within max_distance of the
        PREVIOUS defect center (chain constraint, not all-pairs).
      - Each new center must be at least min_distance from ALL
        previous centers (no overlap).

    Args:
        supercell_info: pydefect SupercellInfo for the pristine host.
        complex_defect: N-component ComplexDefect (N >= 2).
        max_distance: Max distance between consecutive defect centers.
        min_distance: Min distance between any two defect centers.

    Returns:
        List of ComplexDefectEntry objects, one per valid site placement.
    """
    n = complex_defect.n_defects

    if n < 2:
        raise ValueError(
            f"ComplexDefect must have at least 2 components, got {n}"
        )

    # --- Pre-filter invalid combinations ---

    defects = complex_defect.defects

    # Only the first defect can be an interstitial
    for i in range(1, n):
        if _is_interstitial(defects[i].out_atom):
            return []

    # Interstitial + element cycling check (first layer only)
    if _is_interstitial(defects[0].out_atom):
        if complex_defect.out_elements[1] == complex_defect.in_elements[0]:
            return []

    # --- Start recursion ---

    supercell = supercell_info.structure
    lattice = supercell.lattice.matrix
    tree_supercell = KDTree(supercell.frac_coords)
    no_defect_sym = _get_symmetrizer(supercell)

    # Apply defect 0 to pristine
    struct_0, coords_0 = _apply_single_defect(supercell_info, defects[0])

    entries = []
    _recurse_defect_layer(
        supercell_info,
        defects=defects,
        defect_index=1,
        tree_supercell=tree_supercell,
        no_defect_sym=no_defect_sym,
        lattice=lattice,
        max_distance=max_distance,
        min_distance=min_distance,
        prev_structure=struct_0,
        prev_coords=[coords_0],
        prev_site_path=[defects[0].out_atom],
        prev_distances=[],
        output_list=entries,
    )

    return entries


# ---------------------------------------------------------------------------
# Recursive engine
# ---------------------------------------------------------------------------

def _get_symmetrizer(structure: "IStructure") -> "StructureSymmetrizer":
    from vise.util.structure_symmetrizer import StructureSymmetrizer
    return StructureSymmetrizer(structure)


def _recurse_defect_layer(
    supercell_info: "SupercellInfo",
    defects: list,
    defect_index: int,
    tree_supercell: KDTree,
    no_defect_sym: "StructureSymmetrizer",
    lattice: np.ndarray,
    max_distance: float,
    min_distance: float,
    prev_structure: "IStructure",
    prev_coords: list[np.ndarray],
    prev_site_path: list[str],
    prev_distances: list[float],
    output_list: list[ComplexDefectEntry],
):
    """Recursive defect application. Applies defects[defect_index]
    to prev_structure, then recurses for remaining defects.

    Base case: defect_index >= len(defects) → build entry and append.
    """
    n = len(defects)

    if defect_index >= n:
        # All defects applied — build final entry
        comp_name = ComplexDefect.from_defects(defects).name
        output_list.append(
            ComplexDefectEntry(
                name=comp_name,
                complex_defect=ComplexDefect.from_defects(defects),
                site_path=tuple(prev_site_path),
                distances=tuple(prev_distances),
                structure=prev_structure,
                defect_coords=tuple(
                    tuple(float(x) for x in c) for c in prev_coords
                ),
            )
        )
        return

    # Symmetrize the current defected structure
    sym = _get_symmetrizer(prev_structure)

    d = defects[defect_index]
    target_element = _get_element(d.out_atom)

    for site_name in sym.sites:
        # Element match
        if _get_element(site_name) != target_element:
            continue

        # Get site coordinates
        idx = sym.sites[site_name].equivalent_atoms[0]
        site_coords = prev_structure[idx].frac_coords

        # Skip if too close to any previous defect center (overlap check)
        too_close = False
        for pc in prev_coords:
            if _frac_min_image_distance(site_coords, pc, lattice) < min_distance:
                too_close = True
                break
        if too_close:
            continue

        # Map back to pristine supercell labels via KDTree
        kd_dist, tree_idx = tree_supercell.query(site_coords)
        if kd_dist > 1e-2:
            continue

        # Find the pristine site name
        pristine_site = None
        for _site in no_defect_sym.sites:
            if tree_idx in no_defect_sym.sites[_site].equivalent_atoms:
                pristine_site = _site
                break
        if pristine_site is None:
            continue

        # Must match the defect's expected out_atom label
        if pristine_site != d.out_atom:
            continue

        # Compute distance to PREVIOUS defect center (chain constraint)
        dist = _frac_min_image_distance(
            site_coords, prev_coords[-1], lattice
        )
        if not (0 < dist <= max_distance):
            continue

        # Apply this defect
        new_structure, new_coords = _apply_defect_to_defected(
            sym, d.in_atom, site_name
        )

        # Recurse
        _recurse_defect_layer(
            supercell_info,
            defects,
            defect_index + 1,
            tree_supercell,
            no_defect_sym,
            lattice,
            max_distance,
            min_distance,
            new_structure,
            prev_coords + [new_coords],
            prev_site_path + [site_name],
            prev_distances + [dist],
            output_list,
        )