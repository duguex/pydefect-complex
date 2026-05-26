"""Complex defect structure entry and symmetry analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import IStructure

from .core import ComplexDefect


# ---------------------------------------------------------------------------
# Space group → Schoenflies point group mapping
# ---------------------------------------------------------------------------

_SG_TO_SCHOENFLIES = {
    1: "C1", 2: "Ci",
    3: "C2", 4: "C2", 5: "C2",
    6: "Cs", 7: "Cs", 8: "Cs", 9: "Cs",
    10: "C2h", 11: "C2h", 12: "C2h", 13: "C2h", 14: "C2h", 15: "C2h",
    16: "D2", 17: "D2", 18: "D2", 19: "D2", 20: "D2", 21: "D2",
    22: "D2", 23: "D2", 24: "D2",
    25: "C2v", 26: "C2v", 27: "C2v", 28: "C2v", 29: "C2v",
    30: "C2v", 31: "C2v", 32: "C2v", 33: "C2v", 34: "C2v",
    35: "C2v", 36: "C2v", 37: "C2v", 38: "C2v", 39: "C2v",
    40: "C2v", 41: "C2v", 42: "C2v", 43: "C2v", 44: "C2v", 45: "C2v", 46: "C2v",
    47: "D2h",
    75: "C4", 76: "C4", 77: "C4", 78: "C4", 79: "C4", 80: "C4",
    81: "S4", 82: "S4",
    83: "C4h", 84: "C4h", 85: "C4h", 86: "C4h", 87: "C4h", 88: "C4h",
    89: "D4", 90: "D4", 91: "D4", 92: "D4", 93: "D4",
    94: "D4", 95: "D4", 96: "D4", 97: "D4", 98: "D4",
    99: "C4v", 100: "C4v", 101: "C4v", 102: "C4v", 103: "C4v",
    104: "C4v", 105: "C4v", 106: "C4v", 107: "C4v", 108: "C4v", 109: "C4v", 110: "C4v",
    111: "D2d", 112: "D2d", 113: "D2d", 114: "D2d",
    115: "D2d", 116: "D2d", 117: "D2d", 118: "D2d",
    119: "D2d", 120: "D2d", 121: "D2d", 122: "D2d",
    123: "D4h", 124: "D4h", 125: "D4h", 126: "D4h",
    127: "D4h", 128: "D4h", 129: "D4h", 130: "D4h",
    131: "D4h", 132: "D4h", 133: "D4h", 134: "D4h",
    135: "D4h", 136: "D4h", 137: "D4h", 138: "D4h",
    139: "D4h", 140: "D4h", 141: "D4h", 142: "D4h",
    143: "C3", 144: "C3", 145: "C3", 146: "C3",
    147: "S6", 148: "S6",
    149: "D3", 150: "D3", 151: "D3", 152: "D3", 153: "D3", 154: "D3", 155: "D3",
    156: "C3v", 157: "C3v", 158: "C3v", 159: "C3v", 160: "C3v", 161: "C3v",
    162: "D3d", 163: "D3d", 164: "D3d", 165: "D3d", 166: "D3d", 167: "D3d",
    168: "C6", 169: "C6", 170: "C6", 171: "C6", 172: "C6", 173: "C6",
    174: "C3h",
    175: "C6h", 176: "C6h",
    177: "D6", 178: "D6", 179: "D6", 180: "D6", 181: "D6", 182: "D6",
    183: "C6v", 184: "C6v", 185: "C6v", 186: "C6v",
    187: "D3h", 188: "D3h", 189: "D3h", 190: "D3h",
    191: "D6h", 192: "D6h", 193: "D6h", 194: "D6h",
    195: "T", 196: "T", 197: "T", 198: "T", 199: "T",
    200: "Th", 201: "Th", 202: "Th", 203: "Th", 204: "Th", 205: "Th", 206: "Th",
    207: "O", 208: "O", 209: "O", 210: "O", 211: "O", 212: "O", 213: "O", 214: "O",
    215: "Td", 216: "Td", 217: "Td", 218: "Td", 219: "Td", 220: "Td",
    221: "Oh", 222: "Oh", 223: "Oh", 224: "Oh",
    225: "Oh", 226: "Oh", 227: "Oh", 228: "Oh",
    229: "Oh", 230: "Oh",
}


_SG_TO_HM = {
    1: "P1", 2: "P-1",
    3: "P2", 4: "P2_1", 5: "C2",
    6: "Pm", 7: "Pc", 8: "Cm", 9: "Cc",
    10: "P2/m", 11: "P2_1/m", 12: "C2/m", 13: "P2/c", 14: "P2_1/c", 15: "C2/c",
    16: "P222", 17: "P222_1", 18: "P2_12_12", 19: "P2_12_12_1", 20: "C222_1", 21: "C222",
    22: "F222", 23: "I222", 24: "I2_12_12_1",
    25: "Pmm2", 26: "Pmc2_1", 27: "Pcc2", 28: "Pma2", 29: "Pca2_1",
    30: "Pnc2", 31: "Pmn2_1", 32: "Pba2", 33: "Pna2_1", 34: "Pnn2",
    35: "Cmm2", 36: "Cmc2_1", 37: "Ccc2", 38: "Amm2", 39: "Abm2",
    40: "Ama2", 41: "Aba2", 42: "Fmm2", 43: "Fdd2", 44: "Imm2", 45: "Iba2", 46: "Ima2",
    47: "Pmmm", 48: "Pnnn",
    75: "P4", 76: "P4_1", 77: "P4_2", 78: "P4_3", 79: "I4", 80: "I4_1",
    81: "P-4", 82: "I-4",
    83: "P4/m", 84: "P4_2/m", 85: "P4/n", 86: "P4_2/n", 87: "I4/m", 88: "I4_1/a",
    89: "P422", 90: "P42_12", 91: "P4_122", 92: "P4_12_12", 93: "P4_222",
    94: "P4_22_12", 95: "P4_322", 96: "P4_32_12", 97: "I422", 98: "I4_122",
    99: "P4mm", 100: "P4bm", 101: "P4_2cm", 102: "P4_2nm", 103: "P4cc",
    104: "P4nc", 105: "P4_2mc", 106: "P4_2bc",
    107: "I4mm", 108: "I4cm", 109: "I4_1md", 110: "I4_1cd",
    111: "P-42m", 112: "P-4c2", 113: "P-42_1m", 114: "P-42_1c",
    115: "P-4m2", 116: "P-4c2", 117: "P-4b2", 118: "P-4n2",
    119: "I-4m2", 120: "I-4c2", 121: "I-42m", 122: "I-42d",
    123: "P4/mmm", 124: "P4/mcc", 125: "P4/nbm", 126: "P4/nnc",
    127: "P4/mbm", 128: "P4/mnc", 129: "P4/nmm", 130: "P4/ncc",
    131: "P4_2/mmc", 132: "P4_2/mcm", 133: "P4_2/nbc", 134: "P4_2/nnm",
    135: "P4_2/nbc", 136: "P4_2/mnm", 137: "P4_2/nmc", 138: "P4_2/ncm",
    139: "I4/mmm", 140: "I4/mcm", 141: "I4_1/amd", 142: "I4_1/acd",
    143: "P3", 144: "P3_1", 145: "P3_2", 146: "R3",
    147: "P-3", 148: "R-3",
    149: "P312", 150: "P321", 151: "P3_112", 152: "P3_121",
    153: "P3_212", 154: "P3_221", 155: "R32",
    156: "P3m1", 157: "P31m", 158: "P3c1", 159: "P31c",
    160: "R3m", 161: "R3c",
    162: "P-3m1", 163: "P-31m", 164: "P-3m1", 165: "P-31c",
    166: "R-3m", 167: "R-3c",
    168: "P6", 169: "P6_1", 170: "P6_5", 171: "P6_2", 172: "P6_4", 173: "P6_3",
    174: "P-6",
    175: "P6/m", 176: "P6_3/m",
    177: "P622", 178: "P6_122", 179: "P6_522", 180: "P6_222",
    181: "P6_422", 182: "P6_322",
    183: "P6mm", 184: "P6cc", 185: "P6_3cm", 186: "P6_3mc",
    187: "P-6m2", 188: "P-6c2", 189: "P-62m", 190: "P-62c",
    191: "P6/mmm", 192: "P6/mcc", 193: "P6_3/mcm", 194: "P6_3/mmc",
    195: "P23", 196: "F23", 197: "I23", 198: "P2_13", 199: "I2_13",
    200: "Pm-3", 201: "Pn-3", 202: "Fm-3", 203: "Fd-3", 204: "Im-3", 205: "Pa-3", 206: "Ia-3",
    207: "P432", 208: "P4_232", 209: "F432", 210: "F4_132", 211: "I432",
    212: "P4_332", 213: "P4_132", 214: "I4_132",
    215: "P-43m", 216: "F-43m", 217: "I-43m",
    218: "P-43n", 219: "F-43c", 220: "I-43d",
    221: "Pm-3m", 222: "Pn-3n", 223: "Pm-3n", 224: "Pn-3m",
    225: "Fm-3m", 226: "Fm-3c", 227: "Fd-3m", 228: "Fd-3c",
    229: "Im-3m", 230: "Ia-3d",
}


# ---------------------------------------------------------------------------
# Orientation counting
# ---------------------------------------------------------------------------


def _classify_point_group(
    rotations: "list[np.ndarray]", lattice: "np.ndarray",
) -> str:
    """Classify integer rotation matrices into a Schoenflies symbol.

    Generates a set of test vectors by applying *rotations* to a
    reference direction, converts to cartesian, then uses pymatgen's
    ``PointGroupAnalyzer`` for the symbol.

    Args:
        rotations: Integer 3x3 rotation matrices (spglib format) that
                   form the stabilizer of a defect geometry.
        lattice: 3x3 lattice matrix (cartesian conversion).

    Returns:
        Schoenflies symbol (e.g. ``"C2h"``, ``"D3d"``, ``"C1"``).
    """
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    if not rotations:
        return "C1"
    unique = {tuple(r.flatten()) for r in rotations}
    if len(unique) == 1:
        return "C1"

    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    points = set()
    for r_mat in rotations:
        v = lattice @ (r_mat @ ref)
        points.add(tuple(round(c, 8) for c in v))

    if len(points) == 0:
        return "C1"
    pts = np.array(list(points))
    mol = Molecule(["H"] * len(pts), pts)
    try:
        return str(PointGroupAnalyzer(mol).get_pointgroup())
    except Exception:
        return "C1"


def _count_orientations_from_coords(
    frac_coords: "list[tuple[float,...]]",
    pristine_structure: "IStructure",
    sym_ops: "tuple | None" = None,
    tol: float = 0.2,
) -> int:
    """Count orientations of a geometric cluster (no chemistry needed).

    Pure geometry: applies space-group ops, maps to host atoms, groups
    by translational equivalence.  Works on raw frac_coords — usable
    before defect chemistry assignment.

    Args:
        frac_coords: Defect site fractional coordinates.
        pristine_structure: Perfect supercell structure.
        sym_ops: Optional pre-computed (rotations, translations) from
                 spglib.get_symmetry, to avoid repeated spglib calls.
    """
    from scipy.spatial import KDTree
    import spglib

    if sym_ops is None:
        cell = (pristine_structure.lattice.matrix,
                pristine_structure.frac_coords,
                pristine_structure.atomic_numbers)
        sym = spglib.get_symmetry(cell, symprec=0.01)
        symmetries = (sym['rotations'], sym['translations'])
    else:
        symmetries = sym_ops

    pos = pristine_structure.frac_coords
    lattice = pristine_structure.lattice.matrix
    tree = KDTree(pos)

    dc = np.array(frac_coords, dtype=float)
    n = dc.shape[0]
    if n < 2:
        return 1 if n == 1 else 0

    r_to_ts: dict[tuple, list] = {}
    for R_mat, t in zip(symmetries[0], symmetries[1]):
        key = tuple(R_mat.flatten())
        r_to_ts.setdefault(key, []).append(np.array(t))

    stabilizer_rots = []  # R matrices that map geometry to itself

    orig_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            df = dc[i] - dc[j]; df -= np.round(df)
            orig_dists.append(float(np.linalg.norm(lattice @ df)))
    orig_dists.sort()

    # Canonical form: deterministic ordering via lexicographic sort.
    # O(n log n) vs. brute-force O(n·n!) permutation search.
    _canon_cache: dict[tuple, tuple] = {}

    def _canonical(points, ids_key=None):
        if ids_key is not None and ids_key in _canon_cache:
            return _canon_cache[ids_key]
        arr = np.array(points)
        order = np.lexsort(arr.T)
        sorted_pts = arr[order]
        anchor = sorted_pts[0]
        parts = []
        for k in range(1, len(sorted_pts)):
            rel = tuple(sorted_pts[k][j] - anchor[j] for j in range(3))
            parts.append(tuple(round(x - round(x), 8) for x in rel))
        result = tuple(parts)
        if ids_key is not None:
            _canon_cache[ids_key] = result
        return result

    best_orig = _canonical(dc)

    orient_sets: list[tuple] = []
    stabilizer_rots = []

    for r_key, t_list in r_to_ts.items():
        R = np.array(r_key, dtype=int).reshape(3, 3)
        for t in t_list:
            rotated = (dc @ R.T + t) % 1.0
            dists, idxs = tree.query(rotated)
            wrapped = (rotated - tree.data[idxs] + 0.5) % 1.0 - 0.5
            if np.any(np.linalg.norm(wrapped, axis=1) > tol):
                continue
            ids = list(idxs)

            mapped = []
            for i in range(n):
                for j in range(i + 1, n):
                    df = pos[ids[i]] - pos[ids[j]]; df -= np.round(df)
                    mapped.append(round(float(np.linalg.norm(lattice @ df)), 3))
            mapped.sort()
            if not all(abs(a - b) < tol for a, b in zip(orig_dists, mapped)):
                continue

            best = _canonical([pos[i] for i in ids], ids_key=tuple(sorted(ids)))
            if best == best_orig:
                stabilizer_rots.append(R)
            if best not in orient_sets:
                orient_sets.append(best)
            break

    pg = _classify_point_group(stabilizer_rots, lattice)
    return len(orient_sets), pg


def count_defect_orientations(
    entry: "ComplexDefectEntry",
    pristine_structure: "IStructure",
    tol: float = 0.2,
) -> int:
    """Thin wrapper — delegates to _count_orientations_from_coords."""
    n_orient, _ = _count_orientations_from_coords(
        list(entry.defect_coords), pristine_structure, tol=tol)
    return n_orient
@dataclass
class ComplexDefectEntry:
    """A generated complex defect structure with metadata.

    Attributes:
        name: Full defect name (set after dedup, format: "{compact_name}.{index}").
        complex_defect: The ComplexDefect that produced this entry.
        site_path: Tuple of site names for each defect layer.
        distances: Tuple of distances between successive defect centers (Å).
        structure: Final IStructure of the defect supercell.
        defect_coords: N defect center fractional coords, sorted by wyckoff
            (matches ordering of site_path and complex_defect.defects).
        graph: ComplexDefectGraph (set after generation).
    """

    name: str
    complex_defect: ComplexDefect
    site_path: tuple[str, ...]
    distances: tuple[float, ...]
    structure: "IStructure"
    defect_coords: tuple = ()
    graph: object = None

    _point_group: str = ""
    _space_group_number: int = 0
    _space_group_symbol: str = ""
    _n_sym_ops: int = 0
    _n_orientations: int = -1

    pristine_structure_cache: object = None

    @property
    def distance(self) -> float:
        return self.distances[0] if self.distances else 0.0

    def _ensure_symmetry(self, symprec: float = 0.01):
        if self._space_group_number > 0:
            return
        try:
            import spglib
            cell = (self.structure.lattice.matrix,
                    self.structure.frac_coords,
                    self.structure.atomic_numbers)
            ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
            if ds:
                self._space_group_number = ds.number
                self._space_group_symbol = _SG_TO_HM.get(ds.number, ds.international)
                self._point_group = _SG_TO_SCHOENFLIES.get(ds.number, "?")
                self._n_sym_ops = len(ds.rotations)
        except ImportError:
            pass

    @property
    def point_group(self) -> str:
        self._ensure_symmetry()
        return self._point_group

    @property
    def space_group(self) -> str:
        self._ensure_symmetry()
        return self._space_group_symbol

    @property
    def space_group_number(self) -> int:
        self._ensure_symmetry()
        return self._space_group_number

    @property
    def n_sym_ops(self) -> int:
        self._ensure_symmetry()
        return self._n_sym_ops

    @property
    def n_orientations(self) -> int:
        if self._n_orientations >= 0:
            return self._n_orientations
        if self.graph is not None and self.graph.n_orientations >= 0:
            self._n_orientations = self.graph.n_orientations
            return self._n_orientations
        if self.pristine_structure_cache is not None:
            self._n_orientations = count_defect_orientations(
                self, self.pristine_structure_cache)
            return self._n_orientations
        return -1