"""Complex defect graph representation.

HostGraph: the host crystal — a site registry with coordinate lookup.
ComplexDefectGraph: an N-node labeled graph with edges within max_distance.
  Nodes are labeled by host site identity: (wyckoff, element).
  Defect types are assigned AFTER geometry dedup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from pydefect.input_maker.supercell_info import SupercellInfo
    from .structure import ComplexDefectEntry
    from .core import ComplexDefect


# ---------------------------------------------------------------------------
# Host graph: crystal site registry
# ---------------------------------------------------------------------------


@dataclass
class HostNode:
    """A site in the host crystal."""

    id: int
    wyckoff: str
    element: str
    frac_coord: np.ndarray


@dataclass
class HostGraph:
    """Crystal site registry with min-image distance utilities."""

    nodes: list[HostNode]
    lattice: np.ndarray
    _kdtree: Optional[KDTree] = None

    def __post_init__(self):
        self.lattice = np.asarray(self.lattice, dtype=float)
        self._kdtree = KDTree([n.frac_coord for n in self.nodes])

    @classmethod
    def from_supercell_info(cls, supercell_info: "SupercellInfo") -> "HostGraph":
        sc = supercell_info.structure
        sites = supercell_info.sites
        lattice = sc.lattice.matrix
        n_atoms = len(sc)

        # Map each supercell atom to a wyckoff label.
        # Strategy:
        #  1. Use pydefect's site name if the atom index is in the
        #     equivalent_atoms list AND the element matches.
        #  2. Otherwise derive from element name (e.g., Si → "Si1").
        #     This handles multi-element systems where pydefect's
        #     equivalent_atoms use a non-trivial index space.
        idx_to_wyckoff: dict[int, str] = {}
        for wyckoff, site in sites.items():
            for idx in site.equivalent_atoms:
                if 0 <= idx < n_atoms:
                    idx_to_wyckoff[idx] = wyckoff

        nodes = []
        for i, site in enumerate(sc):
            w = idx_to_wyckoff.get(i)
            if w is not None:
                # Verify: does the site name's expected element match?
                # (e.g., "Si1" → should be Si)
                expected_el = w.rstrip("0123456789")
                if site.species_string != expected_el:
                    w = None
            if w is None:
                w = site.species_string + "1"
            nodes.append(HostNode(
                id=i,
                wyckoff=w,
                element=site.species_string,
                frac_coord=site.frac_coords.copy(),
            ))
        return cls(nodes=nodes, lattice=lattice)

    def find_node(self, frac_coord: np.ndarray) -> HostNode:
        d, idx = self._kdtree.query(frac_coord)
        if d > 1e-2:
            raise ValueError(f"No host node near {frac_coord} (d={d:.4f})")
        return self.nodes[int(idx)]

    def min_image_vector(self, fc_a: np.ndarray, fc_b: np.ndarray) -> np.ndarray:
        d = np.asarray(fc_a) - np.asarray(fc_b)
        d -= np.round(d)
        return np.dot(d, self.lattice)

    def min_image_distance(self, fc_a: np.ndarray, fc_b: np.ndarray) -> float:
        """Cartesian min-image distance (Å) between two fractional coords."""
        return float(np.linalg.norm(self.min_image_vector(fc_a, fc_b)))

    def neighbors(self, node_id: int, max_distance: float) -> list[int]:
        """Return all node IDs within max_distance (Å) of node_id.

        Computes exact min-image Cartesian distance (PBC-aware).
        For typical supercells (~200 atoms) brute-force is microseconds;
        no KDTree pre-filter needed at this scale.
        """
        target = self.nodes[node_id].frac_coord
        result = []
        for n in self.nodes:
            if n.id == node_id:
                continue
            if self.min_image_distance(target, n.frac_coord) <= max_distance:
                result.append(n.id)
        return result

    def neighbors_of_set(
        self, node_ids: set[int], max_distance: float,
    ) -> set[int]:
        """Return all node IDs within max_distance of ANY node in the set,
        excluding those already in the set."""
        ext = set()
        for nid in node_ids:
            for nbr in self.neighbors(nid, max_distance):
                if nbr not in node_ids:
                    ext.add(nbr)
        return ext


# ---------------------------------------------------------------------------
# Complex defect graph (geometry only, no defect types)
# ---------------------------------------------------------------------------


def _edge_list(
    coords: list[tuple[float, ...]],
    host_graph: HostGraph,
    max_distance: float,
) -> list[tuple[int, int, np.ndarray]]:
    n = len(coords)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            v = host_graph.min_image_vector(coords[i], coords[j])
            if float(np.linalg.norm(v)) <= max_distance:
                edges.append((i, j, v))
    return edges


@dataclass
class ComplexDefectGraph:
    """Geometry of a defect complex — no defect types.

    Node identity = (wyckoff, element) from the host crystal.
    Defect types are assigned later by composition matching.

    Attributes:
        host_node_ids: N host graph node IDs.
        wyckoffs: N Wyckoff labels.
        elements: N element symbols.
        edges: list of (i, j, v_ij) for pairs within max_distance.
    """

    host_node_ids: tuple[int, ...]
    wyckoffs: tuple[str, ...]
    elements: tuple[str, ...]
    edges: list[tuple[int, int, np.ndarray]] = field(default_factory=list)

    def __post_init__(self):
        self.edges = [(i, j, np.asarray(v, dtype=float)) for i, j, v in self.edges]

    @property
    def n_defects(self) -> int:
        return len(self.host_node_ids)

    @property
    def node_labels(self) -> list[tuple[str, str]]:
        """Host site identity per node: (wyckoff, element)."""
        return list(zip(self.wyckoffs, self.elements))

    @classmethod
    def from_entry(
        cls,
        entry: "ComplexDefectEntry",
        host_graph: HostGraph,
        max_distance: float,
    ) -> "ComplexDefectGraph":
        """Build geometry graph from an entry's defect coordinates."""
        if not entry.defect_coords:
            raise ValueError("entry.defect_coords is empty")

        host_node_ids = []
        wyckoffs = []
        elements = []
        for fc in entry.defect_coords:
            node = host_graph.find_node(np.array(fc))
            host_node_ids.append(node.id)
            wyckoffs.append(node.wyckoff)
            elements.append(node.element)

        edges = _edge_list(entry.defect_coords, host_graph, max_distance)

        return cls(
            host_node_ids=tuple(host_node_ids),
            wyckoffs=tuple(wyckoffs),
            elements=tuple(elements),
            edges=edges,
        )


# ---------------------------------------------------------------------------
# Graph isomorphism (geometry only)
# ---------------------------------------------------------------------------


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def equivalent(
    graph1: ComplexDefectGraph,
    graph2: ComplexDefectGraph,
    eps: float = 0.1,
) -> bool:
    """Test geometric equivalence.

    Two graphs are geometrically equivalent if there exists a node
    permutation π (respecting (wyckoff, element) groups) and a
    rotation R ∈ SO(3) such that every edge matches within ε.
    """
    if graph1.n_defects != graph2.n_defects:
        return False
    if len(graph1.edges) != len(graph2.edges):
        return False

    labels1 = list(zip(graph1.wyckoffs, graph1.elements))
    labels2 = list(zip(graph2.wyckoffs, graph2.elements))
    if sorted(labels1) != sorted(labels2):
        return False

    from collections import defaultdict
    import itertools

    n = graph1.n_defects

    lbl_to_idx2 = defaultdict(list)
    for idx, lbl in enumerate(labels2):
        lbl_to_idx2[lbl].append(idx)

    lbl_to_idx1 = defaultdict(list)
    for idx, lbl in enumerate(labels1):
        lbl_to_idx1[lbl].append(idx)

    lbl_perms = {}
    for lbl in lbl_to_idx1:
        if len(lbl_to_idx1[lbl]) != len(lbl_to_idx2[lbl]):
            return False
        lbl_perms[lbl] = list(itertools.permutations(lbl_to_idx2[lbl]))

    e2_lookup = {}
    for i, j, v in graph2.edges:
        e2_lookup[(i, j)] = v
        e2_lookup[(j, i)] = -v

    for perm_product in itertools.product(*lbl_perms.values()):
        perm_map = [0] * n
        for lbl, perm in zip(lbl_to_idx1.keys(), perm_product):
            for i1, i2 in zip(lbl_to_idx1[lbl], perm):
                perm_map[i1] = i2

        P_edges, Q_edges = [], []
        ok = True
        for i, j, v1 in graph1.edges:
            pi, pj = perm_map[i], perm_map[j]
            v2 = e2_lookup.get((pi, pj))
            if v2 is None:
                ok = False
                break
            P_edges.append(v1)
            Q_edges.append(v2)
        if not ok:
            continue

        P = np.stack(P_edges, axis=0)
        Q = np.stack(Q_edges, axis=0)
        R = _kabsch(P, Q)
        if np.max(np.linalg.norm(P @ R.T - Q, axis=1)) < eps:
            return True
    return False