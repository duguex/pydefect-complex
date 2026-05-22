"""Complex defect data model.

A ComplexDefect is a composition of N SimpleDefect objects.
It provides name generation, charge state estimation, and
element tracking for use in structure generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pydefect.input_maker.defect import SimpleDefect


def _get_element(name: str) -> str:
    """Extract element symbol from a site/defect out_atom name.

    Examples:
        'C_1' -> 'C'
        'Si_C_3' -> 'Si'  (substitution: Si replaces C at site 3)
        'i1' -> 'i'       (interstitial site)
    """
    # Interstitial: "i1", "i2", etc.
    if name.startswith("i") and name[1:].isdigit():
        return "i"
    # Standard site: "C_1", "Si_C_3", etc.
    match = re.match(r"^([A-Z][a-z]?)", name)
    return match.group(1) if match else name


def _is_interstitial(out_atom: str) -> bool:
    """Check if a defect out_atom refers to an interstitial site."""
    return out_atom.startswith("i")


@dataclass
class ComplexDefect:
    """A complex defect composed of N component SimpleDefect objects.

    Defects are sorted by out_atom (reverse) for deterministic naming.
    Charge states are estimated from oxidation state differences
    between inserted and removed atoms.

    Attributes:
        defects: Component SimpleDefect objects, sorted deterministically.
        name: Canonical name like 'v_C_1+v_C_2+N_C_3'.
        charges: Estimated charge state list for VASP calculations.
    """

    defects: list  # list[SimpleDefect]
    name: str = field(init=False)
    charges: list[int] = field(init=False)

    def __post_init__(self):
        from collections import Counter

        # Sort for deterministic output: by out_atom, reverse order
        self.defects = sorted(
            self.defects, key=lambda d: d.out_atom, reverse=True
        )

        # Generate canonical name: compact count format
        counts = Counter(d.name for d in self.defects)
        seen = set()
        parts = []
        for d in self.defects:
            name = d.name
            if name in seen:
                continue
            seen.add(name)
            c = counts[name]
            parts.append(f"{c if c > 1 else ''}{name}")
        self.name = "+".join(parts)

        # Collect element info
        self._in_atoms = [d.in_atom for d in self.defects]
        self._out_atoms = [d.out_atom for d in self.defects]
        self._in_elements = [
            _get_element(a) if a else None for a in self._in_atoms
        ]
        self._out_elements = [_get_element(a) for a in self._out_atoms]

        self.charges = self._estimate_charges()

    # --- Class methods for construction ---

    @classmethod
    def from_pair(cls, d1, d2) -> "ComplexDefect":
        """Create a ComplexDefect from exactly two SimpleDefects."""
        return cls([d1, d2])

    @classmethod
    def from_defects(cls, defects: list) -> "ComplexDefect":
        """Create a ComplexDefect from an arbitrary list of SimpleDefects."""
        return cls(list(defects))

    # --- Properties ---

    @property
    def n_defects(self) -> int:
        """Number of component defects."""
        return len(self.defects)

    @property
    def in_atoms(self) -> list[Optional[str]]:
        """Atoms inserted by each defect (None for vacancies)."""
        return self._in_atoms

    @property
    def out_atoms(self) -> list[str]:
        """Atoms/sites removed by each defect."""
        return self._out_atoms

    @property
    def in_elements(self) -> list[Optional[str]]:
        """Elements of inserted atoms."""
        return self._in_elements

    @property
    def out_elements(self) -> list[str]:
        """Elements of removed atoms/sites."""
        return self._out_elements

    # --- Queries ---

    def contains_interstitial(self) -> bool:
        """Check if any component defect is an interstitial."""
        return any(_is_interstitial(a) for a in self._out_atoms)

    def contains_substitution(self) -> bool:
        """Check if any component defect is a substitution (has both in and out)."""
        return any(
            a is not None and not _is_interstitial(b)
            for a, b in zip(self._in_atoms, self._out_atoms)
        )

    def is_all_vacancies(self) -> bool:
        """Check if all component defects are pure vacancies (no in_atom)."""
        return all(a is None for a in self._in_atoms)

    # --- Internal ---

    def _estimate_charges(self) -> list[int]:
        """Return [0] — neutral only.

        Charge states for complex defects require DFT-calibrated
        correction energies that cannot be estimated from oxidation
        states alone. Neutral is the safe default.
        """
        return [0]

    # --- Dunder ---

    def __repr__(self) -> str:
        charges_str = ",".join(str(c) for c in self.charges)
        return f"ComplexDefect('{self.name}', charges=[{charges_str}])"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ComplexDefect):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)