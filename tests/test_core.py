"""Tests for ComplexDefect data class.

Uses pydefect's actual naming conventions:
- Vacancy on C site 1: SimpleDefect(None, "C1", [0,1]) → name="Va_C1"
- N substitution on C site 1: SimpleDefect("N", "C1", [-1,0,1]) → name="N_C1"
- out_atom is the element+index label (e.g. "C1"), not the element alone
"""

import pytest

pytest.importorskip("pydefect")

from pydefect.input_maker.defect import SimpleDefect
from pydefect_complex.core import ComplexDefect, _get_element, _is_interstitial


class TestGetElement:
    def test_site_label(self):
        assert _get_element("C1") == "C"

    def test_two_letter_element(self):
        assert _get_element("Si1") == "Si"

    def test_interstitial(self):
        assert _get_element("i1") == "i"
        assert _get_element("i12") == "i"


class TestIsInterstitial:
    def test_interstitial(self):
        assert _is_interstitial("i1") is True
        assert _is_interstitial("i12") is True

    def test_not_interstitial(self):
        assert _is_interstitial("C1") is False
        assert _is_interstitial("N_C1") is False


class TestComplexDefect:
    @pytest.fixture
    def v_C1(self):
        # SimpleDefect(in_atom, out_atom, charge_list)
        return SimpleDefect(None, "C1", [0, 1])

    @pytest.fixture
    def v_C2(self):
        return SimpleDefect(None, "C2", [0, 1])

    @pytest.fixture
    def N_C1(self):
        return SimpleDefect("N", "C1", [-1, 0, 1])

    def test_creation_from_pair(self, v_C1, v_C2):
        cd = ComplexDefect.from_pair(v_C1, v_C2)
        # Sorted by out_atom reverse: "C2" before "C1" → "Va_C2" first
        assert cd.name == "Va_C2+Va_C1"
        assert cd.n_defects == 2
        assert cd.is_all_vacancies() is True
        assert cd.contains_interstitial() is False

    def test_deterministic_sorting(self, v_C1, v_C2):
        cd1 = ComplexDefect.from_pair(v_C1, v_C2)
        cd2 = ComplexDefect.from_pair(v_C2, v_C1)
        assert cd1.name == cd2.name
        assert cd1 == cd2
        assert hash(cd1) == hash(cd2)

    def test_vacancy_substitution_pair(self, v_C1, N_C1):
        cd = ComplexDefect.from_pair(v_C1, N_C1)
        # Names sorted by out_atom reverse:
        # out_atom both "C1" → tie, then by name: "Va_C1" vs "N_C1" → "Va" > "N"
        assert cd.name == "Va_C1+N_C1"
        assert cd.is_all_vacancies() is False
        assert cd.contains_substitution() is True

    def test_charges_estimated(self, v_C1, v_C2, N_C1):
        cd_vac = ComplexDefect.from_pair(v_C1, v_C2)
        assert len(cd_vac.charges) > 0
        assert 0 in cd_vac.charges

        cd_mix = ComplexDefect.from_pair(v_C1, N_C1)
        assert len(cd_mix.charges) > 0

    def test_from_defects_list(self, v_C1, v_C2, N_C1):
        cd = ComplexDefect.from_defects([v_C1, v_C2, N_C1])
        assert cd.n_defects == 3

    def test_repr(self, v_C1, v_C2):
        cd = ComplexDefect.from_pair(v_C1, v_C2)
        r = repr(cd)
        assert "ComplexDefect" in r
        assert "Va_C2+Va_C1" in r

    def test_in_out_atoms(self, v_C1, N_C1):
        cd = ComplexDefect.from_pair(v_C1, N_C1)
        # v_C1: in_atom=None, out_atom="C1"
        # N_C1: in_atom="N",  out_atom="C1"
        assert None in cd.in_atoms
        assert "N" in cd.in_atoms
        assert "C1" in cd.out_atoms

    def test_out_elements(self, v_C1, N_C1):
        cd = ComplexDefect.from_pair(v_C1, N_C1)
        # Both out_atoms are "C1" → elements are "C"
        assert all(e == "C" for e in cd.out_elements)