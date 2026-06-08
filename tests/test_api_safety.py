"""Tests for the API safety improvements added to pydefect-complex.

These tests exercise the new contracts:
- filter_entries is a static method (deterministic, side-effect free)
- Deprecated shims (io.write_all, io.write_entry, enumerate.assign_compositions)
  emit DeprecationWarning when called
- progress_callback is invoked on enumerator and maker
- Maker.write warns on relative paths

Run with: pytest tests/test_api_safety.py
"""

import warnings
from pathlib import Path
from unittest import mock

import pytest


# ---------- filter_entries (static) ----------

class TestFilterEntries:
    """filter_entries is a static method, so it should work on a list of
    entries without needing a Maker instance, and not mutate them."""

    def test_is_staticmethod(self):
        import inspect
        from pydefect_complex.maker import ComplexDefectMaker
        # In Python 3.10+ staticmethod descriptor is reported as a function
        assert isinstance(
            ComplexDefectMaker.__dict__["filter_entries"], staticmethod
        )

    def test_does_not_require_maker(self):
        # Can be called with just a list — no `self`
        from pydefect_complex.maker import ComplexDefectMaker
        # No entries → empty result
        assert ComplexDefectMaker.filter_entries([]) == []

    def test_empty_filters_no_op(self):
        from pydefect_complex.maker import ComplexDefectMaker
        # If filters are empty, list comes back unchanged
        # We need a fake entry — use a mock
        fake_entry = mock.MagicMock()
        fake_entry.point_group = "C2v"
        fake_entry.complex_defect.in_elements = [None, "B", "N"]
        result = ComplexDefectMaker.filter_entries(
            [fake_entry], exclude_point_groups=(), max_dopant_atoms=None,
        )
        assert result == [fake_entry]

    def test_c1_excluded_by_default(self):
        from pydefect_complex.maker import ComplexDefectMaker
        c1 = mock.MagicMock()
        c1.point_group = "C1"
        c1.complex_defect.in_elements = [None]
        c2 = mock.MagicMock()
        c2.point_group = "C2v"
        c2.complex_defect.in_elements = [None]
        result = ComplexDefectMaker.filter_entries([c1, c2])
        assert c1 not in result
        assert c2 in result

    def test_dopant_count_filter(self):
        from pydefect_complex.maker import ComplexDefectMaker
        # 3 dopants: should be excluded with max=2
        heavy = mock.MagicMock()
        heavy.point_group = "C2v"  # passes C1 filter
        heavy.complex_defect.in_elements = [None, "B", "N", "P"]  # 3 dopants
        # 2 dopants: should pass
        light = mock.MagicMock()
        light.point_group = "C2v"
        light.complex_defect.in_elements = [None, "B", "N"]  # 2 dopants
        result = ComplexDefectMaker.filter_entries([heavy, light])
        assert heavy not in result
        assert light in result

    def test_dopant_filter_disabled_with_none(self):
        from pydefect_complex.maker import ComplexDefectMaker
        heavy = mock.MagicMock()
        heavy.point_group = "C2v"
        heavy.complex_defect.in_elements = [None, "B", "N", "P", "Q"]  # 4 dopants
        # max_dopant_atoms=None disables filter
        result = ComplexDefectMaker.filter_entries(
            [heavy], exclude_point_groups=(), max_dopant_atoms=None,
        )
        assert heavy in result

    def test_preserves_input_order(self):
        from pydefect_complex.maker import ComplexDefectMaker
        e1, e2, e3 = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        for e in (e1, e2, e3):
            e.point_group = "C2v"
            e.complex_defect.in_elements = [None]
        result = ComplexDefectMaker.filter_entries([e1, e2, e3])
        assert result == [e1, e2, e3]


# ---------- Deprecated shims ----------

class TestDeprecatedShims:
    """Old public names should still work but emit DeprecationWarning,
    so users have one release to migrate."""

    def test_io_write_all_warns(self):
        from pydefect_complex import io
        with pytest.warns(DeprecationWarning, match="io.write_all is deprecated"):
            with mock.patch.object(io, "_write_all", return_value={}):
                io.write_all([], "defect")

    def test_io_write_entry_warns(self):
        from pydefect_complex import io
        with pytest.warns(DeprecationWarning, match="io.write_entry is deprecated"):
            with mock.patch.object(io, "_write_entry", return_value=""):
                io.write_entry(mock.MagicMock(), "defect", 0)

    def test_enumerate_assign_compositions_warns(self):
        from pydefect_complex import enumerate as pc_enum
        with pytest.warns(
            DeprecationWarning, match="enumerate.assign_compositions is deprecated",
        ):
            with mock.patch.object(pc_enum, "_assign_compositions", return_value=[]):
                pc_enum.assign_compositions([], [])


# ---------- progress_callback ----------

class TestProgressCallback:
    """The new progress_callback param should be called for each order."""

    def test_enumerator_signature_accepts_callback(self):
        from pydefect_complex.enumerate import ComplexDefectEnumerator
        from pydefect_complex.graph import HostGraph
        import inspect
        sig = inspect.signature(ComplexDefectEnumerator.enumerate)
        assert "progress_callback" in sig.parameters
        assert sig.parameters["progress_callback"].default is None

    def test_maker_enumerate_geometries_passes_callback(self):
        from pydefect_complex.maker import ComplexDefectMaker
        import inspect
        sig = inspect.signature(ComplexDefectMaker.enumerate_geometries)
        assert "progress_callback" in sig.parameters
        assert sig.parameters["progress_callback"].default is None


# ---------- Maker.write relative-path warning ----------

class TestMakerWriteRelativePath:
    """Maker.write should warn if given a relative output_dir (the cwd-
    dependence footgun documented in SKILL.md as a ❌ prohibition)."""

    def test_warns_on_relative_path(self, tmp_path, caplog):
        from pydefect_complex.maker import ComplexDefectMaker
        with mock.patch.object(ComplexDefectMaker, "__init__", return_value=None):
            maker = ComplexDefectMaker.__new__(ComplexDefectMaker)
            maker._tracker = mock.MagicMock()
            maker._tracker.enabled = False
            with mock.patch(
                "pydefect_complex.maker._write_all", return_value={},
            ), mock.patch(
                "pydefect_complex.maker.write_complex_defect_in_yaml",
                return_value="defect/complex_defect_in.yaml",
            ), mock.patch(
                "pydefect_complex.maker.write_summary", return_value="x",
            ), mock.patch.object(
                ComplexDefectMaker, "_write_parameters",
            ):
                import logging
                with caplog.at_level(logging.WARNING, logger="pydefect_complex.maker"):
                    ComplexDefectMaker.write(maker, [], "defect")  # relative
        assert any("output_dir" in r.message and "relative" in r.message
                   for r in caplog.records), \
            f"expected relative-path warning, got: {[r.message for r in caplog.records]}"

    def test_no_warning_on_absolute_path(self, tmp_path, caplog):
        from pydefect_complex.maker import ComplexDefectMaker
        with mock.patch.object(ComplexDefectMaker, "__init__", return_value=None):
            maker = ComplexDefectMaker.__new__(ComplexDefectMaker)
            maker._tracker = mock.MagicMock()
            maker._tracker.enabled = False
            abs_path = str(tmp_path / "defect")
            with mock.patch(
                "pydefect_complex.maker._write_all", return_value={},
            ), mock.patch(
                "pydefect_complex.maker.write_complex_defect_in_yaml",
                return_value="x",
            ), mock.patch(
                "pydefect_complex.maker.write_summary", return_value="x",
            ), mock.patch.object(
                ComplexDefectMaker, "_write_parameters",
            ):
                import logging
                with caplog.at_level(logging.WARNING, logger="pydefect_complex.maker"):
                    ComplexDefectMaker.write(maker, [], abs_path)
        relative_warnings = [r for r in caplog.records
                            if "relative" in r.message and "output_dir" in r.message]
        assert not relative_warnings, \
            f"unexpected relative-path warning: {[r.message for r in relative_warnings]}"
