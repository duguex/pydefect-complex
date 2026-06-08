"""Error-path and edge-case tests for pydefect-complex.

Covers the failure modes that aren't exercised by the happy-path
integration tests:
- ValueError on bad N (n < 2)
- argparse rejection of n-body < 2
- Empty geometry enumeration when host graph is degenerate
- Silent skip of failed structure generation
- Empty entry list through the dedup + write pipeline
"""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("pydefect")


# ---------------------------------------------------------------------------
# maker.make_all_n_body: ValueError on bad N
# ---------------------------------------------------------------------------


class TestMakeAllNBodyValidation:
    def test_n_zero_raises(self, diamond_supercell_info):
        from pydefect_complex import ComplexDefectMaker

        maker = ComplexDefectMaker(diamond_supercell_info)
        with pytest.raises(ValueError, match="n must be >= 2"):
            maker.make_all_n_body(n=0)

    def test_n_negative_raises(self, diamond_supercell_info):
        from pydefect_complex import ComplexDefectMaker

        maker = ComplexDefectMaker(diamond_supercell_info)
        with pytest.raises(ValueError, match="n must be >= 2"):
            maker.make_all_n_body(n=-1)


# ---------------------------------------------------------------------------
# CLI: argparse rejection of n-body < 2
# ---------------------------------------------------------------------------


class TestCLIValidation:
    @pytest.fixture
    def supercell_info_json(self, diamond_supercell_info, tmp_path):
        """Local copy of the test_cli.py fixture."""
        path = tmp_path / "supercell_info.json"
        with open(path, "w") as f:
            json.dump(diamond_supercell_info.as_dict(), f)
        return path

    def test_n_body_one_exits_with_error(self, supercell_info_json, tmp_path):
        """CLI must reject n-body < 2 via argparse error (SystemExit)."""
        from pydefect_complex.cli import main

        orig = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with pytest.raises(SystemExit):
                main(["-n", "1"])
        finally:
            os.chdir(orig)

    def test_n_body_zero_exits_with_error(self, supercell_info_json, tmp_path):
        from pydefect_complex.cli import main

        orig = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with pytest.raises(SystemExit):
                main(["-n", "0"])
        finally:
            os.chdir(orig)

    def test_min_distance_larger_than_max_distance_works_but_produces_no_geometries(
        self, diamond_supercell_info,
    ):
        """When min_distance >= max_distance, geometry enumeration finds nothing.

        This is not an error — the API is permissive, but the result is
        an empty geometry dict. Callers must handle this.
        """
        from pydefect_complex import ComplexDefectMaker

        maker = ComplexDefectMaker(
            diamond_supercell_info, max_distance=2.0, min_distance=5.0,
        )
        geoms = maker.enumerate_geometries(N_max=2)
        # All anchors are filtered by min_distance > max_distance — should
        # produce 0 unique 2-node geometries.
        assert geoms.get(2, []) == []


# ---------------------------------------------------------------------------
# Empty / degenerate output through the full pipeline
# ---------------------------------------------------------------------------


class TestEmptyOutputHandling:
    def test_generate_entries_with_no_geometries_returns_empty(
        self, diamond_supercell_info,
    ):
        """If geometry enumeration yields 0 geometries, generate_entries
        must return [] without raising."""
        from pydefect_complex import ComplexDefectMaker

        maker = ComplexDefectMaker(
            diamond_supercell_info, max_distance=2.0, min_distance=5.0,
        )
        entries = maker.generate_entries(n_or_geometries=2)
        assert entries == []

    def test_write_with_empty_entries_produces_minimal_output(
        self, diamond_supercell_info, tmp_output_dir,
    ):
        """maker.write() with no entries should not crash; it should
        produce a complex_defect_in.yaml with no entries (or an empty file)."""
        from pydefect_complex import ComplexDefectMaker
        import yaml

        maker = ComplexDefectMaker(
            diamond_supercell_info, max_distance=2.0, min_distance=5.0,
        )
        # No entries to write — should not raise.
        # maker.write expects a list of entries; pass [] to exercise the path.
        yaml_path = maker.write([], tmp_output_dir)
        assert os.path.exists(yaml_path)
        data = yaml.safe_load(Path(yaml_path).read_text()) or {}
        assert data == {}


# ---------------------------------------------------------------------------
# dedup: empty and degenerate inputs
# ---------------------------------------------------------------------------


class TestDeduplicateEdgeCases:
    def test_deduplicate_single_entry_returns_one(self, diamond_supercell_info):
        """A single entry should pass through unchanged (one cluster of 1)."""
        from pydefect_complex import ComplexDefectMaker
        from pydefect_complex.graph import HostGraph
        from pydefect_complex.symmetry import deduplicate

        maker = ComplexDefectMaker(diamond_supercell_info, max_distance=3.5)
        maker.make_all_pairs()
        entries = maker.generate_entries(n_or_geometries=2)
        if not entries:
            pytest.skip("No entries produced")

        hg = HostGraph.from_supercell_info(diamond_supercell_info)
        out = deduplicate(entries[:1], hg, max_distance=3.5)
        assert len(out) == 1
        # Name should end in .001
        assert out[0].name.rsplit(".", 1)[-1] == "001"


# ---------------------------------------------------------------------------
# Structure generation: silent skip on bad composition
# ---------------------------------------------------------------------------


class TestStructureGenerationFailure:
    def test_silent_skip_on_structure_generation_failure(
        self, diamond_supercell_info, monkeypatch,
    ):
        """When _generate_structure raises ValueError, the entry is
        silently skipped. We patch the helper to always raise and
        verify generate_all_entries still returns (an empty) list
        without propagating the exception.
        """
        from pydefect_complex import ComplexDefectMaker
        from pydefect_complex.enumerate import _generate_structure

        def always_fails(*args, **kwargs):
            raise ValueError("simulated failure")

        monkeypatch.setattr(
            "pydefect_complex.enumerate._generate_structure", always_fails,
        )

        maker = ComplexDefectMaker(
            diamond_supercell_info, dopants=["N"], max_distance=4.0,
        )
        # Should not raise — caller is supposed to silently skip.
        # generate_all_entries calls _generate_structure which now
        # always fails. All entries are skipped → empty list.
        try:
            entries = maker.generate_entries(n_or_geometries=2)
            # If it returns, the silent-skip path is exercised.
            assert entries == []
        except (ValueError, IndexError):
            # If it propagates, that's a regression — the silent
            # except was removed at some point.
            pytest.fail(
                "generate_entries should silently skip failed structure "
                "generation, not propagate ValueError"
            )
