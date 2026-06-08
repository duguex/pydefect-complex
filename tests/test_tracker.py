"""Unit tests for PipelineTracker (tracker.py)."""

from pathlib import Path

import pytest


class TestPipelineTrackerDisabled:
    """When enabled=False, all writes should be no-ops (return None)."""

    def test_disabled_constructor(self, tmp_path):
        from pydefect_complex.tracker import PipelineTracker

        t = PipelineTracker(str(tmp_path), enabled=False)
        assert t.enabled is False
        assert t.cache_dir == Path(tmp_path) / "pipeline_track"
        # Cache dir should NOT be created when disabled.
        assert not t.cache_dir.exists()

    def test_disabled_write_returns_none(self, tmp_path):
        from pydefect_complex.tracker import PipelineTracker

        t = PipelineTracker(str(tmp_path), enabled=False)
        assert t.write_parameters({"n": 1}) is None
        assert t.write_geometries({2: []}, {"max_distance": 3.0}) is None
        assert t.write_dedup_verification({"ok": True}) is None
        assert t.write_entries_metadata([]) is None


class TestPipelineTrackerEnabled:
    """When enabled=True, writes should produce files in pipeline_track/."""

    def test_enabled_constructor_creates_cache_dir(self, tmp_path):
        from pydefect_complex.tracker import PipelineTracker

        PipelineTracker(str(tmp_path), enabled=True)
        assert (Path(tmp_path) / "pipeline_track").is_dir()

    def test_write_parameters_creates_yaml(self, tmp_path):
        from pydefect_complex.tracker import PipelineTracker

        t = PipelineTracker(str(tmp_path), enabled=True)
        path = t.write_parameters({"n_single_defects": 5, "dopants": ["N"]})
        assert path is not None
        assert Path(path).is_file()
        assert Path(path).name == "00_parameters.yaml"
        text = Path(path).read_text()
        assert "n_single_defects: 5" in text
        assert "dopants:" in text

    def test_write_geometries_creates_yaml(self, tmp_path):
        import numpy as np
        from pydefect_complex.tracker import PipelineTracker
        from pydefect_complex.graph import ComplexDefectGraph

        g = ComplexDefectGraph(
            host_node_ids=(0, 1),
            wyckoffs=("C1", "C2"),
            elements=("C", "C"),
            edges=[(0, 1, np.array([1.5, 0.0, 0.0]))],
            n_orientations=2,
            point_group="C2",
        )
        t = PipelineTracker(str(tmp_path), enabled=True)
        path = t.write_geometries(
            {2: [g]},
            {"max_distance": 3.0, "min_distance": 0.3},
        )
        assert path is not None
        assert Path(path).name == "01_geometries.yaml"
        text = Path(path).read_text()
        assert "01_geometry_enumerate" in text
        assert "C2" in text

    def test_write_dedup_verification_warns_on_false_positives(self, tmp_path):
        from pydefect_complex.tracker import PipelineTracker

        t = PipelineTracker(str(tmp_path), enabled=True)
        report = {
            "ok": False,
            "n_entries": 5,
            "n_clusters": 4,
            "false_positives": [{"fingerprint": (1.5,), "cluster_ids": [0, 1], "n_entries": 3}],
            "false_negatives": [],
        }
        path = t.write_dedup_verification(report)
        assert path is not None
        assert Path(path).name == "04_dedup_verification.yaml"
        text = Path(path).read_text()
        assert "ok: false" in text
        assert "false_positives" in text

    def test_write_entries_metadata_creates_json(self, tmp_path):
        from pymatgen.core import Structure, Lattice
        from pydefect_complex.tracker import PipelineTracker
        from pydefect_complex.structure import ComplexDefectEntry
        from pydefect_complex.core import ComplexDefect
        from pydefect.input_maker.defect import SimpleDefect

        # Build a minimal entry.
        s = Structure(Lattice.cubic(5.0), ["C", "C"], [[0, 0, 0], [0.5, 0, 0]])
        d = SimpleDefect(None, "C1", [0])
        cd = ComplexDefect.from_defects([d, d])
        entry = ComplexDefectEntry(
            name="2Va_C1.001",
            complex_defect=cd,
            site_path=("C1", "C1"),
            distances=(2.5,),
            structure=s,
            defect_coords=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        )
        t = PipelineTracker(str(tmp_path), enabled=True)
        path = t.write_entries_metadata([entry])
        assert path is not None
        assert Path(path).name == "05_entries.json"
        import json
        data = json.loads(Path(path).read_text())
        assert data["n_entries"] == 1
        assert data["entries"][0]["name"] == "2Va_C1.001"
        assert data["entries"][0]["n_defects"] == 2

    def test_write_entries_metadata_skips_empty(self, tmp_path):
        from pydefect_complex.tracker import PipelineTracker

        t = PipelineTracker(str(tmp_path), enabled=True)
        assert t.write_entries_metadata([]) is None
        # No file should be created.
        cache = Path(tmp_path) / "pipeline_track"
        assert not (cache / "05_entries.json").exists()
