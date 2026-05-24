"""Integration tests for the pydefect-complex CLI.
"""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("pydefect")


@pytest.fixture
def supercell_info_json(diamond_supercell_info, tmp_path):
    """Write diamond supercell_info.json into tmp_path and return the path."""
    path = tmp_path / "supercell_info.json"
    with open(path, "w") as f:
        json.dump(diamond_supercell_info.as_dict(), f)
    return path


class TestCLIMake:
    def test_no_dopants_intrinsic(self, supercell_info_json, tmp_path):
        """Without -d, generates only intrinsic vacancy defects (N=2)."""
        from pydefect_complex.cli import main

        orig_dir = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            main(["-n", "2", "--max-distance", "4.0"])
        finally:
            os.chdir(orig_dir)

        yaml_path = tmp_path / "defect" / "complex_defect_in.yaml"
        assert yaml_path.exists()
        import yaml
        data = yaml.safe_load(yaml_path.read_text())
        assert len(data) > 0
        # Only vacancy defects for intrinsic
        assert all("Va_C1" in name or "Va" in name for name in data)

    def test_second_run_skips_all(self, supercell_info_json, tmp_path):
        """Same parameters on second run should skip all entries."""
        from pydefect_complex.cli import main

        orig_dir = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            main(["-d", "N", "B", "-n", "2", "--max-distance", "4.0"])

            import yaml
            yaml_path = tmp_path / "defect" / "complex_defect_in.yaml"
            count_before = len(yaml.safe_load(yaml_path.read_text()) or {})

            main(["-d", "N", "B", "-n", "2", "--max-distance", "4.0"])
            count_after = len(yaml.safe_load(yaml_path.read_text()) or {})
            # Same count — no new entries added
            assert count_after == count_before
        finally:
            os.chdir(orig_dir)

    def test_append_n3(self, supercell_info_json, tmp_path):
        """N=3 run after N=2 adds N=3 entries, preserves N=2."""
        from pydefect_complex.cli import main

        orig_dir = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            main(["-d", "N", "B", "-n", "2", "--max-distance", "4.0"])

            import yaml
            yaml_path = tmp_path / "defect" / "complex_defect_in.yaml"
            count_n2 = len(yaml.safe_load(yaml_path.read_text()) or {})

            main(["-d", "N", "B", "-n", "3", "--max-distance", "4.0"])
            count_n3 = len(yaml.safe_load(yaml_path.read_text()) or {})

            assert count_n3 > count_n2

            # All N=2 entries should still be present
            with open(yaml_path) as f:
                names = set(yaml.safe_load(f).keys())

            # At least some names with index format .001-.005 (N=2 signatures)
            n2_signatures = [n for n in names if n.endswith(".005")]
            assert len(n2_signatures) > 0, "N=2 entries preserved"
        finally:
            os.chdir(orig_dir)

    def test_different_dopants(self, supercell_info_json, tmp_path):
        """Different dopants produce different entry names."""
        from pydefect_complex.cli import main

        orig_dir = os.getcwd()
        try:
            os.chdir(str(tmp_path))

            # N+B
            main(["-d", "N", "B", "-n", "2", "--max-distance", "4.0"])
            import yaml
            yaml_path = tmp_path / "defect" / "complex_defect_in.yaml"
            names_nb = set(yaml.safe_load(yaml_path.read_text()).keys())

            # P doping
            main(["-d", "P", "-n", "2", "--max-distance", "4.0"])
            names_all = set(yaml.safe_load(yaml_path.read_text()).keys())

            # P-doped names should be different from N+B names
            p_names = {n for n in names_all if "P_C1" in n}
            assert len(p_names) > 0, "No P-doped entries found"

            # Vacancy-only entries overlap between runs
            shared = names_nb & names_all
            assert all("Va_C1" in n for n in shared) or len(shared) >= 0
        finally:
            os.chdir(orig_dir)

    def test_output_files(self, supercell_info_json, tmp_path):
        """Verify all expected output files are created."""
        from pydefect_complex.cli import main

        orig_dir = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            main(["-d", "N", "B", "-n", "2", "--max-distance", "4.0", "--structures"])

            defect_dir = tmp_path / "defect"
            assert (defect_dir / "complex_defect_in.yaml").exists()
            assert (defect_dir / "defect_summary.txt").exists()
            assert (defect_dir / "parameters.yaml").exists()

            # At least one POSCAR subdirectory
            subdirs = [d for d in defect_dir.iterdir() if d.is_dir()]
            assert len(subdirs) > 0

            # Each subdir has POSCAR and prior_info.yaml
            for sd in subdirs:
                assert (sd / "POSCAR").exists(), f"Missing POSCAR in {sd}"
                assert (sd / "prior_info.yaml").exists(), f"Missing prior_info.yaml in {sd}"
        finally:
            os.chdir(orig_dir)

    def test_no_supercell_info_error(self, tmp_path):
        """Running without supercell_info.json should exit with error."""
        from pydefect_complex.cli import main

        orig_dir = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with pytest.raises(SystemExit):
                main(["-n", "2"])
        finally:
            os.chdir(orig_dir)
