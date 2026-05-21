# Diamond Complex Defect Generation

"""
Example: generating complex defect structures for diamond (C).

This notebook demonstrates the full workflow:
1. Create a SupercellInfo from a diamond POSCAR (using pydefect)
2. Initialize ComplexDefectMaker
3. Generate all defect pairs within a distance cutoff
4. Write output for pydefect post-processing

Requirements:
    pip install pydefect pydefect-complex
"""

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------

# If you already have a supercell_info.json from pydefect, skip to step 2.
# Otherwise, generate one:

import subprocess
import os
from pathlib import Path

# Write diamond POSCAR
diamond_poscar = """diamond
1.0
    3.56679000000000    0.00000000000000    0.00000000000000
    0.00000000000000    3.56679000000000    0.00000000000000
    0.00000000000000    0.00000000000000    3.56679000000000
C
8
Direct
    0.00000000000000    0.00000000000000    0.00000000000000
    0.00000000000000    0.50000000000000    0.50000000000000
    0.50000000000000    0.00000000000000    0.50000000000000
    0.50000000000000    0.50000000000000    0.00000000000000
    0.25000000000000    0.25000000000000    0.25000000000000
    0.25000000000000    0.75000000000000    0.75000000000000
    0.75000000000000    0.25000000000000    0.75000000000000
    0.75000000000000    0.75000000000000    0.25000000000000
"""

workdir = Path("./diamond_defects")
workdir.mkdir(exist_ok=True)
(workdir / "POSCAR").write_text(diamond_poscar)

# Generate supercell
subprocess.run(
    ["pydefect", "s", "-p", "POSCAR", "--max_atoms", "300", "--min_atoms", "100"],
    cwd=workdir,
    check=True,
)

# ---------------------------------------------------------------------------
# 1. Initialize ComplexDefectMaker
# ---------------------------------------------------------------------------

from pydefect_complex import ComplexDefectMaker

maker = ComplexDefectMaker.from_supercell_info(
    str(workdir / "supercell_info.json"),
    dopants=["N", "B"],    # Substitutional dopants to consider
    max_distance=4.0,      # Max center-to-center distance (Å)
    min_distance=0.5,      # Min distance to avoid overlapping defects
)

print(maker)
print(f"\nSingle defect types: {maker.defect_names}")
print(f"Total defect pairs: {len(maker.defect_pairs)}")

# ---------------------------------------------------------------------------
# 2. Generate all complex defect entries
# ---------------------------------------------------------------------------

entries = maker.make_all_pairs()

print(f"\nGenerated {len(entries)} complex defect entries")

# Show first few entries
for e in entries[:10]:
    print(f"  {e.name:40s}  d={e.distance:.3f} Å")

# ---------------------------------------------------------------------------
# 3. Inspect a specific pair
# ---------------------------------------------------------------------------

# Look at vacancy-vacancy pairs
vac_entries = [e for e in entries if "v_C" in e.name]
print(f"\nVacancy-vacancy pairs: {len(vac_entries)}")
for e in sorted(vac_entries, key=lambda x: x.distance)[:5]:
    print(f"  {e.name:40s}  d={e.distance:.3f} Å")

# ---------------------------------------------------------------------------
# 4. Write output for pydefect pipeline
# ---------------------------------------------------------------------------

output_dir = workdir / "defect"
output_dir.mkdir(exist_ok=True)

maker.write(entries, str(output_dir))

print(f"\nOutput written to {output_dir}/")
print(f"Files: {sorted(os.listdir(output_dir))[:10]}...")

# ---------------------------------------------------------------------------
# 5. Next steps with pydefect
# ---------------------------------------------------------------------------

print("""
Now you can run the standard pydefect pipeline:
  cd diamond_defects/defect
  pydefect_vasp de                           # Generate VASP inputs
  # ... submit VASP calculations ...
  pydefect efnv -d *_* -pcr perfect/calc_results.json \\
      -u ../unitcell/unitcell.yaml
  pydefect des -d *_* -u ../unitcell/unitcell.yaml \\
      -pbes perfect/perfect_band_edge_state.json \\
      -t ../cpd/target_vertices.yaml
""")

# ---------------------------------------------------------------------------
# 6. Programmatic access: generate one specific pair
# ---------------------------------------------------------------------------

# Fine-grained control: generate only V_C + N_C pairs
n_entries = maker.make_pair("v_C_1", "N_C_1", max_distance=3.0)
print(f"v_C_1 + N_C_1: {len(n_entries)} entries within 3.0 Å")
for e in n_entries:
    print(f"  site={e.site_path[1]}, d={e.distance:.3f} Å")