# pydefect-complex

Systematic complex (multi-component) defect generation for [pydefect](https://github.com/kumagai-group/pydefect).

Generate n-body defect clusters — vacancy pairs, vacancy+dopant complexes, co-doping configurations — with symmetry-aware site enumeration and distance filtering. Output is fully compatible with pydefect's post-processing pipeline (`pydefect efnv` → `pydefect des` → `pydefect pe`).

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10 and pydefect.

## Quick Start

```python
from pydefect_complex import ComplexDefectMaker

# From a pydefect supercell_info.json
maker = ComplexDefectMaker.from_supercell_info(
    "defect/supercell_info.json",
    dopants=["N", "B"],
    max_distance=4.0,
)

# Generate all defect pairs
entries = maker.make_all_pairs()
print(f"Generated {len(entries)} complex defect entries")

# Write to pydefect-compatible directory structure
maker.write(entries, "defect")
```

After writing, run the standard pydefect pipeline:

```bash
cd defect
pydefect_vasp de                    # Generate VASP inputs
# ... submit calculations ...
pydefect efnv -d *_* -pcr perfect/calc_results.json -u ../unitcell/unitcell.yaml
pydefect des -d *_* -u ../unitcell/unitcell.yaml -pbes perfect/perfect_band_edge_state.json -t ../cpd/target_vertices.yaml
```

## API

### ComplexDefectMaker

Main entry point. Mirrors pydefect's `DefectSetMaker` API style.

```python
maker = ComplexDefectMaker(supercell_info, dopants=["N"], max_distance=5.0)

# Or from file:
maker = ComplexDefectMaker.from_supercell_info("supercell_info.json", dopants=["N"])

# Inspect
maker.single_defects    # list[SimpleDefect]
maker.defect_names      # list[str]
maker.defect_pairs      # list[tuple[SimpleDefect, SimpleDefect]]

# Generate
entries = maker.make_pair("v_C_1", "N_C_1", max_distance=4.0)
entries = maker.make_all_pairs(max_distance=4.0, deduplicate_symmetry=True)

# Write
maker.write(entries, "defect", merge=True)  # merge=True updates defect_in.yaml
```

### ComplexDefect

Data class representing an n-component defect.

```python
from pydefect_complex import ComplexDefect

cd = ComplexDefect.from_pair(defect1, defect2)
cd.name        # "v_C_1+N_C_2"
cd.charges     # [-1, 0, 1]  (estimated charge states)
cd.n_defects   # 2
```

### ComplexDefectEntry

Generated structure with metadata.

```python
entry.name         # "v_C_1+v_C_2.C_5"
entry.distance     # 2.52  (Å between defect centers)
entry.structure    # pymatgen IStructure
entry.site_path    # ("C_1", "C_5")  (sites of each defect)
```

## Algorithm

The structure generation works by layer-by-layer defect application:

1. Apply defect₁ to pristine supercell → struct₁
2. Symmetrize struct₁ → find inequivalent sites for defect₂
3. Filter by element match + distance thresholds
4. Apply defect₂ at each valid site → struct₂
5. (Future: recurse for n > 2)

Symmetry deduplication removes entries where the site pair is
equivalent under the pristine structure's space group.

## Output Structure

```
defect/
├── v_C_1+v_C_2.C_5_-1/
│   ├── POSCAR
│   ├── prior_info.yaml
│   └── defect_entry.json
├── v_C_1+v_C_2.C_5_0/
│   └── ...
├── v_C_1+N_C_2.C_3_0/
│   └── ...
├── complex_defect_in.yaml
└── defect_in.yaml          # (if merge=True)
```

## Architecture

```
ComplexDefectMaker          # Main entry point
├── core.py                 # ComplexDefect data model (n-body support)
├── structure.py            # Layer-by-layer structure generation
├── symmetry.py             # Symmetry-based deduplication
└── io.py                   # pydefect-compatible file output
```

Currently implements N=2 (defect pairs). The architecture supports
N-body extension via `ComplexDefect.defects` (list of N SimpleDefects)
and the recursive structure of `structure.generate_entries()`.

## License

MIT