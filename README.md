# pydefect-complex

Systematic complex (multi-component) defect generation compatible with [pydefect](https://github.com/kumagai-group/pydefect).

Generates N-body defect clusters — vacancy pairs, vacancy+dopant complexes, co-doping — with symmetry-aware site enumeration and distance filtering.

## Quick start

```python
from pydefect_complex import ComplexDefectMaker

# 1. Create maker from pydefect's supercell_info.json
maker = ComplexDefectMaker.from_supercell_info(
    "defect/supercell_info.json",
    dopants=["N", "B"],
    max_distance=4.0,     # cutoff for defect-defect edges (Å)
)

# 2. Enumerate geometries (no chemistry)
geoms = maker.make_all_pairs()       # N=2
geoms = maker.make_all_n_body(n=3)   # N-body
maker.show_geometries(N_max=3)       # human-readable summary

# 3. Assign defect compositions + generate structures
entries = maker.generate_entries(n=2)
entries = maker.generate_entries(n=3, dopants=["N", "B"])
entries = maker.generate_entries(geoms, dopants=["Si"])

# Or generate a specific pair
entries = maker.make_pair("Va_C1", "N_C1")

# 4. Write pydefect-compatible output
maker.write(entries, "defect/")                  # POSCAR + prior_info.yaml
maker.write(entries, "defect/", merge=True)      # also merge into defect_in.yaml
```

Output per defect: `defect/{name}_{charge}/` containing `POSCAR`, `prior_info.yaml`, and `defect_entry.json`.

## Pipeline

```
POSCAR → pydefect (supercell_info.json) → pydefect-complex → pydefect (efnv/des/pe)
```

1. Run pydefect's `defect_set_maker` to get `supercell_info.json` + `defect_in.yaml`
2. Use pydefect-complex to generate complex defect structures
3. Merge `complex_defect_in.yaml` into `defect_in.yaml` (or use separately)
4. Continue with standard pydefect VASP workflow

## Architecture

Graph-based Apriori enumeration (PLAN-C):

| Module | Role |
|--------|------|
| `core.py` | `ComplexDefect` — composition of N `SimpleDefect` objects |
| `graph.py` | `HostGraph` (crystal site registry) + `ComplexDefectGraph` (geometry-only) |
| `enumerate.py` | Apriori incremental enumeration + wyckoff composition matching + structure generation |
| `structure.py` | `ComplexDefectEntry` + orientation counting + point group classification |
| `symmetry.py` | Cross-composition geometric deduplication |
| `io.py` | pydefect-compatible file output |

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥3.10, pydefect, pymatgen, scipy, numpy, pyyaml.

## Test

```bash
pytest                          # all tests
pytest tests/test_core.py       # single module
pytest -k "test_make_pair"      # keyword match
```

Performance: N=2 < 2s, N=3 < 20s (diamond 128-atom supercell, 4.0 Å cutoff).