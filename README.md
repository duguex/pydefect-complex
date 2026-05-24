# pydefect-complex

Systematic complex (multi-component) defect generation compatible with [pydefect](https://github.com/kumagai-group/pydefect).

Generates N-body defect clusters — vacancy pairs, vacancy+dopant complexes, co-doping — with symmetry-aware site enumeration and distance filtering.

## Quick start (CLI)

```bash
pydefect supercell -p POSCAR --matrix 3 3 3          # 1. supercell_info.json
pydefect_complex -d N B -n 2                          # 2. complex defects
```

Output in ``defect/``: each defect as ``defect/{name}_{charge}/POSCAR`` + ``prior_info.yaml`` + ``defect_entry.json``, plus ``complex_defect_in.yaml``.

See ``examples/`` for a full walkthrough.

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