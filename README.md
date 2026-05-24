# pydefect-complex

Systematic complex (multi-component) defect generation compatible with [pydefect](https://github.com/kumagai-group/pydefect).

Generates N-body defect clusters — vacancy pairs, vacancy+dopant complexes, co-doping — with symmetry-aware site enumeration and distance filtering.

## Quick start (CLI)

```bash
# 1. Prerequisite: create supercell_info.json (standard pydefect)
pydefect supercell -p POSCAR --matrix 3 3 3

# 2. Complex defect generation (registry only, no POSCAR files)
pydefect_complex -d N B -n 2
```

Output in ``defect/``:

| File | Contents |
|------|----------|
| `complex_defect_in.yaml` | Defect registry (name → charge list), same format as pydefect's ``defect_in.yaml`` |
| `defect_summary.txt` | Human-readable table with point group, space group, orientation count |
| `parameters.yaml` | Run parameters and cache status |
| `geometries_N*.yaml` | Geometry cache (cross-process, reused on subsequent runs) |

Use ``--structures`` to write per-defect POSCAR directories for VASP calculations:

```bash
pydefect_complex -d N B -n 2 --structures
# → defect/{name}_{charge}/POSCAR + prior_info.yaml + defect_entry.json
```

See ``examples/`` for a full walkthrough with ``pydefect_complex.log``.

## Usage

```
pydefect_complex [-d DOPANTS ...] [-n N_BODY] [-g] [--structures] [--workers N] [-v]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --dopants` | intrinsic only | Dopant elements (e.g. ``-d N B``) |
| `-n, --n-body` | 2 | Maximum order (generates 2..n) |
| `-g, --geometries-only` | off | Enumerate geometry only, no entries or output files |
| `--max-distance` | 3.0 Å | Defect-defect edge cutoff |
| `--min-distance` | 0.3 Å | Minimum defect separation |
| `--charges` | [0] | Charge states to generate |
| `--structures` | off | Write per-defect POSCAR directories |
| `--workers` | CPU count | Worker processes for parallel enumeration + structure generation |
| `-v, --verbose` | off | Debug logging + pipeline tracking |

Progress bars are shown automatically when `tqdm` is installed (``pip install tqdm``).

Geometry cache (``defect/geometries_N*.yaml``) is written on every run and
automatically loaded on the next run — geometry enumeration is never repeated
for the same supercell + distance parameters.

## Pipeline

```
POSCAR → pydefect (supercell_info.json) → pydefect-complex → pydefect (efnv/des/pe)
```

1. Run pydefect's ``supercell`` command to get ``supercell_info.json``
2. Run ``pydefect_complex`` to generate complex defect registry
3. (Optional) Run ``pydefect_complex --structures`` to generate VASP POSCAR files
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

Key design: **geometry is decoupled from chemistry**. `ComplexDefectGraph` nodes
carry only (wyckoff, element) labels — defect compositions are assigned after
geometry enumeration by wyckoff label matching. Geometry cache is persisted
across process boundaries, so changing dopants reuses the same geometry skeleton.

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
Geometry enumeration and structure generation are automatically parallelized
across all available CPU cores (``--workers`` to override).