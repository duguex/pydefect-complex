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
pydefect_complex [-d DOPANTS ...] [-n N_BODY] [--max-distance DIST]
                 [--min-distance DIST] [--charges CHARGES]
                 [-g] [--structures] [--workers N] [-v]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --dopants` | intrinsic only | Dopant element symbols, e.g. ``-d N B``. Omit for vacancies only |
| `-n, --n-body` | 2 | Maximum order (generates all complexes 2..n). N=2 for pairs, N=3 for trimers, etc. |
| `--max-distance` | 3.0 Å | Maximum defect-defect distance for edge inclusion. Smaller = faster, fewer complexes |
| `--min-distance` | 0.3 Å | Minimum defect separation (rejects overlap) |
| `--charges` | [0] | Charge states to generate, e.g. ``--charges -2 -1 0 1 2`` |
| `-g, --geometries-only` | off | Enumerate geometries only, save cache, exit (no entries or POSCAR) |
| `--structures` | off | Write per-defect POSCAR directories (otherwise registry-only) |
| `--workers` | CPU count | Number of worker processes for parallel enumeration |
| `-v, --verbose` | off | Verbose DEBUG logging + pipeline tracking |

### Examples

```bash
# Intrinsic vacancies only (no dopants)
pydefect_complex -n 2

# N-B co-doping pairs
pydefect_complex -d N B -n 2 --max-distance 4.0

# N=3 complexes (trimers) with multiple charge states, using 8 workers
pydefect_complex -d N B -n 3 --charges -2 -1 0 1 2 --workers 8

# Geometry enumeration only (skip structure generation for inspection)
pydefect_complex -d N B -n 4 --geometries-only

# Full pipeline with POSCAR output
pydefect_complex -d N B -n 3 --structures
```

### Output directory

After a run, ``defect/`` contains:

| File | Contents |
|------|----------|
| ``complex_defect_in.yaml`` | Defect registry (name → charge list), pydefect-compatible |
| ``defect_summary.txt`` | Human-readable table: point group, orientations per geometry |
| ``parameters.yaml`` | Run parameters and cache status |
| ``geometries_N*.yaml`` | Geometry cache (cross-process, reused on subsequent runs) |

With ``--structures``, each defect also gets a subdirectory:

```
defect/Va_C1+Va_C1.001_0/POSCAR + prior_info.yaml
defect/N_C1+B_C1.002_1/POSCAR + prior_info.yaml
...
```

### Progress bars

Progress bars are shown automatically when ``tqdm`` is installed:

```bash
pip install tqdm
```

Geometry enumeration (``_extend_order``), structure generation (``generate_all_entries``),
and deduplication all display live progress. If ``tqdm`` is not installed, runs silently.

### Caching

Geometry cache (``defect/geometries_N*.yaml``) is written on every run and
automatically loaded on the next run — geometry enumeration is never repeated
for the same supercell + distance parameters. Changing dopants reuses cached
geometries (only entry generation is re-run).

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