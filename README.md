# pydefect-complex

Systematic complex (multi-component) defect generation compatible with [pydefect](https://github.com/kumagai-group/pydefect).

Generates N-body defect clusters — vacancy pairs, vacancy+dopant complexes, co-doping — with symmetry-aware site enumeration and distance filtering.

## When to use this

`pydefect-complex` is the right tool when you need to enumerate **multi-component (N≥2) defects** in a supercell and want them output in a form that drops straight into the standard pydefect pipeline (`efnv` → `des` → `pe`).

| Use this when | Don't use this when |
|---|---|
| You need vacancy pairs, vacancy+dopant pairs, or N+B co-doping | You only need single-point defects — use `pydefect` directly |
| Your supercell is 50–300 atoms (typical 2×2×2 to 4×4×4 of a small cell) | Your cell is >500 atoms — enumeration time scales with neighbor count |
| N ≤ 4 (pairs, trimers, quadrimers) is the relevant regime | N ≥ 5 — combinatorial explosion, requires AI-guided search |
| You have a target charge-state list and need it stamped on every entry | You need charge-state *estimation* — pydefect-complex only assigns; you supply `--charges` |
| You have cubic / hexagonal / orthorhombic / monoclinic lattice symmetry | Your material is amorphous or highly disordered — geometric equivalence is ill-defined |
| You want VASP POSCARs (with `prior_info.yaml` + `defect_entry.json`) | You use a different DFT code (QE, CP2K, …) — output is VASP-specific |

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

End-to-end flow from a primitive POSCAR to VASP-ready complex defect inputs:

1. **Input (`supercell_info.json`).** Run `pydefect supercell -p POSCAR --matrix 3 3 3` (or similar) to produce the supercell metadata file. This carries the structure, the symmetry-derived site labels, and the space-group information.
2. **Host graph construction (`graph.py`).** The pristine supercell is unfolded into a `HostGraph`: every atom becomes a node labeled by `(wyckoff, element)` with fractional coordinates; the lattice matrix is carried for PBC-aware min-image distance computation.
3. **Geometry enumeration (`enumerate.py:ComplexDefectEnumerator`).** Apriori-style incremental enumeration produces all geometrically unique N-node site configurations — first N=2 (anchor + neighbor pairs), then extended by external neighbors up to the requested `N_max`. Online Kabsch-based dedup keeps only one representative per equivalence class; results are cached to `defect/geometries_N*.yaml` for cross-process reuse.
4. **Composition assignment (`enumerate.py:assign_compositions`).** Each geometry is matched against the user's `SimpleDefect` list by comparing the wyckoff multiset of the geometry to the `out_atom` multiset of the defect combination. Filter rules: at most one interstitial, and only in the first layer.
5. **Structure generation (`enumerate.py:_generate_structure`).** For each (geometry, composition) pair, the actual `IStructure` is built: vacancies pop the nearest host atom, substitutions swap the host element for the dopant, interstitials insert the dopant at the wyckoff coordinate. Failures (e.g. element mismatch) are silently skipped.
6. **Cross-composition deduplication (`symmetry.py:deduplicate`).** All entries from step 5 are clustered by geometric equivalence *across compositions* — a vacancy pair and a vacancy+dopant pair on the same sites are recognized as the same geometry. Each surviving cluster gets a `.001`, `.002`, … index per composition.
7. **Orientation counting (`structure.py:_count_orientations_from_coords`).** Each unique geometry is rotated through the pristine crystal's space-group operations, mapped back to host atoms via KDTree, and the number of distinct embeddings is recorded. Used in `defect_summary.txt` and the registry metadata.
8. **Output (`io.py`).** Writes `defect/complex_defect_in.yaml` (the registry), `defect_summary.txt` (human-readable), `defect/parameters.yaml` (run metadata), and — with `--structures` — one `defect/{name}_{charge}/POSCAR` + `prior_info.yaml` + `defect_entry.json` per entry. Files are merged across runs, not overwritten.
9. **Downstream (`pydefect efnv` → `des` → `pe`).** The registry and POSCARs feed straight into pydefect's VASP post-processing: formation energies with finite-size corrections, defect structure analysis, and phase-diagram construction.

The geometry cache (step 3) and the entry cache (step 6) are persisted across runs, so changing dopants re-uses prior geometry enumeration — only steps 4-7 re-run.

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

## Related projects

`pydefect-complex` sits in a small ecosystem of defect-simulation tools.
Knowing what's upstream, downstream, and parallel helps you pick the right
piece for each stage.

**Upstream (this project reads):**
- [pydefect](https://github.com/kumagai-group/pydefect) — `supercell_info.json` (POSCAR + sites + space group), `SimpleDefect` definitions, `DefectSetMaker` for the canonical single-defect list.
- [pymatgen](https://pymatgen.org) — `Structure` / `Lattice` / `PointGroupAnalyzer` (point-group classification via pymatgen, not spglib directly).
- [spglib](https://spglib.github.io/spglib/) — `get_symmetry` for space-group rotations used in orientation counting.
- [scipy](https://scipy.org) — `KDTree` for coordinate lookup.

**Downstream (this project writes):**
- [pydefect](https://github.com/kumagai-group/pydefect) again — `complex_defect_in.yaml` is consumed by `pydefect efnv` (formation energy with corrections), `pydefect des` (defect structure analysis), and `pydefect pe` (phase diagram).
- [vise](https://github.com/kumagai-group/vise) — the GUI / workflow layer above pydefect; also reads `complex_defect_in.yaml`.

**Parallel / alternative tools** (different inputs, may suit different needs):
- [pymatgen-analysis-defects](https://github.com/materialsproject/pymatgen-analysis-defects) — official pymatgen extension, focuses on single-point defects; integrates with the Materials Project API for charge corrections.
- [doped](https://github.com/SMTG-UCL/doped) (SMTG-UCL) — auto charge-state estimation and thermodynamic analysis. If you need pydefect to *guess* sensible charge states, `doped` can supply them as the `--charges` argument here.
- [PyCD](https://github.com/WMD-group/PyCD) — another complex-defect generator with different enumeration strategy (genetic algorithm); useful for N≥4.

When choosing between these, the main axis is **what the input is**: if you
already run pydefect's `supercell` command, stay in this ecosystem and use
`pydefect-complex`. If you start from a Materials Project ID, look at
`pymatgen-analysis-defects` or `doped`.

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