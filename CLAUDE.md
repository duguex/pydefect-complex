# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

pydefect-complex — systematic complex (multi-component) defect generation for [pydefect](https://github.com/kumagai-group/pydefect). Generates N-body defect clusters (vacancy pairs, vacancy+dopant complexes, co-doping) with symmetry-aware site enumeration and distance filtering. Output is pydefect-compatible.

## Build & Test

```bash
pip install -e ".[dev]"
pytest                          # run all tests
pytest tests/test_core.py       # single test file
pytest -k "test_make_pair"      # run tests matching keyword
```

Tests require pydefect. The fixture `diamond_supercell_info` creates a diamond supercell via pydefect's `SupercellMaker` API; tests skip if pydefect is not installed.

## Architecture (graph-based, Apriori enumeration — PLAN-C)

```
ComplexDefectMaker          # main entry point (maker.py)
├── core.py                 # ComplexDefect data class — N SimpleDefect composition
├── graph.py                # HostGraph (crystal site registry) + ComplexDefectGraph (geometry)
├── enumerate.py            # ComplexDefectEnumerator — Apriori incremental geometry enumeration
├── structure.py            # ComplexDefectEntry dataclass + generate_structure()
├── symmetry.py             # geometry-first deduplication (Kabsch SO(3) alignment)
└── io.py                   # pydefect-compatible file output
```

**Pipeline**: Enumerate geometrically unique N-node site configurations → assign defect compositions by wyckoff label matching → generate structures → deduplicate → write output.

**Key design decisions**:

- `ComplexDefectGraph` is geometry-only (no defect types). Defect compositions are assigned after geometry enumeration by matching wyckoff label multisets.
- The `ComplexDefectEnumerator` uses Apriori-style incremental building: N=2 base geometries → extend unique k-geometries to k+1 → online `equivalent()` dedup. Results are cached — calling `enumerate(N_max=4)` reuses prior N=2,3 results.
- `equivalent()` (graph.py:227) tests geometric equivalence via node permutation within (wyckoff, element) groups + Kabsch optimal rotation of edge vectors.
- Deduplication (`symmetry.py`) is cross-composition: entries with geometrically equivalent defect site configurations are collapsed regardless of composition. Per-composition indices are then assigned.
- `HostGraph` uses min-image distance for all pair computations (PBC-aware). The `find_node()` method uses KDTree for coordinate lookup.
- pydefect naming convention: vacancies are `Va_C1`, substitutions are `N_C1`, interstitials are `i1`/`i2`. Defect sorting in `ComplexDefect.__post_init__` is by `out_atom` (reverse) for determinism.

**Top-level public exports** (`__init__.py`): `ComplexDefectMaker`, `ComplexDefect`, `ComplexDefectEntry`, `ComplexDefectGraph`, `HostGraph`, `ComplexDefectEnumerator`, `equivalent`.