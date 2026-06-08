# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

pydefect-complex — systematic complex (multi-component) defect generation for [pydefect](https://github.com/kumagai-group/pydefect). Generates N-body defect clusters (vacancy pairs, vacancy+dopant complexes, co-doping) with symmetry-aware site enumeration and distance filtering. Output is pydefect-compatible.

**Scope (in):** geometry enumeration (Apriori + Kabsch dedup), wyckoff-based composition assignment, structure generation for vacancies/substitutions/interstitials, cross-composition deduplication, orientation counting via spglib, pydefect-compatible file output.

**Scope (out):** DFT calculations, charge-state estimation (always defaults to `[0]`), structure relaxation, formation-energy correction, non-VASP DFT code support. Charge states are an *input* (`--charges`), never a *product*.

**Ecosystem position:** reads `supercell_info.json` from upstream pydefect; writes `complex_defect_in.yaml` + per-defect POSCAR dirs to be consumed by downstream pydefect (`efnv`/`des`/`pe`) or vise. See README.md "Related projects" for parallel tools (pymatgen-analysis-defects, doped, PyCD) and how to pick between them.

## Build & Test

```bash
pip install -e ".[dev]"
pytest                          # run all tests
pytest tests/test_core.py       # single test file
pytest -k "test_make_pair"      # run tests matching keyword
```

Tests require pydefect. The fixture `diamond_supercell_info` creates a diamond supercell via pydefect's `SupercellMaker` API (session-scoped, 100–300 atoms); tests skip if pydefect is not installed. `diamond_single_defects` and `tmp_output_dir` are also exposed (see `tests/conftest.py`).

Performance test thresholds: N=2 < 2s at 3.0 Å default cutoff, N=3 < 20s at 4.0 Å (see `TestEnumerator.test_performance_n2/n3` in `tests/test_maker.py`).

`tests/test_validate.py` doubles as a standalone CLI script for validating output directories:
```bash
python tests/test_validate.py /path/to/defect_output_dir
```

## Before implementing (mandatory)

**Always check existing libraries first.** The dependency stack already provides:

- **spglib**: space group info, point group symbols (`get_symmetry_dataset`)
- **pymatgen**: `SpacegroupAnalyzer`, `PointGroupAnalyzer`, `StructureMatcher`
- **pydefect**: site labels, wyckoff mapping (via `SupercellInfo.sites`)
- **scipy**: `KDTree`, `cdist`, spatial transforms

Do NOT reimplement functionality these libraries already provide. Only write custom code when the existing tool is genuinely insufficient (wrong abstraction, too slow, incompatible interface) — and document the reason in a code comment and the commit message.

## Architecture (graph-based, Apriori enumeration — PLAN-C)

```
ComplexDefectMaker          # main entry point (maker.py)
├── core.py                 # ComplexDefect data class — N SimpleDefect composition
├── graph.py                # HostGraph (crystal site registry) + ComplexDefectGraph (geometry) + equivalent()
├── enumerate.py            # ComplexDefectEnumerator + assign_compositions() + generate_all_entries() + generate_structure()
├── structure.py            # ComplexDefectEntry dataclass + count_defect_orientations() + space group/point group mappings
├── symmetry.py             # geometry-first cross-composition deduplication (deduplicate, verify_dedup)
├── io.py                   # pydefect-compatible file output (write_all, write_summary, merge_defect_in)
├── log.py                  # configure_logging() + get_logger() (call once at startup)
├── tracker.py              # PipelineTracker (offline stage dumps under pipeline_track/, opt-in)
└── cli.py                  # argparse entry point → `pydefect_complex` console script
```

**Pipeline**: Enumerate geometrically unique N-node site configurations → assign defect compositions by wyckoff label matching → generate structures → deduplicate → write output.

**Key design decisions**:

- `ComplexDefectGraph` is geometry-only (no defect types). Nodes carry (wyckoff, element) labels from the host crystal. Defect compositions are assigned after geometry enumeration by matching `out_atom` multisets to wyckoff labels.
- `generate_structure()` lives in `enumerate.py`, not `structure.py`. `structure.py` holds the `ComplexDefectEntry` dataclass, space-group→point-group lookup tables (`_SG_TO_SCHOENFLIES`, `_SG_TO_HM`), and `count_defect_orientations()`.
- The `ComplexDefectEnumerator` uses Apriori-style incremental building: N=2 base geometries (one anchor per (wyckoff, element) class → neighbor pairs) → extend unique k-geometries to k+1 via external neighbors → online `equivalent()` dedup. Results are cached — calling `enumerate(N_max=4)` reuses prior N=2,3 results.
- `equivalent()` (`graph.py`) tests geometric equivalence via node permutation within (wyckoff, element) groups + Kabsch optimal rotation of edge vectors. Used both online during enumeration and offline in `symmetry.deduplicate`.
- Deduplication (`symmetry.py`) is cross-composition: entries with geometrically equivalent defect site configurations are collapsed regardless of composition. Per-composition sequential indices are then assigned (e.g., `Va_C1+Va_C2.001`).
- `HostGraph` uses min-image distance for all pair computations (PBC-aware). The `find_node()` method uses KDTree for coordinate lookup. `neighbors()` uses brute-force search (O(N) per call); at typical supercell sizes (~200 atoms) this is microseconds.
- `ComplexDefectMaker` has two distance parameters: `max_distance` (edge inclusion cutoff) and `min_distance` (0.3 Å default, rejects too-close defect pairs). Both flow into `ComplexDefectEnumerator`.
- `make_all_n_body(n, ...)` uses `generate_all_entries()` which produces entries for **all orders 2..N**, then filters to exactly order `n`. If you call `make_all_n_body(n=2)` after `n=3`, the cached geometries are reused.
- `generate_all_entries()` accepts an `orders: set[int]` parameter to generate only specific orders. `make_all_n_body` uses this together with `_entry_cache` (a `dict[int, list[ComplexDefectEntry]]`) so that calling `n=4` after `n=2` only computes N=3 and N=4 — lower-order entries are returned from cache instantly.
- Entry cache is invalidated by `set_dopants()` (defect list changed) or by changing `max_distance`/`min_distance` (geometry changed). Geometry enumeration cache in `ComplexDefectEnumerator` is preserved across `set_dopants` calls.
- `count_defect_orientations()` (structure.py) applies pristine crystal space group rotations to defect centroid coordinates, maps back to host atoms via KDTree, and counts distinct orientation embeddings.

### pydefect naming convention

pydefect's `SimpleDefect` names their defect objects (e.g., `Va_C1` for a vacancy, `N_C1` for N substitution). However, `out_atom` returns the raw site label (e.g., `"C1"`), not the defect name. This is the value used in wyckoff matching — `assign_compositions()` compares `sorted(G.wyckoffs)` to `sorted(d.out_atom for d in combo)`.

Host graph wyckoff labels use the same convention: element symbol + a digit (e.g., `"C1"`, `"Si1"`). For interstitials, `out_atom` is `"i1"`, `"i2"`, etc.

Defect sorting in `ComplexDefect.__post_init__` is by `out_atom` (reverse, alphabetical) for determinism.

### Parameter flow

```
ComplexDefectMaker(max_distance, min_distance)
  └─ ComplexDefectEnumerator(max_distance, min_distance)
       └─ HostGraph.neighbors(max_distance)   — edge inclusion
       └─ min_distance check                  — rejection of too-close nodes
  └─ deduplicate(entries, host_graph, max_distance, eps=0.1)
  └─ generate_all_entries(enumerator, supercell_info, single_defects, N_max, eps=0.1)
```

**Top-level public exports** (`__init__.py`): `ComplexDefectMaker`, `ComplexDefect`, `ComplexDefectEntry`, `ComplexDefectGraph`, `HostGraph`, `ComplexDefectEnumerator`, `equivalent`, plus `configure_logging` and `get_logger` from `log.py` (call `configure_logging()` once at startup or internal log messages are silently dropped).

## Parallel execution

Geometry extension (`ComplexDefectEnumerator._extend_order`) and entry generation (`generate_all_entries`) both use `ProcessPoolExecutor` when `n_workers > 1`; the entry-generation path additionally requires `len(pairs) >= n_workers`. The CLI flag `--workers` defaults to `0` (= auto = `os.cpu_count()`); pass an explicit integer to override.

## CLI behavior worth knowing

- `pydefect_complex` reads `supercell_info.json` from **CWD** and writes to `defect/`. There is no `--output-dir` flag — change directory instead.
- Re-running with the same `-d/-n/--max-distance/--min-distance` is *additive*: new entries are appended to `defect/complex_defect_in.yaml` (and `--structures` POSCAR dirs are written for them), but entries from a prior run with **different** parameters are NOT removed and their POSCAR dirs are NOT cleaned up. To start fresh, delete `defect/` first.
- Without `--structures`, only the registry YAML + summary + parameters are written — no per-defect `POSCAR` directories.
