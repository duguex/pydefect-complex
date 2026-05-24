Examples
=======

Prerequisite
------------
Create ``supercell_info.json`` using pydefect:

```bash
pydefect supercell -p POSCAR --matrix 3 3 3
```


4-step workflow (geometry skeleton → complex defects)
------------------------------------------------------
All commands run in the directory containing ``supercell_info.json``.
Geometry cache (``defect/geometries_N*.yaml``) is written each run and
automatically loaded by the next.

### 1. N=2 geometry skeleton (pure geometry, no chemistry)

```bash
pydefect_complex -g -n 2
```

Output: ``defect/geometries_N2.yaml`` — 7 unique N=2 geometries.

### 2. N=2 + B complex defects (geometry cached, no re-enumeration)

```bash
pydefect_complex -d B -n 2
```

Output: ``defect/complex_defect_in.yaml`` — 21 entries (3 compositions × 7 geometries).

### 3. N=3 geometry skeleton (N=2 cache reused)

```bash
pydefect_complex -g -n 3
```

Output: ``defect/geometries_N3.yaml`` — 109 unique N=3 geometries.

### 4. N=3 + O complex defects (N=2+N=3 cache reused)

```bash
pydefect_complex -d O -n 3
```

Output: ``defect/complex_defect_in.yaml`` — 471 total entries (21+450 O-doped).


Output layout
-------------
::

    defect/
      complex_defect_in.yaml     # defect registry (name → charge list)
      defect_summary.txt         # human-readable table (point group, orientations)
      parameters.yaml            # run metadata
      geometries_N2.yaml         # geometry cache (cross-process)
      geometries_N3.yaml         # geometry cache (cross-process)

Use ``--structures`` to generate per-defect POSCAR directories:

```bash
pydefect_complex -d O -n 3 --structures
# → defect/2O_C1.001_0/POSCAR
# → defect/2O_C1.001_0/prior_info.yaml
# → defect/2O_C1.001_0/defect_entry.json
# → ...
```

Log
---
``pydefect_complex.log`` records every run with ``COMMAND:``, ``CACHE HIT:``,
``GEOMETRY:``, ``ENTRIES:``, and ``OUTPUT:`` lines — no external echo needed.
