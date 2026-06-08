MAO example
===========

Spinel-type MgAl₂O₄ (MAO) supercell, run with N+B co-doping at N=3.

Final run::

    pydefect_complex -d N B -n 3 --max-distance 4.0

Result: 22 entries in ``defect/complex_defect_in.yaml`` spanning vacancy
pairs (``Va_Mg1+Va_Mg1``, …) and N/B substitutions on Mg, Al, O sites.

Output layout
-------------
::

    defect/
      complex_defect_in.yaml     # defect registry (name → charge list)
      defect_summary.txt         # human-readable table
      parameters.yaml            # run metadata
      geometries_N2.yaml         # geometry cache

The defect subdirectories (``defect/2Al_O1.001_0/POSCAR`` etc.) were
generated via ``--structures`` and are committed for inspection.

Log
---
``pydefect_complex.log`` records every run with ``COMMAND:``, ``ARGS:``,
``CACHE HIT:``, ``GEOMETRY:``, ``ENTRIES:``, and ``OUTPUT:`` lines.
