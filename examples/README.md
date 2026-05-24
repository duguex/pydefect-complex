Examples
=======

Prerequisites
-------------
Run pydefect's supercell generation first:

```bash
pydefect supercell -p POSCAR --matrix 3 3 3
```

This creates ``supercell_info.json`` in the current directory.


CLI workflow
------------
All commands read ``supercell_info.json`` from the current directory.

1. Generate intrinsic vacancy pairs (no dopants):

```bash
pydefect_complex -n 2
```

2. Generate N+B doped complexes, order 2..3:

```bash
pydefect_complex -d N B -n 3
```

Output layout::

    defect/
      complex_defect_in.yaml     # machine-readable defect registry
      defect_summary.txt         # human-readable summary
      parameters.yaml            # run metadata
      2B_C1.001_0/POSCAR         # VASP structure per (name, charge)
      2B_C1.001_0/prior_info.yaml
      2B_C1.001_0/defect_entry.json
      ...

3. Append different dopants (geometry cache preserved, no re-enumeration):

```bash
pydefect_complex -d P -n 2
```

4. Append higher order (existing entries skipped):

```bash
pydefect_complex -d N B -n 3
```


Pre-generated output
--------------------
``diamond_output/`` contains the output of running the above commands
on a 3x3x3 diamond supercell (216 atoms, space group Fd-3m, a=3.567 Å)
with dopants N and B:

```bash
cd examples/diamond_output
pydefect_complex -d N B -n 2
pydefect_complex -d N B -n 3
```
