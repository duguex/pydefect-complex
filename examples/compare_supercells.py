"""对比不同超胞大小的枚举结果和速度。

运行: python examples/compare_supercells.py
输出: diamond_example/（保留所有文件）
"""

import time
import json
from pathlib import Path

diamond_poscar = """diamond
1.0
    3.56679000000000    0.00000000000000    0.00000000000000
    0.00000000000000    3.56679000000000    0.00000000000000
    0.00000000000000    0.00000000000000    3.56679000000000
C
8
Direct
    0.00000000000000    0.00000000000000    0.00000000000000
    0.00000000000000    0.50000000000000    0.50000000000000
    0.50000000000000    0.00000000000000    0.50000000000000
    0.50000000000000    0.50000000000000    0.00000000000000
    0.25000000000000    0.25000000000000    0.25000000000000
    0.25000000000000    0.75000000000000    0.75000000000000
    0.75000000000000    0.25000000000000    0.75000000000000
    0.75000000000000    0.75000000000000    0.25000000000000
"""

from pymatgen.core import Structure
from pydefect.input_maker.supercell_info import SupercellInfo
from pydefect.input_maker.supercell_maker import SupercellMaker
from pydefect_complex import ComplexDefectMaker

structure = Structure.from_str(diamond_poscar, fmt="poscar")

# 用 SupercellMaker 生成不同尺寸超胞。
# diamond 只有特定立方超胞可用：2x2x2=64, 3x3x3=216, 4x4x4=512
configs = [
    ("2x2x2 (64 原子)",  60,  80),
    ("3x3x3 (216 原子)", 210, 230),
    ("4x4x4 (512 原子)", 510, 520),
]

results = []
for label, lo, hi in configs:
    sm = SupercellMaker(structure, max_num_atoms=hi, min_num_atoms=lo)
    sc_info = sm.supercell_info
    n_atoms = len(sc_info.structure)

    print(f"\n{'='*60}")
    print(f"{label}: 实际 {n_atoms} 原子")
    print(f"{'='*60}")

    maker = ComplexDefectMaker(sc_info, max_distance=4.0)

    # N=2
    t0 = time.perf_counter()
    e2 = maker.make_all_n_body(n=2)
    t2 = time.perf_counter()
    n_geom_2 = len(maker.enumerator.geometries.get(2, []))

    # N=3
    e3 = maker.make_all_n_body(n=3)
    t3 = time.perf_counter()
    n_geom_3 = len(maker.enumerator.geometries.get(3, []))

    print(f"N=2: {len(e2):4d} 条目, {n_geom_2:3d} 几何构型, {t2-t0:.2f}s")
    print(f"N=3: {len(e3):4d} 条目, {n_geom_3:3d} 几何构型, {t3-t2:.2f}s")
    print(f"总耗时: {t3-t0:.2f}s")

    results.append({
        "label": label,
        "n_atoms": n_atoms,
        "n_geom_2": n_geom_2,
        "n_entries_2": len(e2),
        "time_2_s": round(t2 - t0, 3),
        "n_geom_3": n_geom_3,
        "n_entries_3": len(e3),
        "time_3_s": round(t3 - t2, 3),
    })

# 汇总
print(f"\n{'='*70}")
print(f"汇总对比")
print(f"{'='*70}")
print(f"{'超胞':20s} {'原子':>5s}  {'N=2几何':>7s} {'N=2条目':>7s} {'N=2耗时':>8s}  {'N=3几何':>7s} {'N=3条目':>7s} {'N=3耗时':>8s}")
print(f"{'-'*20} {'-'*5}  {'-'*7} {'-'*7} {'-'*8}  {'-'*7} {'-'*7} {'-'*8}")
for r in results:
    print(f"{r['label']:20s} {r['n_atoms']:5d}  {r['n_geom_2']:7d} {r['n_entries_2']:7d} {r['time_2_s']:7.3f}s  {r['n_geom_3']:7d} {r['n_entries_3']:7d} {r['time_3_s']:7.3f}s")

# 保存
workdir = Path(__file__).parent / "diamond_output"
workdir.mkdir(exist_ok=True)
(workdir / "compare_supercells.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False))
print(f"\n结果已保存到 {workdir / 'compare_supercells.json'}")