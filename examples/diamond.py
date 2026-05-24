"""Diamond 复合缺陷生成示例 (Plan C: Apriori 增量枚举)

运行: python examples/diamond.py
输出: diamond_example/
  ├── cache_geometry_N2.json    # 几何缓存 — N=2 唯一构型
  ├── cache_geometry_N3.json    # 几何缓存 — N=3 唯一构型
  ├── cache_entries_N2.json     # 条目缓存 — N=2 条目元数据
  ├── cache_entries_N3.json     # 条目缓存 — N=3 条目元数据
  └── defect/                   # pydefect 兼容目录（POSCAR 等）

依赖: pip install pydefect pydefect-complex
"""

import os, time
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 准备 diamond 超胞
# ---------------------------------------------------------------------------

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
from pydefect.input_maker.supercell_maker import SupercellMaker
from pydefect_complex import ComplexDefectMaker

workdir = Path(__file__).parent / "diamond_output"
workdir.mkdir(exist_ok=True)
(workdir / "POSCAR").write_text(diamond_poscar)

structure = Structure.from_file(str(workdir / "POSCAR"))
supercell_info = SupercellMaker(
    structure, max_num_atoms=300, min_num_atoms=100
).supercell_info

print(f"超胞原子数: {len(supercell_info.structure)}")
# 超胞原子数: 216
print(f"Wyckoff 格点: {list(supercell_info.sites.keys())}")
# Wyckoff 格点: ['C1']

# ---------------------------------------------------------------------------
# 1. 初始缓存状态（全空）
# ---------------------------------------------------------------------------

maker = ComplexDefectMaker(supercell_info, dopants=["N", "B"], max_distance=4.0)

print(f"缺陷类型: {maker.defect_names}")
# 缺陷类型: ['B_C1', 'Va_C1', 'N_C1']

print(f"几何缓存: {dict((k, len(v)) for k, v in maker.enumerator.geometries.items())}")
# 几何缓存: {}

print(f"条目缓存: {dict((k, len(v)) for k, v in maker.entry_cache.items())}")
# 条目缓存: {}
#            ↑ 两个缓存都为空

# ---------------------------------------------------------------------------
# 2. 几何缓存 — dict[int, list[ComplexDefectGraph]]
#
#    几何构型是纯几何信息，不含缺陷类型。节点用 (wyckoff, element) 标识，
#    边是 3D 向量（Å，PBC 感知的最小镜像距离）。
# ---------------------------------------------------------------------------

t0 = time.perf_counter()
maker.make_all_n_body(n=2)
entries_n2 = maker.entry_cache[2]
t1 = time.perf_counter()

print(f"N=2 耗时: {t1 - t0:.2f}s")
# N=2 耗时: 0.34s

# 几何缓存: {2: [G0, G1, G2, G3, G4]}
geoms_2 = maker.enumerator.geometries[2]
print(f"几何缓存 [N=2]: {len(geoms_2)} 个唯一几何构型")
# 几何缓存 [N=2]: 5 个唯一几何构型

for i, g in enumerate(geoms_2[:3]):
    print(f"  {g}")
#   ComplexDefectGraph(host_node_ids=(0, 1),
#     wyckoffs=('C1', 'C1'), elements=('C', 'C'),
#     edges=[(0, 1, array([ 0.   ,  0.   , -3.567]))])    ← 第一近邻方向
#   ComplexDefectGraph(host_node_ids=(0, 27),
#     wyckoffs=('C1', 'C1'), elements=('C', 'C'),
#     edges=[(0, 1, array([-1.783, -1.783,  0.   ]))])    ← 第二近邻方向
#   ComplexDefectGraph(host_node_ids=(0, 108),
#     wyckoffs=('C1', 'C1'), elements=('C', 'C'),
#     edges=[(0, 1, array([-0.892, -0.892, -0.892]))])    ← 第三近邻方向

# ---------------------------------------------------------------------------
# 3. 条目缓存 — dict[int, list[ComplexDefectEntry]]
#
#    条目 = 几何构型 + 缺陷组分 + 实际超胞结构 + 元数据。已去重编号。
# ---------------------------------------------------------------------------

print(f"条目缓存 [N=2]: {len(maker.entry_cache[2])} 个条目")
# 条目缓存 [N=2]: 30 个条目

for e in maker.entry_cache[2][:4]:
    print(f"  {e.name:30s}  site={str(e.site_path):20s}  d={e.distance:.2f}Å  "
          f"atoms={len(e.structure)}")
#   2B_C1.001                      site=('C1', 'C1')         d=3.57Å  atoms=216
#   2B_C1.002                      site=('C1', 'C1')         d=2.52Å  atoms=216
#   2B_C1.003                      site=('C1', 'C1')         d=1.54Å  atoms=216
#   2B_C1.004                      site=('C1', 'C1')         d=2.96Å  atoms=216

e = maker.entry_cache[2][0]
print(f"\n  name             = {e.name}")
print(f"  complex_defect   = {e.complex_defect}")
print(f"  site_path        = {e.site_path}")
print(f"  distances        = {e.distances}")
print(f"  defect_coords    = {e.defect_coords}")
print(f"  structure.formula= {e.structure.composition.formula}")
print(f"  graph.n_defects  = {e.graph.n_defects}")
print(f"  n_orientations   = {e.n_orientations}")
print(f"  point_group      = {e.point_group}")
print(f"  space_group      = {e.space_group}")
#   name             = 2B_C1.001
#   complex_defect   = ComplexDefect('2B_C1', charges=[0])
#   site_path        = ('C1', 'C1')
#   distances        = (3.567,)
#   defect_coords    = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5))
#   structure.formula= B2 C214
#   graph.n_defects  = 2
#   n_orientations   = 1
#   point_group      = D2h
#   space_group      = Pmmm

# ---------------------------------------------------------------------------
# 4. 增量计算 N=3 — N=2 完全从缓存复用
# ---------------------------------------------------------------------------

print(f"当前缓存阶: {sorted(maker.entry_cache.keys())}")
# 当前缓存阶: [2]

t0 = time.perf_counter()
maker.make_all_n_body(n=3)
entries_n3 = maker.entry_cache[3]
t1 = time.perf_counter()

print(f"N=3 耗时: {t1 - t0:.2f}s")
# N=3 耗时: 5.82s
print(f"N=3 几何构型: {len(maker.enumerator.geometries[3])} 个")
# N=3 几何构型: 42 个
print(f"N=3 条目数: {len(entries_n3)}")
# N=3 条目数: 420

# N=2 仍可即时获取
t0 = time.perf_counter()
_ = maker.make_all_n_body(n=2)
t1 = time.perf_counter()
print(f"N=2 缓存命中耗时: {t1 - t0:.4f}s")
# N=2 缓存命中耗时: 0.0000s

# ---------------------------------------------------------------------------
# 4.5 序列化缓存到 JSON（在换掺杂之前，保留完整 N=2 + N=3）
# ---------------------------------------------------------------------------

import json


def _entry_to_dict(e):
    return {
        "name": e.name,
        "complex_defect": {
            "name": e.complex_defect.name,
            "charges": e.complex_defect.charges,
            "components": [d.name for d in e.complex_defect.defects],
        },
        "site_path": list(e.site_path),
        "distances": list(e.distances),
        "defect_coords": [list(c) for c in e.defect_coords],
        "formula": str(e.structure.composition.formula) if e.structure else None,
        "n_atoms": len(e.structure) if e.structure else 0,
        "graph_n_defects": e.graph.n_defects if e.graph else 0,
        "n_orientations": e.n_orientations,
        "point_group": e.point_group,
        "space_group": e.space_group,
    }


for order, geoms in maker.enumerator.geometries.items():
    path = workdir / f"cache_geometry_N{order}.json"
    path.write_text(json.dumps(
        [g.to_dict() for g in geoms], indent=2, ensure_ascii=False))
    print(f"写入 {path.name} ({len(geoms)} 个构型)")

for order, entries in maker.entry_cache.items():
    path = workdir / f"cache_entries_N{order}.json"
    path.write_text(json.dumps(
        [_entry_to_dict(e) for e in entries], indent=2, ensure_ascii=False))
    print(f"写入 {path.name} ({len(entries)} 个条目)")

# ---------------------------------------------------------------------------
# 5. set_dopants — 仅清条目缓存，几何缓存保留
# ---------------------------------------------------------------------------

print(f"切换前: 几何缓存={dict((k, len(v)) for k, v in maker.enumerator.geometries.items())}")
print(f"        条目缓存={sorted(maker.entry_cache.keys())}")
# 切换前: 几何缓存={2: 5, 3: 42}
#         条目缓存=[2, 3]

maker.set_dopants(["P"])
print(f"缺陷类型: {maker.defect_names}")
# 缺陷类型: ['P_C1', 'Va_C1']

print(f"切换后: 几何缓存={dict((k, len(v)) for k, v in maker.enumerator.geometries.items())}  ← 保留")
print(f"        条目缓存={sorted(maker.entry_cache.keys())}  ← 已清空")
# 切换后: 几何缓存={2: 5, 3: 42}  ← 保留
#         条目缓存=[]  ← 已清空

t0 = time.perf_counter()
entries_p = maker.generate_entries(n_or_geometries=2)
t1 = time.perf_counter()
print(f"P 掺杂 N=2 耗时: {t1 - t0:.2f}s (仅组分分配+去重，无需几何枚举)")
# P 掺杂 N=2 耗时: 0.12s (仅组分分配+去重，无需几何枚举)
print(f"P 掺杂 N=2 条目数: {len(entries_p)}")
# P 掺杂 N=2 条目数: 15

# ---------------------------------------------------------------------------
# 6. 写入 pydefect 兼容目录
# ---------------------------------------------------------------------------

output_dir = workdir / "defect"
output_dir.mkdir(exist_ok=True)
maker.write(entries_n2, str(output_dir))
print(f"输出目录: {output_dir}/")
# 输出目录: diamond_example/defect/

n_dirs = sum(1 for d in output_dir.iterdir() if d.is_dir())
print(f"子目录数: {n_dirs}")
# 子目录数: 30

yaml_path = output_dir / "complex_defect_in.yaml"
print(f"\n--- {yaml_path.name} ---")
print(yaml_path.read_text()[:800])
# --- complex_defect_in.yaml ---
# 2B_C1.001: [0]
# 2B_C1.002: [0]
# ...
# Va_C1+N_C1.005: [0]

# 每个子目录的内容:
sample_dir = next(d for d in sorted(output_dir.iterdir()) if d.is_dir())
print(f"\n--- {sample_dir.name}/ ---")
for f in sorted(sample_dir.iterdir()):
    kind = type(f).__name__
    print(f"  {f.name}")
print(f"\n--- {sample_dir.name}/prior_info.yaml ---")
print((sample_dir / "prior_info.yaml").read_text())
# --- 2B_C1.001_0/prior_info.yaml ---
# charge: 0

print(f"--- {sample_dir.name}/POSCAR (前 8 行) ---")
print("".join((sample_dir / "POSCAR").read_text().splitlines(keepends=True)[:8]))

# ---------------------------------------------------------------------------
# 缓存层级总结
#
#   ┌─────────────────────────────────────────────────────┐
#   │ ComplexDefectEnumerator._cache                      │
#   │   dict[int, list[ComplexDefectGraph]]               │
#   │   纯几何: 节点ID、Wyckoff标签、元素、边向量           │
#   │   失效条件: 改变 max_distance / min_distance         │
#   │   保留条件: set_dopants() ← 与掺杂无关              │
#   └──────────────────┬──────────────────────────────────┘
#                      │ assign_compositions() + structure + dedup
#                      ▼
#   ┌─────────────────────────────────────────────────────┐
#   │ ComplexDefectMaker._entry_cache                     │
#   │   dict[int, list[ComplexDefectEntry]]               │
#   │   几何+组分+结构+元数据: name, structure, charges...  │
#   │   失效条件: set_dopants() 或 改变距离参数              │
#   └─────────────────────────────────────────────────────┘
#
#   缓存序列化为 JSON 文件:
#     cache_geometry_N{2,3}.json  — 几何构型（不含 ndarray）
#     cache_entries_N{2,3}.json   — 条目元数据（不含 pymatgen Structure）
# ---------------------------------------------------------------------------