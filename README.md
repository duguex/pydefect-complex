# pydefect-complex

[pydefect](https://github.com/kumagai-group/pydefect) 的多组分复杂缺陷系统生成工具。支持生成 N 体缺陷团簇（空位对、空位+掺杂复合体、共掺杂构型），输出与 pydefect 完全兼容。

## 安装

```bash
pip install -e .
```

依赖：Python ≥ 3.10、pydefect。

## 快速开始

```python
from pydefect_complex import ComplexDefectMaker

# 从 pydefect 的 supercell_info.json 创建
maker = ComplexDefectMaker.from_supercell_info(
    "defect/supercell_info.json",
    dopants=["N", "B"],
    max_distance=4.0,
)

# 生成所有缺陷对
entries = maker.make_all_pairs()
print(f"生成 {len(entries)} 个复合缺陷条目")

# 写入 pydefect 兼容的目录结构
maker.write(entries, "defect")
```

写入后运行标准 pydefect 后处理流程：

```bash
cd defect
pydefect_vasp de                    # 生成 VASP 输入
# ... 提交计算 ...
pydefect efnv -d *_* -pcr perfect/calc_results.json -u ../unitcell/unitcell.yaml
pydefect des -d *_* -u ../unitcell/unitcell.yaml -pbes perfect/perfect_band_edge_state.json -t ../cpd/target_vertices.yaml
```

## API

### ComplexDefectMaker

主入口，与 pydefect 的 `DefectSetMaker` API 风格一致。

```python
maker = ComplexDefectMaker(supercell_info, dopants=["N"], max_distance=5.0)

# 或从文件创建：
maker = ComplexDefectMaker.from_supercell_info("supercell_info.json", dopants=["N"])

# 属性
maker.single_defects    # list[SimpleDefect]
maker.defect_names      # list[str]
maker.defect_pairs      # list[tuple[SimpleDefect, SimpleDefect]]
maker.host_graph        # HostGraph — 晶体格点注册表

# 生成指定缺陷对
entries = maker.make_pair("Va_C1", "N_C1", max_distance=4.0)

# 生成指定 N 体复合体
entries = maker.make_complex(["Va_C1", "N_C2", "Va_C3"])

# 生成所有二体缺陷对
entries = maker.make_all_pairs(max_distance=4.0)

# 生成所有 N 体复合缺陷（N ≥ 2），结果按阶缓存
entries_n2 = maker.make_all_n_body(n=2, max_distance=4.0)
entries_n3 = maker.make_all_n_body(n=3)  # 仅计算 N=3，N=2 直接复用缓存

# 枚举几何构型（缓存复用）
geometries = maker.enumerate_geometries(N_max=4)

# 更换掺杂种类（复用几何枚举缓存，无需重枚举）
maker.set_dopants(["P", "B"])
entries = maker.make_all_pairs()  # 直接使用已缓存的几何构型

# 写入
maker.write(entries, "defect", merge=True)  # merge=True 将合并 defect_in.yaml
```

### ComplexDefect

N 组分缺陷的数据模型。

```python
from pydefect_complex import ComplexDefect

cd = ComplexDefect.from_pair(defect1, defect2)
cd.name        # "Va_C1+Va_C2"
cd.charges     # [-2, -1, 0, 1]  （估算的电荷态）
cd.n_defects   # 2
cd.is_all_vacancies()   # True
```

### ComplexDefectEntry

带元数据的缺陷结构。

```python
entry.name         # "Va_C1+Va_C2.001" (去重后带编号)
entry.complex_defect  # ComplexDefect 对象
entry.distance     # 2.52 Å （缺陷对间距）
entry.distances    # (d_12, d_23, ...) — 链式距离
entry.structure    # pymatgen IStructure
entry.site_path    # ("C1", "C5") — 每个缺陷对应的格点
entry.defect_coords  # 缺陷中心分数坐标
entry.graph        # ComplexDefectGraph — 几何图
entry.point_group  # 点群（Schoenflies 符号）
entry.space_group  # 空间群（Hermann-Mauguin 符号）
entry.n_orientations  # 对称不等价取向数
```

## 算法（Plan C：图 + Apriori 增量枚举）

1. **枚举几何构型**：`ComplexDefectEnumerator` 采用 Apriori 式增量构建——从每个 Wyckoff 等价类代表格点出发，生成 N=2 的基元几何构型，逐层向外扩展到 k+1，在线去重。几何构型按阶缓存。
2. **分配缺陷组分**：将几何构型的 Wyckoff 标签与缺陷的 `out_atom` 标签做多重集匹配，为每个构型分配兼容的缺陷类型组合。
3. **生成结构**：按几何构型+缺陷组分逐层施加缺陷操作，生成超胞结构。**条目结果也按阶缓存**——计算 N=4 时，N=2 和 N=3 直接复用缓存，仅生成 N=4。
4. **对称性去重**：跨组分进行几何图等价性判定（节点置换 + Kabsch 最优旋转），去除几何上等价的条目，按组分内顺序编号。
5. **输出文件**：写入 pydefect 兼容的目录结构。

### 核心模块

```
ComplexDefectMaker          # 主入口 (maker.py)
├── core.py                 # ComplexDefect — N 体缺陷数据模型
├── graph.py                # HostGraph（晶体格点注册表）+ ComplexDefectGraph（几何图）
├── enumerate.py            # ComplexDefectEnumerator（Apriori 枚举）+ 组分分配 + 结构生成
├── structure.py            # ComplexDefectEntry + 取向计数 + 空间群/点群映射
├── symmetry.py             # 跨组分几何去重
└── io.py                   # pydefect 兼容文件输出
```

### 关键设计决策

- `ComplexDefectGraph` 仅存几何信息（格点 ID、Wyckoff 标签、元素、边），不包含缺陷类型。缺陷类型在几何枚举完成后通过 Wyckoff 标签匹配来分配。
- 几何枚举和条目生成均按阶缓存。先算 N=2、再算 N=3，N=2 直接从缓存返回（约 0.000s）；`set_dopants()` 仅清除条目缓存（几何缓存保留），距离参数变化则清空全部缓存。
- 去重是跨组分的：几何上等价的缺陷构型会被合并，不论其化学组分是否相同，然后为每个组分内的构型分配顺序编号。
- `HostGraph` 使用最小镜像约定（PBC 感知）计算所有原子对距离，`find_node()` 使用 KDTree 进行坐标查找。
- pydefect 命名约定：空位为 `Va_C1`，替位掺杂为 `N_C1`，间隙为 `i1`/`i2`。代码中 `out_atom` 返回的是原始格点标签（如 `"C1"`），这才是 Wyckoff 匹配实际使用的值。

## 输出结构

```
defect/
├── Va_C1+Va_C2.001_-2/
│   ├── POSCAR
│   ├── prior_info.yaml
│   └── defect_entry.json
├── Va_C1+Va_C2.001_-1/
│   └── ...
├── Va_C1+N_C2.002_0/
│   └── ...
├── complex_defect_in.yaml
└── defect_in.yaml          # (merge=True 时生成)
```

## 测试

```bash
pip install -e ".[dev]"
pytest                          # 运行所有测试
pytest tests/test_core.py       # 单文件
pytest -k "test_make_pair"      # 按关键字筛选

# 验证输出目录
python tests/validate.py /path/to/defect_output_dir
```

## 许可

MIT