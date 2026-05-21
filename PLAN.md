# N≥3 复杂缺陷扩展方案

## 核心数据结构

### 图表示

```
复合缺陷 = 带节点标签 + 边向量的完全图 G = (V, E)

V = {(defect_type, element), ...}
    例: {(Va, C), (Va, C), (N, C)}

E = {v_ij ∈ R³ | 1 ≤ i < j ≤ N}
    v_ij = 笛卡尔 min-image 位移向量 (单位: Å)
    从缺陷超胞结构中两缺陷中心的坐标差计算

此表示不依赖超胞选择、坐标原点、节点排列。
```

### 图同构判定

```
G₁ ≅ G₂  ⇔  存在节点排列 π 和旋转 R ∈ SO(3), 使得:
             对每条边 (i,j):  |R·v_ij(G₁) - v_π(i)π(j)(G₂)| < ε

算法:
  1. 按节点类型分组 → 约束有效排列空间
  2. 对每种有效排列 π:
     a. 取 G₁ 边向量集, 按 π 重标号 → {v'_k}
     b. Kabsch: 求最优 R = argmin Σ|R·v_k - v'_k|²
     c. 若 max|R·v_k - v'_k| < ε → 等价
  3. 全部排列失败 → 不等价
  
  复杂度: O(K × N²) 其中 K ≤ N! 是有效排列数
  容差 ε: 建议 ~0.05 Å (晶格常数不确定性)
```

### 数据层次

```
ComplexDefectGraph
├── nodes: [(defect_type, element), ...]
├── edges: [v_12, v_13, ..., v_N-1,N]  # 笛卡尔 min-image 向量
├── equivalent(G1, G2, eps) → bool      # 图同构判定
└── (内部用 Kabsch 求最优旋转)

ComplexDefect (成分包装)
├── defects: [SimpleDefect, ...]
├── name: "3Va_C1" | "Va_C1+2N_C1"    # 紧凑计数
├── elements: ['C', 'C', 'C']
└── charges: [0, 1, 2, ...]

ComplexDefectEntry (输出包装)
├── complex_defect: ComplexDefect
├── graph: ComplexDefectGraph
├── name: "3Va_C1.001"                # 去重后顺序编号
├── structure: IStructure
└── charge: int
```

### 命名

```
ComplexDefect.name      = "3Va_C1"          # 紧凑成分名
ComplexDefectEntry.name  = "3Va_C1.001"      # 成分.去重后序号
目录名                   = "3Va_C1.001_-1"   # entry.name_charge
```

### 去重流程

```
entries = generate_entries(...)   # 生成所有 N 体构型

unique = [entries[0]]
for e in entries[1:]:
    if not any(graph.equivalent(e.graph, u.graph, eps) for u in unique):
        unique.append(e)

# 按 composition_name 分组编号
for group in group_by_composition(unique):
    for i, e in enumerate(sorted(group)):
        e.name = f"{e.complex_defect.name}.{i+1:03d}"
```

## 不变性

| 不变性 | 方法 |
|--------|------|
| 坐标原点 | 边是 min-image 相对向量 |
| 超胞选择 | 笛卡尔实空间向量 |
| 空间群旋转 | Kabsch 最优旋转匹配 |
| 节点排列 | 枚举排列 |
| 晶格常数微扰 | 容差 ε |

## 实施步骤

### Phase 1: ComplexDefectGraph (新文件 graph.py)
- `defect_centers(entry, pristine)` → 缺陷中心坐标
- `edge_vectors(centers, lattice)` → C(N,2) 个 min-image 笛卡尔向量
- `from_entry(entry, supercell_info)` → 构造图
- `equivalent(graph1, graph2, eps)` → 图同构判定 (Kabsch + 排列枚举)

### Phase 2: ComplexDefectEntry (structure.py)
- 新增 graph 字段
- 更新 name = f"{compact_name}.{index:03d}"

### Phase 3: ComplexDefect 命名 (core.py)
- `_composition_name(defects)` → 紧凑计数格式

### Phase 4: 去重 (symmetry.py)
- `deduplicate(entries, eps)` → 基于 graph.equivalent 去重 + 编号

### Phase 5: 测试
- N=3 全流程
- 图同构判定正确性 (相同/不同构型)
- 容差行为