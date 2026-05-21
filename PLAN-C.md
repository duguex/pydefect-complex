# Plan C: Incremental Enumeration of Complex Defect Geometries

## 核心思想

**枚举 N 最大意味着顺便拿到了所有 < N 的几何。**
出发自一个简单事实：DFS 逐层扩展连通子图时，深度 k 的中间态就是 k 节点的几何。

因此设计原则：
1. 一次扫描出 2..N_max 所有 order 的几何
2. Apriori 式增量构建：k-order 唯一几何 → 扩展邻居 → (k+1)-order 候选
3. **在线几何去重**——生成 (k+1) 候选时立即做 `equivalent()`，等价的直接丢
4. 几何枚举完后，再统一分配缺陷成分

## 架构对比

```
旧 (Plan B)                              新 (Plan C)
─────────────────────                    ─────────────────────
maker.make_all_n_body(n=2)              enumerator.enumerate(N_max=3)
  → enumerate_sites() DFS                 → 一次 scan
  → 生成所有 entry                      → 输出:
  → deduplicate() 最后去重                  geometries[2] = [G₁, G₂, ...]
                                          geometries[3] = [G₁, G₂, ...]
maker.make_all_n_body(n=3)              assign_compositions(geometries[3], specs)
  → enumerate_sites() 重新 DFS              → 成分 → 结构
  → 生成所有 entry 
  → deduplicate() 最后去重
```

**关键区别**：
- 旧：每个 N 单独枚举，各自从头 DFS → 重复计算 + 最后才去重 → 中间状态全浪费
- 新：一次扫描，中间态自动收集为低阶几何 → Apriori 增量避免重复 DFS → 在线剪枝

## 数据结构

### 1. HostGraph 增强

现有 `HostGraph` 只提供 KDTree 查找，需要增加**邻居查询**能力：

```python
@dataclass
class HostGraph:
    nodes: list[HostNode]
    lattice: np.ndarray
    
    def neighbors(self, node_id: int, max_distance: float) -> list[int]:
        """返回 max_distance 内的所有邻居节点 ID."""
        # 利用周期性 + KDTree 实现
    
    def neighbors_of_set(self, node_ids: set[int], max_distance: float) -> set[int]:
        """已有节点集合的外部邻居."""
```

### 2. ComplexDefectGraph (不变)

现有结构已足够：
- `host_node_ids`: N 个宿主节点 ID
- `wyckoffs`, `elements`: 节点标签
- `edges`: 距离 ≤ max_d 的边列表
- `equivalent()` 不改

### 3. 枚举器

```python
@dataclass
class ComplexDefectEnumerator:
    host_graph: HostGraph
    max_distance: float = 5.0
    min_distance: float = 0.3
    
    def enumerate(self, N_max: int, eps: float = 0.1) -> dict[int, list[ComplexDefectGraph]]:
        """返回 {2: [...], 3: [...], ..., N_max: [...]}"""
```

## 算法

### Phase 1: N=2 基元几何

```
geometries[2] = []

# 用 wyckoff 类别选代表锚点（避免对称等价锚点重复枚举）
anchor_classes = set()
for node in host_graph.nodes:
    if (node.wyckoff, node.element) in anchor_classes:
        continue
    anchor_classes.add((node.wyckoff, node.element))
    
    for nbr_id in host_graph.neighbors(node.id, max_distance):
        nbr = host_graph.nodes[nbr_id]
        G = ComplexDefectGraph(
            host_node_ids=(node.id, nbr_id),
            wyckoffs=(node.wyckoff, nbr.wyckoff),
            elements=(node.element, nbr.element),
            edges=_edge_list([node, nbr], host_graph, max_distance),
        )
        if min_distance ≤ |vec| ≤ max_distance (overlap check):
            if not any(equivalent(G, g, eps) for g in geometries[2]):
                geometries[2].append(G)
```

### Phase 2: Apriori 增量扩展

```
for k in range(2, N_max):
    geometries[k+1] = []
    
    for G_k in geometries[k]:              # 只用唯一几何扩展
        ext_neighbors = host_graph.neighbors_of_set(
            set(G_k.host_node_ids), max_distance
        )
        
        for nbr_id in ext_neighbors:
            nbr = host_graph.nodes[nbr_id]
            
            # 距离检查（只检查与新节点的最小距离）
            too_close = any(
                min_image_dist(nbr, host_graph.nodes[hid]) < min_distance
                for hid in G_k.host_node_ids
            )
            if too_close:
                continue
            
            G_next = build_graph(G_k, nbr_id, host_graph, max_distance)
            
            # 在线去重
            if not any(equivalent(G_next, g, eps) for g in geometries[k+1]):
                geometries[k+1].append(G_next)
```

### Phase 3: 成分分配

```
def assign_compositions(
    geometries: list[ComplexDefectGraph],
    single_defects: list[SimpleDefect],
) -> list[tuple[ComplexDefectGraph, ComplexDefect]]:
    """对每个几何，找出所有可以分配上去的缺陷成分组合."""
    
    results = []
    for G in geometries:
        # G 的节点标签 = [(wyckoff, element), ...]
        # 按 wyckoff 分组: {wyckoff: count}
        node_counts = Counter(zip(G.wyckoffs, G.elements))
        
        for combo in itertools.combinations_with_replacement(single_defects, G.n_defects):
            # combo 的 out_atom 需求: {wyckoff: count}
            combo_counts = Counter(d.out_atom for d in combo)
            
            # 检查是否兼容 (wyckoff 标签匹配)
            if compatible(node_counts, combo_counts):
                cd = ComplexDefect.from_defects(list(combo))
                results.append((G, cd))
    
    return results
```

这里 `compatible` 的语义是：几何提供了一组 `(wyckoff, element)` 位置，缺陷成分需要一组 `out_atom` (即 wyckoff 标签)。当集合一致时（考虑置换），该成分可以分配到这个几何上。

### Phase 4: 结构生成 (复用现有 generate_structure)

```
对每个 (G, cd):
    entry = generate_structure(host_graph, supercell_info, G, cd)
    → ComplexDefectEntry
```

### Phase 5: 命名 & 输出

```
# 成分内编号：同一 composition 的不同几何按出现顺序编号
for composition, entries in group_by_composition(all_entries):
    for i, e in enumerate(entries):
        e.name = f"{composition}.{i+1:03d}"
```

## 复杂度分析

以钻石 216 原子超胞、max_distance=5Å（每节点 ~16 近邻）为例：

| 阶段 | 操作 | 无去重 | 有在线去重 |
|------|------|--------|-----------|
| N=2 枚举 | 锚点 × 邻居 | ~30×16 = 480 对 | → ~15-30 唯一天几何 |
| N=2→3 扩展 | 30 × 16 × 15/2 | ~3,600 候选 | → ~20-80 唯一天几何 |
| N=3→4 扩展 | 80 × 20 × 18/3 | ~9,600 候选 | → ~50-200 唯一天几何 |

在线去重的效果：因为晶体对称性，**实际唯一几何数远小于候选数**（通常 5-20 倍压缩）。而且 Apriori 避免了每次从锚点重数，中间态自然产出低阶几何。

## API 设计

```python
from pydefect_complex import ComplexDefectMaker

# 新 API（向后兼容旧 API）
maker = ComplexDefectMaker.from_supercell_info(
    "supercell_info.json",
    dopants=["N", "B"],
    max_distance=4.0,
)

# === 方式 1: 一次性枚举所有 order ===
geometries = maker.enumerate_geometries(N_max=3)
# → {2: [ComplexDefectGraph, ...], 3: [ComplexDefectGraph, ...]}

# 对 N=3 分配成分并生成
entries = maker.assign_and_generate(geometries[3])
maker.write(entries, "defect")

# 也可以手动操作
for G in geometries[3]:
    print(f"{G.host_node_ids}  {G.wyckoffs}  {len(G.edges)} edges")

# === 方式 2: 旧 API 不改（内部改用新枚举器） ===
entries_2 = maker.make_all_pairs()       # 内部调 enumerate(N_max=2)
entries_3 = maker.make_all_n_body(n=3)   # 内部调 enumerate(N_max=3)
# 两者共享缓存，n=3 复用 n=2 的结果
```

## 实现阶段

### Step 1: HostGraph.neighbors() 增强
- `host_graph.neighbors(node_id, max_distance)` — KDTree + min-image
- `host_graph.neighbors_of_set(node_ids, max_distance)` — 集合外部邻居
- 测试：钻石超胞中每个 C 有 4 NN + 12 2NN

### Step 2: ComplexDefectEnumerator.enumerate()
- N=2 基元生成 + 在线去重
- Apriori 增量 k → k+1
- 缓存：`_cache = {2: [...], 3: [...]}`，调用 N_max 更大时复用缓存
- 测试：N=2 几何数 vs 预期（钻石 N=2: ~15-30）

### Step 3: assign_compositions()
- wyckoff 标签匹配 + 组合生成
- 测试：为钻石 N=3 几何分配 {Va_C1, N_C1, B_C1} 成分

### Step 4: 集成到 ComplexDefectMaker
- `make_all_n_body()` 改用新枚举器
- 旧 `make_all_pairs()` / `make_pair()` 保持不变
- 旧 `_recurse_sym` / `enumerate_sites` 标记 deprecated

### Step 5: 测试 & 验证
- 钻石 N=2/3 全流程
- 用 `validate.py` 验证生成的结构合理性
- 性能基准：N=2 < 1s, N=3 < 5s, N=4 < 30s (目标)

## 不变性保证

| 不变性 | 方法 | 阶段 |
|--------|------|------|
| 超胞/原点 | min-image 相对向量 | graph.py: edges |
| 空间群旋转 | Kabsch 最优旋转 | graph.py: equivalent() |
| 节点排列 | (wyckoff, element) 排列枚举 | graph.py: equivalent() |
| 几何唯一性 | 在线 equivalent() 剪枝 | 枚举阶段 |
| 成分兼容性 | wyckoff 计数匹配 | 分配阶段 |