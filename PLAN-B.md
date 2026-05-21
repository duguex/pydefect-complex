# 做法 B：图优先的复合缺陷生成

## 数据模型

### 母体图 HostGraph
```
节点 = 超胞中每个原子
  属性: id, wyckoff, element, frac_coord
  功能: 坐标 → 节点ID 查找 (KDTree)
  不预计算边
```

### 复合缺陷图 ComplexDefectGraph (纯几何)
```
节点 = (host_node_id, wyckoff, element)   ← 来自母体晶格，无缺陷类型
边   = { (i, j, v_ij) | |v_ij| ≤ max_distance }
       v_ij = min-image 笛卡尔位移向量
```

### 图同构判定 equivalent()
```
G1 ≅ G2 ⇔ 存在节点排列 π + 旋转 R ∈ SO(3):
  相同 (wyckoff, element) 标签的节点可互换
  每条边的 |R·v_ij(G1) - v_π(i)π(j)(G2)| < ε
```

### 成分 ComplexDefect
```
节点标签 = (defect_type, element)
name = 紧凑计数格式: "3Va_C1", "Va_C1+2N_C1"
```

## 流程

```
Phase 1: 几何枚举
  HostGraph → 枚举 N 节点子图（距离约束 + 对称性剪枝）
  → 去重（几何等价）
  → 唯一几何簇 {G_k}

Phase 2: 成分分配
  对每个唯一几何 G_k:
    节点有 (wyckoff, element) 标签
    → 检查哪些缺陷成分可以分配到这些节点上
    → 生成 (几何, 缺陷类型) 配对
    
Phase 3: 结构生成
  对每个有效配对:
    按节点位置移除/替换原子 → defected_structure

Phase 4: 命名
  ComplexDefectEntry.name = "{composition}.{index:03d}"
  目录名 = "{name}_{charge}"
```

## 不变性

| 不变性 | 方法 |
|--------|------|
| 坐标原点 | min-image 相对向量 |
| 超胞选择 | 超胞分数坐标 + KDTree 映射 |
| 空间群旋转 | Kabsch 最优旋转 |
| 节点排列 | (wyckoff, element) 组内枚举 |
| 晶格常数微扰 | 容差 ε |
| 截断半径 | 边距离 ≤ max_distance |

## 当前状态

- [x] HostGraph (坐标注册表)
- [x] ComplexDefectGraph (纯几何图)
- [x] equivalent() (Kabsch + wyckoff 标签排列)
- [x] deduplicate() (跨成分几何聚类)
- [x] 紧凑成分命名
- [ ] 几何枚举 (enumerate.py — WIP, N=3 性能优化)
- [ ] 成分分配逻辑
- [ ] 结构生成
- [ ] 完整 N=3 测试