# 匹配函数详解

## 📋 4种匹配函数对比

### 1. `l1` - 基础L1距离匹配

**函数**: `l1_dist_matching`

**原理**:
```python
# 直接计算 L1 距离
distance = sum(|hidden_state - vocab_hidden_state|)
```

**适用场景**:
- ✅ **无置换** (perm_type = "None")
- ✅ 最简单、最快的匹配方法
- ✅ 适合测试基础攻击

**工作方式**:
```python
# 目标隐藏状态
target = [1.2, 3.4, 2.1, 5.6]

# 词汇表中某个 token 的隐藏状态
vocab  = [1.3, 3.5, 2.0, 5.7]

# L1 距离
distance = |1.2-1.3| + |3.4-3.5| + |2.1-2.0| + |5.6-5.7|
         = 0.1 + 0.1 + 0.1 + 0.1 = 0.4
```

**特点**:
- ⚡ 速度最快
- 🎯 精度最高（无防御时）
- ❌ 不能处理任何置换

---

### 2. `l1_permuted` - 支持序列置换的L1匹配

**函数**: `l1_dist_matching_permuted`

**原理**:
```python
# 对所有可能的 (序列位置, token) 对计算距离
for each position in sequence:
    for each token in vocab:
        distance = L1(hidden_state[position], vocab[token])
# 找全局最小距离
```

**适用场景**:
- ✅ **序列置换** (perm_type = "S")
- ✅ 当 token 的顺序被打乱时

**工作方式**:
```python
# 假设有3个 token，被序列置换了
# 原始顺序: [A, B, C]
# 置换后:   [C, A, B]

# 目标隐藏状态（置换后）
target = [hidden_C, hidden_A, hidden_B]

# 匹配时：
# 位置0 (hidden_C) vs 词汇表所有token → 找到 C
# 位置1 (hidden_A) vs 词汇表所有token → 找到 A  
# 位置2 (hidden_B) vs 词汇表所有token → 找到 B

# 结果恢复: [C, A, B]
```

**特点**:
- 🔄 处理序列顺序打乱
- 🐌 比 l1 慢（需要全局搜索）
- ❌ 不能处理维度置换

---

### 3. `l1_sort` - 排序L1匹配（支持维度置换）⭐

**函数**: `l1_sort_dist_matching`

**原理**:
```python
# 先对向量排序，再计算 L1 距离
sorted_target = sort(hidden_state)
sorted_vocab = sort(vocab_hidden_state)
distance = L1(sorted_target, sorted_vocab)
```

**适用场景**:
- ✅ **维度置换** (perm_type = "D")
- ✅ 当向量的维度被打乱时
- ⭐ **当前使用的方法**

**工作方式**:
```python
# 原始向量
target = [5.6, 1.2, 3.4, 2.1]
vocab  = [1.3, 3.5, 2.0, 5.7]

# 维度置换后
# target 的维度被打乱（但我们不知道如何打乱的）
permuted_target = [3.4, 5.6, 2.1, 1.2]  # 维度重排

# 排序后对比
sorted_target = [1.2, 2.1, 3.4, 5.6]
sorted_vocab  = [1.3, 2.0, 3.5, 5.7]

# L1 距离（排序后）
distance = |1.2-1.3| + |2.1-2.0| + |3.4-3.5| + |5.6-5.7|
         = 0.1 + 0.1 + 0.1 + 0.1 = 0.4  ✅ 仍然很小！
```

**关键洞察**:
- 🔑 排序操作对置换不变！
- 🎯 即使维度被打乱，排序后的向量仍然相似
- 💡 这就是绕过维度置换防御的方法

**特点**:
- 🛡️ 能突破维度置换防御
- ⚡ 速度较快（只需排序）
- ❌ 不能处理序列置换

---

### 4. `l1_sort_permuted` - 全能匹配（支持所有置换）

**函数**: `l1_sort_dist_matching_permuted`

**原理**:
```python
# 结合排序和全局搜索
# 1. 对所有向量排序（处理维度置换）
# 2. 全局搜索最佳匹配（处理序列置换）
for each position:
    sorted_target = sort(hidden_state[position])
    for each token:
        sorted_vocab = sort(vocab[token])
        distance = L1(sorted_target, sorted_vocab)
```

**适用场景**:
- ✅ **所有置换类型**
  - "None" - 无置换
  - "S" - 序列置换
  - "D" - 维度置换
  - "SD" - 序列+维度置换
- ✅ 最强大但最慢的方法

**工作方式**:
```python
# 最复杂的情况：序列和维度都被置换

# 原始: [A, B, C]
# 序列置换: [C, A, B]
# 维度置换: 每个向量的维度也被打乱

# 攻击策略：
# 1. 对每个位置的向量排序（消除维度置换）
# 2. 全局搜索所有位置和token的组合（消除序列置换）
# 3. 找到全局最佳匹配
```

**特点**:
- 🛡️🛡️ 最强攻击能力
- 🐌🐌 最慢（排序 + 全局搜索）
- ✅ 处理所有防御组合

---

## 📊 对比表格

| 匹配函数 | 支持置换 | 速度 | 攻击能力 | 使用场景 |
|---------|---------|------|---------|---------|
| `l1` | None | ⚡⚡⚡ 最快 | ⭐⭐⭐ | 无防御测试 |
| `l1_permuted` | None, S | ⚡⚡ 快 | ⭐⭐ | 序列打乱 |
| `l1_sort` ⭐ | None, D | ⚡⚡ 快 | ⭐⭐⭐ | **维度置换**（当前） |
| `l1_sort_permuted` | None, S, D, SD | ⚡ 慢 | ⭐⭐⭐ | 所有防御 |

## 🎯 选择建议

### 场景 1: 测试基础攻击（无防御）
```python
dist_funct = "l1"
perm_type = "None"
```
- 速度最快
- 精度最高
- 验证攻击原理

### 场景 2: 攻击维度置换（SCX-MLP 防御）⭐ 当前使用
```python
dist_funct = "l1_sort"
perm_type = "D"
```
- 针对维度置换防御
- 使用排序绕过
- 速度和效果平衡

### 场景 3: 攻击序列置换
```python
dist_funct = "l1_permuted"
perm_type = "S"
```
- 针对序列打乱
- 全局搜索最佳匹配

### 场景 4: 攻击所有防御
```python
dist_funct = "l1_sort_permuted"
perm_type = "SD"
```
- 最强攻击能力
- 最慢但最全面

## 🔬 技术细节

### 为什么排序能绕过维度置换？

**数学原理**:
```
向量 v = [a, b, c, d]
置换 π 后: v' = [c, a, d, b]

排序后:
sort(v)  = [a, b, c, d]
sort(v') = [a, b, c, d]  ← 相同！

∴ L1(sort(v), sort(v')) = 0
```

排序是**置换不变的操作** (Permutation Invariant)

### 为什么需要全局搜索？

序列置换会改变 token 的位置，但不改变 token 本身：
```
原始: [Hello, World, !]
置换: [!, Hello, World]

如果只按顺序匹配：
位置0: "!" vs "Hello" ❌ 不匹配

全局搜索：
位置0: "!" vs 所有词汇 → 找到 "!"  ✅
位置1: "Hello" vs 所有词汇 → 找到 "Hello" ✅
```

## 📝 当前配置

```python
# vocab_matching_attack_scx.py 当前使用
dist_funct = "l1_sort"      # 排序L1匹配
perm_type = "D"             # 维度置换

# 适合攻击 SCX 的 MLP 层维度置换防御
```

## ✅ 总结

- **l1**: 基础版，最快，无防御
- **l1_permuted**: 处理序列打乱
- **l1_sort**: ⭐ 处理维度打乱（排序绕过）← 当前使用
- **l1_sort_permuted**: 全能版，处理所有情况

选择哪个取决于你要攻击什么样的防御！

