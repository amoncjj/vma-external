# Error计算方法对比

## test_matching_eps_kv.py (compute_kv_errors)

```python
# 提取最后一个token的K和V
k_cache = outputs.past_key_values[layer][0]  # shape: (1, num_heads, seq_len, head_dim)
v_cache = outputs.past_key_values[layer][1]

batch_size, num_heads, seq_len, head_dim = k_cache.shape  # batch_size=1
k_last = k_cache[:, :, -1, :].reshape(num_heads * head_dim)  # shape: (num_heads * head_dim,)
v_last = v_cache[:, :, -1, :].reshape(num_heads * head_dim)

# 获取permuted的K和V
perm_k_row = perm_k_states[i, :]  # shape: (num_heads * head_dim,)
perm_v_row = perm_v_states[i, :]

# 排序（如果perm_type == "D"）
if use_sort:
    sorted_perm_k, _ = torch.sort(perm_k_row)
    sorted_k, _ = torch.sort(k_last)
    sorted_perm_v, _ = torch.sort(perm_v_row)
    sorted_v, _ = torch.sort(v_last)
else:
    sorted_perm_k = perm_k_row
    sorted_k = k_last
    sorted_perm_v = perm_v_row
    sorted_v = v_last

# 计算error
k_error = torch.sum(torch.abs(sorted_perm_k - sorted_k)).item()
v_error = torch.sum(torch.abs(sorted_perm_v - sorted_v)).item()
total_error = k_error + v_error
```

## vocab_matching_attack_kv.py (kv_matching_attack)

```python
# 提取最后一个token的K和V
k_cache = outputs.past_key_values[layer][0]  # shape: (batch_size, num_heads, seq_len, head_dim)
v_cache = outputs.past_key_values[layer][1]

batch_size, num_heads, seq_len, head_dim = k_cache.shape
batch_k = k_cache[:, :, -1, :].reshape(batch_size, num_heads * head_dim)  # shape: (batch_size, num_heads * head_dim)
batch_v = v_cache[:, :, -1, :].reshape(batch_size, num_heads * head_dim)

# 获取permuted的K和V
perm_k_row = perm_k_states[i, :]  # shape: (num_heads * head_dim,)
perm_v_row = perm_v_states[i, :]

# 排序（如果perm_type == "D"）
if use_sort:
    sorted_perm_k, _ = torch.sort(perm_k_row)
    sorted_perm_v, _ = torch.sort(perm_v_row)
else:
    sorted_perm_k = perm_k_row
    sorted_perm_v = perm_v_row

# 对每个候选token计算error
for j in range(actual_batch_sz):
    if use_sort:
        sorted_k, _ = torch.sort(batch_k[j])  # batch_k[j] shape: (num_heads * head_dim,)
        sorted_v, _ = torch.sort(batch_v[j])
    else:
        sorted_k = batch_k[j]
        sorted_v = batch_v[j]
    
    k_error = torch.sum(torch.abs(sorted_perm_k - sorted_k)).item()
    v_error = torch.sum(torch.abs(sorted_perm_v - sorted_v)).item()
    total_error = k_error + v_error
```

## 结论

**计算方法是一致的**：
1. 都使用相同的reshape方式：`num_heads * head_dim`
2. 都使用相同的排序逻辑（当perm_type == "D"时）
3. 都使用相同的error计算公式：
   - `k_error = sum(|sorted_perm_k - sorted_k|)`
   - `v_error = sum(|sorted_perm_v - sorted_v|)`
   - `total_error = k_error + v_error`

**唯一区别**：
- test中：batch_size=1，所以可以省略batch维度
- attack中：batch_size>1，需要保留batch维度，然后对每个候选token分别计算

**潜在问题**：
- 在test中，如果batch_size不是1，`reshape(num_heads * head_dim)`可能会出错
- 但实际使用中，test的batch_size确实是1，所以没有问题

