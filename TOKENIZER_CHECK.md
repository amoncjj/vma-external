# Tokenizer 特殊符号检查报告

## ✅ 检查结果：全部正确

### 📋 检查项目

#### 1. TOKENIZER.encode (编码)

**位置 1: gen_hidden_states() - 第 58 行**
```python
token_ids = TOKENIZER.encode(
    sentence, 
    return_tensors="pt", 
    add_special_tokens=False  # ✅ 正确
).to(device_map)
```
- ✅ 不添加 BOS/EOS token
- ✅ 用于生成攻击目标的隐藏状态

**位置 2: truncate_prompt() - 第 697 行**
```python
token_ids = TOKENIZER.encode(
    text, 
    add_special_tokens=False  # ✅ 正确
)[:num_tokens]
```
- ✅ 不添加特殊 token
- ✅ 用于截断文本到指定 token 数

**位置 3: 统计 token 数量 - 第 816 行**
```python
'num_tokens': len(TOKENIZER.encode(
    prompt, 
    add_special_tokens=False  # ✅ 正确
))
```
- ✅ 不添加特殊 token
- ✅ 用于统计实际 token 数量

#### 2. TOKENIZER.decode (解码)

**位置 1: truncate_prompt() - 第 698 行**
```python
return TOKENIZER.decode(
    token_ids, 
    skip_special_tokens=True  # ✅ 正确
)
```
- ✅ 跳过特殊 token
- ✅ 返回纯文本

**位置 2: 攻击结果解码 - 第 806 行**
```python
obtained_prediction = TOKENIZER.decode(
    decoded_tokens, 
    skip_special_tokens=True  # ✅ 正确
)
```
- ✅ 跳过特殊 token
- ✅ 解码攻击恢复的 token

## 🎯 总结

### ✅ 全部通过
- **3个 encode 位置**: 全部使用 `add_special_tokens=False`
- **2个 decode 位置**: 全部使用 `skip_special_tokens=True`

### 🔒 保证
1. ❌ **不会添加** BOS (Beginning of Sentence) token
2. ❌ **不会添加** EOS (End of Sentence) token  
3. ❌ **不会添加** PAD (Padding) token
4. ✅ **只处理** 实际文本内容的 token

### 📊 示例

#### Llama-3 的特殊 token
```python
BOS token: <|begin_of_text|>  (ID = 128000)
EOS token: <|end_of_text|>    (ID = 128001)
```

#### 我们的编码方式
```python
# ❌ 错误（会添加特殊 token）
tokens = tokenizer.encode("Hello world")
# 结果: [128000, 9906, 1917]
#       ↑ BOS token

# ✅ 正确（不添加特殊 token）
tokens = tokenizer.encode("Hello world", add_special_tokens=False)
# 结果: [9906, 1917]
#       ↑ 直接是 "Hello"
```

### 🔍 验证方法

如果想手动验证：
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/junjie_chen/models/llama3-8B")

text = "The history of Nero"

# 带特殊 token
with_special = tokenizer.encode(text)
print("带特殊token:", with_special)
print("解码:", tokenizer.decode(with_special))

# 不带特殊 token
without_special = tokenizer.encode(text, add_special_tokens=False)
print("不带特殊token:", without_special)
print("解码:", tokenizer.decode(without_special))
```

## ✅ 结论

**所有 tokenizer 调用都已正确配置，不会引入 BOS、EOS 等特殊符号！**

攻击过程中只处理实际的文本 token，确保了：
1. 目标隐藏状态 = 纯文本的隐藏状态
2. 词汇表隐藏状态 = 纯文本的隐藏状态
3. 匹配过程 = 在相同空间中进行
4. 结果解码 = 纯净的文本内容

---

**检查时间**: 2025-12-22  
**检查状态**: ✅ 通过  
**建议**: 保持当前配置

