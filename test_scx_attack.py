"""
快速测试脚本 - 验证 SCX 攻击设置是否正确
仅在一层上测试少量样本
"""

import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置随机种子以确保可重复性
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

print("="*80)
print("SCX 攻击快速测试")
print("="*80)
print(f"\n✓ 随机种子已设置: {RANDOM_SEED} (确保结果可重复)")

# 配置
model_path = "/home/junjie_chen/models/llama3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n1. 检查设备: {device}")
if device == "cpu":
    print("   警告: 未检测到 GPU，攻击将会非常慢")

print(f"\n2. 加载模型: {model_path}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(device)
    model.eval()
    print(f"   ✓ 模型加载成功")
    print(f"   - 层数: {model.config.num_hidden_layers}")
    print(f"   - 隐藏维度: {model.config.hidden_size}")
    print(f"   - 词汇表大小: {model.config.vocab_size}")
except Exception as e:
    print(f"   ✗ 模型加载失败: {e}")
    exit(1)

print(f"\n3. 测试前向传播")
try:
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    print(f"   ✓ 前向传播成功")
    print(f"   - 输入 token 数: {inputs['input_ids'].shape[1]}")
    print(f"   - 隐藏状态层数: {len(outputs.hidden_states)}")
except Exception as e:
    print(f"   ✗ 前向传播失败: {e}")
    exit(1)

print(f"\n4. 计算攻击层")
num_layers = model.config.num_hidden_layers
attack_layers = [int(i * (num_layers - 1) / 4) for i in range(5)]
print(f"   ✓ 攻击层 (共5层): {attack_layers}")

print(f"\n5. 测试 wikitext-2 数据集加载")
try:
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    print(f"   ✓ 数据集加载成功")
    print(f"   - 总样本数: {len(dataset)}")
    
    # 找到第一个有效样本
    valid_samples = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 50 and not text.startswith('='):
            valid_samples.append(text[:100])
            if len(valid_samples) >= 3:
                break
    
    print(f"   - 找到 {len(valid_samples)} 个有效样本")
    print(f"\n   示例样本:")
    for i, sample in enumerate(valid_samples):
        print(f"     [{i+1}] {sample}...")
        
except Exception as e:
    print(f"   ✗ 数据集加载失败: {e}")
    print(f"   提示: 运行 'pip install datasets' 安装依赖")
    exit(1)

print(f"\n6. 运行小规模攻击测试 (1个样本, 1层)")
try:
    test_sample = valid_samples[0][:50]  # 截断到50个字符
    test_layer = attack_layers[0]  # 只测试第一层
    
    print(f"   测试样本: {test_sample}")
    print(f"   测试层: {test_layer}")
    
    # 生成隐藏状态
    token_ids = tokenizer.encode(test_sample, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.forward(token_ids, output_hidden_states=True)
        hidden_states = output.hidden_states[test_layer]
    
    print(f"   ✓ 隐藏状态生成成功")
    print(f"   - Token 数: {token_ids.shape[1]}")
    print(f"   - 隐藏状态形状: {hidden_states.shape}")
    
    # 测试简单的词汇匹配（只测试前10个token）
    print(f"\n   测试简单词汇匹配 (前10个候选token)...")
    test_vocab_size = 10
    test_token_ids = torch.arange(0, test_vocab_size).to(device).reshape(-1, 1)
    
    with torch.no_grad():
        test_outputs = model.forward(test_token_ids, output_hidden_states=True)
        test_hidden_states = test_outputs.hidden_states[test_layer]
    
    print(f"   ✓ 词汇匹配测试成功")
    print(f"   - 测试候选 token 形状: {test_hidden_states.shape}")
    
except Exception as e:
    print(f"   ✗ 攻击测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n{'='*80}")
print("✓ 所有测试通过！")
print("='*80}")
print("\n现在可以运行完整攻击:")
print("  ./run_scx_attack.sh")
print("  或")
print("  python vocab_matching_attack_scx.py")
print()

