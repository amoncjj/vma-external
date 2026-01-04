#!/usr/bin/env python3
"""
将llama-7B模型从.bin格式转换为safetensors格式
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil
from pathlib import Path
import argparse

# 配置
SOURCE_MODEL = "/home/junjie_chen/models/llama-7B"
TARGET_MODEL = "/home/junjie_chen/models/llama-7B-safetensors"

# 命令行参数
parser = argparse.ArgumentParser(description="将llama-7B转换为safetensors格式")
parser.add_argument("--device", type=str, default="auto", 
                    choices=["auto", "cuda", "cpu"],
                    help="使用的设备: auto(自动选择), cuda(GPU), cpu")
parser.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"],
                    help="模型精度: float16, bfloat16, float32")
args = parser.parse_args()

# 确定使用的设备
if args.device == "auto":
    if torch.cuda.is_available():
        device_map = "auto"  # 自动分配到GPU
        device_name = "GPU (自动分配)"
    else:
        device_map = "cpu"
        device_name = "CPU"
elif args.device == "cuda":
    if torch.cuda.is_available():
        device_map = "auto"
        device_name = "GPU"
    else:
        print("❌ 错误: CUDA不可用，请使用 --device cpu")
        exit(1)
else:
    device_map = "cpu"
    device_name = "CPU"

# 确定数据类型
dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
model_dtype = dtype_map[args.dtype]

print("="*80)
print("llama-7B 模型格式转换工具")
print("="*80)
print(f"源模型路径: {SOURCE_MODEL}")
print(f"目标路径: {TARGET_MODEL}")
print(f"使用设备: {device_name}")
print(f"模型精度: {args.dtype}")
if torch.cuda.is_available():
    print(f"GPU信息: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("="*80)
print()

# 检查源模型是否存在
if not os.path.exists(SOURCE_MODEL):
    print(f"❌ 错误: 源模型路径不存在: {SOURCE_MODEL}")
    exit(1)

# 创建目标目录
os.makedirs(TARGET_MODEL, exist_ok=True)
print(f"✅ 创建目标目录: {TARGET_MODEL}")

# 步骤1 & 2: 复制tokenizer和config文件
print("\n步骤1-2: 复制tokenizer和配置文件...")
try:
    files_to_copy = [
        "config.json",
        "generation_config.json", 
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "tokenizer.json",
    ]
    
    for file_name in files_to_copy:
        src_file = os.path.join(SOURCE_MODEL, file_name)
        dst_file = os.path.join(TARGET_MODEL, file_name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"  ✅ {file_name}")
        else:
            print(f"  ⚠️  {file_name} 不存在，跳过")
    
    print("✅ 文件复制完成")
except Exception as e:
    print(f"❌ 文件复制失败: {e}")
    exit(1)

# 步骤3: 转换模型权重
print("\n步骤3: 转换模型权重 (.bin -> safetensors)...")
if device_map == "auto":
    print("⚠️  注意: 使用GPU加载，需要足够的GPU内存（约14GB+）")
else:
    print("⚠️  注意: 使用CPU加载，需要足够的RAM（约30GB+）")
print("正在加载模型...")

try:
    # 设置环境变量跳过torch版本检查（仅用于转换）
    os.environ["TRANSFORMERS_NO_TORCH_LOAD_SAFEGUARDS"] = "1"
    
    print(f"使用{device_name}加载模型（精度: {args.dtype}）...")
    
    # 检查是否有accelerate
    try:
        import accelerate
        has_accelerate = True
    except ImportError:
        has_accelerate = False
        if device_map == "auto" or device_map.startswith("cuda"):
            print("⚠️  accelerate未安装，使用简单的GPU加载")
    
    if has_accelerate:
        model = AutoModelForCausalLM.from_pretrained(
            SOURCE_MODEL,
            torch_dtype=model_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        # 不使用device_map，直接指定device
        print("加载到CPU，然后移动到GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            SOURCE_MODEL,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if device_map != "cpu":
            print("移动模型到GPU...")
            model = model.to("cuda")
    
    print("✅ 模型加载完成")
    
    # 如果使用GPU，打印显存使用情况
    if torch.cuda.is_available() and device_map != "cpu":
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
    
    # 保存为safetensors格式
    print("正在保存为safetensors格式...")
    model.save_pretrained(
        TARGET_MODEL,
        safe_serialization=True,  # 使用safetensors格式
        max_shard_size="5GB",  # 每个分片最大5GB
    )
    print("✅ 模型保存完成")
    
    # 清理环境变量
    if "TRANSFORMERS_NO_TORCH_LOAD_SAFEGUARDS" in os.environ:
        del os.environ["TRANSFORMERS_NO_TORCH_LOAD_SAFEGUARDS"]
    
except Exception as e:
    print(f"❌ 模型转换失败: {e}")
    print("\n可能的原因:")
    if device_map == "auto":
        print("1. GPU内存不足 - llama-7B需要约14GB+ GPU内存")
        print("2. 尝试使用CPU: python convert_llama7b_to_safetensors.py --device cpu")
    else:
        print("1. CPU内存不足 - 需要至少30GB+ RAM")
        print("2. 如果有GPU，尝试使用GPU会更快: python convert_llama7b_to_safetensors.py --device cuda")
    print("3. torch版本问题 - 当前版本无法加载.bin文件")
    print("\n建议:")
    print("1. 使用float16或bfloat16减少内存使用")
    print("2. 关闭其他占用内存的程序")
    print("3. 或者直接从HuggingFace下载safetensors版本的llama-7B")
    import traceback
    traceback.print_exc()
    exit(1)

# 步骤4: 验证转换结果
print("\n步骤4: 验证转换结果...")
safetensors_files = list(Path(TARGET_MODEL).glob("*.safetensors"))
if safetensors_files:
    print(f"✅ 找到 {len(safetensors_files)} 个safetensors文件:")
    for f in safetensors_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
else:
    print("❌ 错误: 未找到safetensors文件")
    exit(1)

# 步骤5: 测试加载
print("\n步骤5: 测试加载转换后的模型...")
try:
    # 使用相同的设备进行测试
    if has_accelerate:
        test_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL,
            torch_dtype=model_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    else:
        test_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        if device_map != "cpu":
            test_model = test_model.to("cuda")
    
    print("✅ 模型加载测试成功")
    
    # 清理测试模型
    del test_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as e:
    print(f"❌ 模型加载测试失败: {e}")
    exit(1)

print("\n" + "="*80)
print("🎉 转换完成!")
print("="*80)
print(f"原始模型: {SOURCE_MODEL}")
print(f"转换后模型: {TARGET_MODEL}")
print("\n下一步:")
print("1. 更新MODEL_CONFIGS使用新路径:")
print(f'   "llama-7B": "{TARGET_MODEL}"')
print("\n2. 取消注释llama-7B配置:")
print("   在test_matching_eps_kv.py和vocab_matching_attack_kv.py中")
print("\n3. 运行测试:")
print("   ./run_kv_test.sh")
print("="*80)

