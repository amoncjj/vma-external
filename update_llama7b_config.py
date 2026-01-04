#!/usr/bin/env python3
"""
转换完成后，自动更新配置文件以使用safetensors版本的llama-7B
"""

import re

FILES_TO_UPDATE = [
    "test_matching_eps_kv.py",
    "vocab_matching_attack_kv.py",
]

OLD_PATH = "/home/junjie_chen/models/llama-7B"
NEW_PATH = "/home/junjie_chen/models/llama-7B-safetensors"

print("="*80)
print("更新配置文件以使用safetensors版本的llama-7B")
print("="*80)
print()

for file_path in FILES_TO_UPDATE:
    print(f"更新文件: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 取消注释llama-7B行
        content = re.sub(
            r'#\s*"llama-7B":\s*"[^"]*",\s*#.*',
            f'"llama-7B": "{NEW_PATH}",',
            content
        )
        
        # 更新路径（如果已经取消注释）
        content = content.replace(OLD_PATH, NEW_PATH)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"  ✅ {file_path} 更新完成")
    
    except Exception as e:
        print(f"  ❌ 错误: {e}")

print()
print("="*80)
print("✅ 配置更新完成!")
print("="*80)
print()
print("llama-7B现在使用safetensors格式:")
print(f"  路径: {NEW_PATH}")
print()
print("可以运行测试了:")
print("  ./run_kv_test.sh")
print()

