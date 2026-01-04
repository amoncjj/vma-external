#!/bin/bash

# 单独测试llama-7B模型

echo "=================================="
echo "测试 llama-7B 模型"
echo "=================================="
echo ""
echo "配置:"
echo "  模型: llama-7B"
echo "  样本数: 10 (快速测试)"
echo "  置换类型: None"
echo ""
echo "=================================="
echo ""

# 运行测试
python test_matching_eps_kv.py \
    --models llama-7B \
    --num_samples 10 \
    --max_tokens 50 \
    --perm_type None \
    --output test_llama7b_config.json

echo ""
echo "=================================="
echo "测试完成!"
echo "=================================="
echo ""

