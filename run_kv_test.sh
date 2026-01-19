#!/bin/bash

# 测试KV cache攻击的matching_eps配置
# 会测试所有4个模型，生成两种配置（有置换和无置换）

echo "=================================="
echo "KV Cache 攻击 - 配置测试"
echo "=================================="
echo ""
echo "将测试以下模型:"
echo "  - llama3.2-3B"
echo "  - chatglm3-6B"
echo ""
echo "注意: llama-7B已移除（torch版本兼容性问题）"
echo ""
echo "每个模型测试:"
echo "  - 100个样本"
echo "  - 每个样本最多50个token"
echo "  - 3层（第一层、中间层、最后一层）"
echo "  - 2种置换类型（无置换、维度置换）"
echo ""
echo "结果将保存到:"
echo "  - kv_attack_config_no_perm.json (无置换)"
echo "  - kv_attack_config_with_perm.json (维度置换)"
echo ""
echo "=================================="
echo ""

# 测试无置换场景
echo "第一步：测试无置换场景 (perm_type=None)"
echo "=================================="
python test_matching_eps_kv.py \
    --models llama3.2-3B chatglm3-6B \
    --num_samples 100 \
    --max_tokens 50 \
    --perm_type None \
    --output kv_attack_config_no_perm.json

echo ""
echo "第一步完成！"
echo ""

# 测试维度置换场景
echo "第二步：测试维度置换场景 (perm_type=D)"
echo "=================================="
python test_matching_eps_kv.py \
    --models llama3.2-3B chatglm3-6B \
    --num_samples 100 \
    --max_tokens 50 \
    --perm_type D \
    --output kv_attack_config_with_perm.json

echo ""
echo "=================================="
echo "所有测试完成!"
echo "=================================="
echo ""
echo "请查看生成的配置文件:"
echo "  - kv_attack_config_no_perm.json (无置换)"
echo "  - kv_attack_config_with_perm.json (维度置换)"
echo ""

