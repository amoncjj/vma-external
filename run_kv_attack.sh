#!/bin/bash

# KV Cache 攻击脚本
# 使用推荐的配置对指定模型进行攻击

if [ $# -lt 2 ]; then
    echo "用法: $0 <model_name> <perm_type> [options]"
    echo ""
    echo "可用的模型:"
    echo "  - llama3.2-1B"
    echo "  - llama3-8B"
    echo "  - llama-7B"
    echo "  - qwen3-8B"
    echo ""
    echo "置换类型:"
    echo "  - None: 无置换"
    echo "  - D: 维度置换"
    echo ""
    echo "示例:"
    echo "  $0 llama3-8B None           # 无置换攻击"
    echo "  $0 llama3-8B D              # 维度置换攻击"
    echo "  $0 llama3-8B None --num_samples 50 --layers 1 16 32"
    echo ""
    exit 1
fi

MODEL=$1
PERM_TYPE=$2
shift 2  # 移除前两个参数，剩下的传给python脚本

# 验证perm_type
if [ "$PERM_TYPE" != "None" ] && [ "$PERM_TYPE" != "D" ]; then
    echo "错误: perm_type 必须是 'None' 或 'D'"
    exit 1
fi

# 根据perm_type选择配置文件
if [ "$PERM_TYPE" == "None" ]; then
    CONFIG_FILE="kv_attack_config_no_perm.json"
    PERM_DESC="无置换"
else
    CONFIG_FILE="kv_attack_config_with_perm.json"
    PERM_DESC="维度置换"
fi

echo "=================================="
echo "KV Cache 攻击 - 开始攻击"
echo "=================================="
echo ""
echo "模型: $MODEL"
echo "置换类型: $PERM_TYPE ($PERM_DESC)"
echo "配置文件: $CONFIG_FILE"
echo "数据集: lmsys-chat-1m (默认100句)"
echo ""
echo "=================================="
echo ""

# 运行攻击
python vocab_matching_attack_kv.py \
    --model $MODEL \
    --perm_type $PERM_TYPE \
    --config $CONFIG_FILE \
    "$@"

echo ""
echo "=================================="
echo "攻击完成!"
echo "=================================="
echo ""

