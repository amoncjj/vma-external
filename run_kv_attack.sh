#!/bin/bash

# KV Cache 攻击脚本 - 批量测试所有组合
# 测试所有模型、所有置换类型、所有层

# 模型列表
MODELS=("llama3.2-1B" "llama3-8B" "qwen3-8B")

# 置换类型列表
PERM_TYPES=("None" "D")

# 从配置文件提取层信息的Python脚本
GET_LAYERS_PY=$(cat <<'PYTHON_EOF'
import json
import sys

config_file = sys.argv[1]
model_name = sys.argv[2]

with open(config_file, 'r') as f:
    config = json.load(f)

if model_name in config:
    layers = list(config[model_name]['layers'].keys())
    # 转换为整数并排序
    layers = sorted([int(l) for l in layers])
    # 输出为空格分隔的字符串
    print(' '.join(map(str, layers)))
else:
    print("")
PYTHON_EOF
)

# 检查是否有参数（如果提供了参数，则使用旧的行为）
if [ $# -ge 2 ]; then
    # 旧的行为：单个模型和置换类型
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
    exit 0
fi

# 新的行为：批量测试所有组合
echo "=================================="
echo "KV Cache 攻击 - 批量测试所有组合"
echo "=================================="
echo ""
echo "将测试以下组合:"
echo "  - 模型: ${MODELS[@]}"
echo "  - 置换类型: ${PERM_TYPES[@]}"
echo "  - 层: 从配置文件读取（第一层、中间层、最后一层）"
echo ""
echo "=================================="
echo ""

# 计数器
CURRENT_COMBINATION=0
TOTAL_COMBINATIONS=0

# 先计算总组合数（跳过已运行的组合）
for PERM_TYPE in "${PERM_TYPES[@]}"; do
    if [ "$PERM_TYPE" == "None" ]; then
        CONFIG_FILE="kv_attack_config_no_perm.json"
    else
        CONFIG_FILE="kv_attack_config_with_perm.json"
    fi
    
    for MODEL in "${MODELS[@]}"; do
        LAYERS=$(python3 -c "$GET_LAYERS_PY" "$CONFIG_FILE" "$MODEL")
        if [ -n "$LAYERS" ]; then
            LAYER_ARRAY=($LAYERS)
            # 计算该模型的层数（跳过已运行的层）
            for LAYER in "${LAYER_ARRAY[@]}"; do
                # 跳过 llama3.2-1B + None 的第一层（层0）
                if [ "$MODEL" == "llama3.2-1B" ] && [ "$PERM_TYPE" == "None" ] && [ "$LAYER" == "0" ]; then
                    continue
                fi
                TOTAL_COMBINATIONS=$((TOTAL_COMBINATIONS + 1))
            done
        fi
    done
done

echo "总共: $TOTAL_COMBINATIONS 种组合"
echo ""
echo "=================================="
echo ""

# 遍历所有组合
for PERM_TYPE in "${PERM_TYPES[@]}"; do
    # 根据perm_type选择配置文件
    if [ "$PERM_TYPE" == "None" ]; then
        CONFIG_FILE="kv_attack_config_no_perm.json"
        PERM_DESC="无置换"
    else
        CONFIG_FILE="kv_attack_config_with_perm.json"
        PERM_DESC="维度置换"
    fi
    
    for MODEL in "${MODELS[@]}"; do
        # 从配置文件提取层信息
        LAYERS=$(python3 -c "$GET_LAYERS_PY" "$CONFIG_FILE" "$MODEL")
        
        if [ -z "$LAYERS" ]; then
            echo "⚠️  警告: 无法从配置文件 $CONFIG_FILE 中找到模型 $MODEL 的层信息，跳过"
            continue
        fi
        
        # 将层信息转换为数组
        LAYER_ARRAY=($LAYERS)
        
        # 遍历每一层
        for LAYER in "${LAYER_ARRAY[@]}"; do
            # 跳过 llama3.2-1B + None 的第一层（层0，已运行过）
            if [ "$MODEL" == "llama3.2-1B" ] && [ "$PERM_TYPE" == "None" ] && [ "$LAYER" == "0" ]; then
                echo "⏭️  跳过: $MODEL + $PERM_TYPE + Layer $LAYER (已运行过)"
                continue
            fi
            CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))
            
            echo ""
            echo "============================================================"
            echo "组合 $CURRENT_COMBINATION/$TOTAL_COMBINATIONS"
            echo "============================================================"
            echo "模型: $MODEL"
            echo "置换类型: $PERM_TYPE ($PERM_DESC)"
            echo "层: $LAYER"
            echo "配置文件: $CONFIG_FILE"
            echo "============================================================"
            echo ""
            
            # 运行攻击（只测试这一层）
            python vocab_matching_attack_kv.py \
                --model "$MODEL" \
                --perm_type "$PERM_TYPE" \
                --config "$CONFIG_FILE" \
                --layers "$LAYER"
            
            EXIT_CODE=$?
            
            if [ $EXIT_CODE -eq 0 ]; then
                echo ""
                echo "✅ 组合 $CURRENT_COMBINATION/$TOTAL_COMBINATIONS 完成"
            else
                echo ""
                echo "❌ 组合 $CURRENT_COMBINATION/$TOTAL_COMBINATIONS 失败 (退出码: $EXIT_CODE)"
            fi
            
            echo ""
            echo "============================================================"
            echo ""
        done
    done
done

echo ""
echo "=================================="
echo "🎉 所有测试完成!"
echo "=================================="
echo ""
echo "总共测试了 $CURRENT_COMBINATION 种组合"
echo ""
