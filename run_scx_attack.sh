#!/bin/bash

# SCX Attack Script
# 词汇匹配攻击 - 维度置换测试

echo "=================================================="
echo "SCX 词汇匹配攻击"
echo "=================================================="
echo ""
echo "配置信息:"
echo "  模型: Llama-3-8B"
echo "  攻击层: 5层 [0, 7, 15, 23, 31]"
echo "  置换类型: D (维度置换)"
echo "  匹配函数: l1_sort (排序L1)"
echo "  数据集: WikiText-2 (100样本)"
echo "  随机种子: 42 (可重复)"
echo ""
echo "开始攻击..."
echo ""
nohup python vocab_matching_attack_scx.py > my_log.txt 2>&1 &

echo ""
echo "=================================================="
echo "攻击完成!"
echo "结果保存在: scx_attack_results.json"
echo "=================================================="
echo ""
echo "查看结果:"
echo "  python analyze_results.py"
echo ""

