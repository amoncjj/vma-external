#!/bin/bash

# 将llama-7B转换为safetensors格式

echo "=================================="
echo "llama-7B 格式转换工具"
echo "=================================="
echo ""
echo "选择转换设备:"
echo "  1) 自动选择 (推荐 - 优先使用GPU)"
echo "  2) GPU (需要约14GB+ GPU内存，速度快)"
echo "  3) CPU (需要约30GB+ RAM，速度慢)"
echo ""
read -p "请选择 (1/2/3) [默认:1]: " device_choice

case "$device_choice" in
    2)
        DEVICE="cuda"
        DEVICE_NAME="GPU"
        MEMORY_REQ="14GB+ GPU内存"
        ;;
    3)
        DEVICE="cpu"
        DEVICE_NAME="CPU"
        MEMORY_REQ="30GB+ RAM"
        ;;
    *)
        DEVICE="auto"
        DEVICE_NAME="自动选择"
        MEMORY_REQ="14GB+ GPU内存 或 30GB+ RAM"
        ;;
esac

echo ""
echo "转换配置:"
echo "  源模型: /home/junjie_chen/models/llama-7B"
echo "  目标路径: /home/junjie_chen/models/llama-7B-safetensors"
echo "  使用设备: $DEVICE_NAME"
echo "  内存需求: $MEMORY_REQ"
echo "  预计时间: 5-20 分钟 (取决于设备)"
echo ""
read -p "确认继续? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "开始转换..."
echo ""

# 运行转换脚本
python convert_llama7b_to_safetensors.py --device $DEVICE --dtype float16

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ 转换成功!"
    echo "=================================="
    echo ""
    echo "现在可以使用safetensors版本的llama-7B了"
    echo ""
else
    echo ""
    echo "=================================="
    echo "❌ 转换失败"
    echo "=================================="
    echo ""
    echo "请查看上面的错误信息"
    echo ""
    exit 1
fi

