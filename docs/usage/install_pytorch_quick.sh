#!/bin/bash
# PyTorch CUDA 快速安装脚本（Linux/Mac）
# 适用于 CUDA 12.9 / RTX 4070

echo "========================================"
echo "PyTorch CUDA 版本快速安装"
echo "========================================"
echo ""
echo "检测到系统信息:"
echo "- CUDA Version: 12.9"
echo "- GPU: NVIDIA GeForce RTX 4070"
echo ""

echo "正在安装 PyTorch (CUDA 12.4, 兼容 12.9)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "========================================"
echo "验证安装..."
echo "========================================"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "========================================"
if [ $? -eq 0 ]; then
    echo "安装完成！"
else
    echo "安装可能有问题，请检查错误信息"
fi
echo "========================================"

