# PyTorch CUDA 版本安装指南

## 快速安装

### 1. 检查 CUDA 版本

首先检查您的系统 CUDA 版本：

```bash
nvidia-smi
```

查看右上角显示的 CUDA Version（例如：12.9, 12.4, 12.1, 11.8 等）

**注意**: 如果您的 CUDA 版本是 12.9，PyTorch 可能没有专门的 cu129 wheel，但可以使用 cu124 或 cu121（向后兼容）

### 2. 根据 CUDA 版本安装 PyTorch

访问 PyTorch 官网获取最新的安装命令：
**https://pytorch.org/get-started/locally/**

#### 常见 CUDA 版本安装命令：

**CUDA 12.9 / 12.4（推荐，兼容 CUDA 12.9）:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**CUDA 12.1（向后兼容 CUDA 12.9）:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 11.7:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

**CPU 版本（如果不需要 GPU）:**
```bash
pip install torch torchvision torchaudio
```

### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

如果输出显示 `CUDA available: True` 和您的 GPU 名称（如 "NVIDIA GeForce RTX 4070"），说明安装成功！

## 完整安装流程

### 步骤 1: 安装其他依赖

```bash
pip install -r requirements.txt
```

注意：这会跳过 PyTorch（因为已在 requirements.txt 中注释）

### 步骤 2: 安装 PyTorch CUDA 版本

根据您的 CUDA 版本，从上面的命令中选择一个执行。

### 步骤 3: 验证

运行验证命令确认 PyTorch 和 CUDA 正常工作。

## 常见问题

### Q: 如何知道我的 CUDA 版本？

A: 运行 `nvidia-smi` 查看，或运行：
```bash
nvcc --version
```

### Q: PyTorch 版本和 CUDA 版本不匹配怎么办？

A: 卸载旧版本后重新安装：
```bash
pip uninstall torch torchvision torchaudio
# 然后根据您的 CUDA 版本重新安装
```

### Q: 安装后 `torch.cuda.is_available()` 返回 False？

A: 可能的原因：
1. CUDA 版本不匹配
2. 显卡驱动版本过旧
3. 安装了 CPU 版本的 PyTorch

解决方法：
- 检查 CUDA 版本：`nvidia-smi`
- 检查 PyTorch 版本：`python -c "import torch; print(torch.__version__)"`
- 重新安装匹配的 CUDA 版本

## 参考链接

- PyTorch 官方安装页面：https://pytorch.org/get-started/locally/
- PyTorch 版本兼容性：https://pytorch.org/get-started/previous-versions/

