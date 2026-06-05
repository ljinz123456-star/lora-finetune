#!/bin/bash
# ================================================================
# AutoDL 云端环境一键安装脚本
#
# 使用方法:
#   1. 在 AutoDL 创建实例后，打开 JupyterLab 终端
#   2. 将整个项目上传到 /root/autodl-tmp/
#   3. 运行: bash scripts/cloud_setup.sh
# ================================================================
set -e

echo "========================================"
echo "  LoRA 微调环境安装"
echo "========================================"

# 进入工作目录
WORKDIR="/root/autodl-tmp"
mkdir -p $WORKDIR
cd $WORKDIR

# ---- 1. 检查 GPU ----
echo ""
echo "[1/7] 检查 GPU 状态..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 2. 配置国内镜像加速 ----
echo ""
echo "[2/7] 配置镜像源..."
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# HuggingFace 镜像（下载模型和数据集用）
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# ---- 3. 安装 LLaMA-Factory ----
echo ""
echo "[3/7] 安装 LLaMA-Factory..."
if [ -d "LLaMA-Factory" ]; then
    echo "LLaMA-Factory 已存在，跳过克隆"
else
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --quiet

# ---- 4. 安装额外依赖 ----
echo ""
echo "[4/7] 安装额外依赖..."
pip install modelscope datasets sentence-transformers --quiet
pip install bitsandbytes --quiet
# Unsloth 加速（可选，节省显存）
pip install unsloth --quiet 2>/dev/null || echo "Unsloth 安装跳过（需要特定 CUDA 版本）"

# ---- 5. 创建目录结构 ----
echo ""
echo "[5/7] 创建目录结构..."
cd $WORKDIR
mkdir -p data output

# ---- 6. 准备数据集 ----
echo ""
echo "[6/7] 检查数据集..."
if [ -f "$WORKDIR/data/train.json" ] && [ -f "$WORKDIR/data/eval.json" ]; then
    echo "数据文件已存在，跳过下载"
else
    echo "未找到数据文件，从 HuggingFace 下载默认数据集..."
    python scripts/prepare_dataset.py --source huggingface
fi

# ---- 7. 预下载模型（可选，节省训练启动时间）----
echo ""
echo "[7/7] 预下载模型..."
# LLaMA-Factory 训练时会自动下载，此步骤可选但推荐
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/root/autodl-tmp/models')
print('模型下载完成')
" 2>/dev/null || echo "模型预下载跳过（训练时 LLaMA-Factory 会自动下载）"

echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "下一步 - 开始训练："
echo "  cd $WORKDIR/LLaMA-Factory"
echo "  cp $WORKDIR/scripts/train_config.yaml examples/train_lora/"
echo "  llamafactory-cli train examples/train_lora/train_config.yaml"
echo ""
