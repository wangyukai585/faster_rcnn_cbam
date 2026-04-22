#!/bin/bash
# ============================================================
# AutoDL 一键安装脚本
# 适用镜像：PyTorch 2.1.0 + CUDA 11.8 + Ubuntu 20.04
# 执行方式：bash install.sh
# ============================================================

set -e   # 任何命令出错立即退出

echo "====== [1/5] 配置 pip 国内镜像 ======"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo "====== [2/5] 安装 mmengine ======"
pip install mmengine>=0.8.0

echo "====== [3/5] 安装 mmcv（预编译 wheel，对应 PyTorch 2.1.0 + CUDA 11.8）======"
# 官方预编译 wheel 地址（国内可访问）
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

echo "====== [4/5] 安装 mmdet 及其他依赖 ======"
pip install mmdet>=3.2.0
pip install numpy pillow scipy matplotlib seaborn pandas pycocotools tqdm lxml

echo "====== [5/5] 下载 ResNet50 预训练权重 ======"
CKPT_DIR=~/.cache/torch/hub/checkpoints
mkdir -p "$CKPT_DIR"
CKPT_FILE="$CKPT_DIR/resnet50-0676ba61.pth"
if [ ! -f "$CKPT_FILE" ]; then
    echo "  正在下载 resnet50-0676ba61.pth ..."
    # 尝试官方地址
    wget -q --show-progress \
        https://download.pytorch.org/models/resnet50-0676ba61.pth \
        -O "$CKPT_FILE" || \
    # 备用：OpenMMLab 镜像
    wget -q --show-progress \
        https://download.openmmlab.com/pretrain/third_party/resnet50-0676ba61.pth \
        -O "$CKPT_FILE"
    echo "  已保存到 $CKPT_FILE"
else
    echo "  权重文件已存在，跳过下载"
fi

echo ""
echo "====== 安装完成！验证环境 ======"
python -c "
import torch, mmdet, mmcv, mmengine
print(f'PyTorch:   {torch.__version__}')
print(f'CUDA:      {torch.version.cuda}')
print(f'GPU可用:   {torch.cuda.is_available()}')
print(f'mmengine:  {mmengine.__version__}')
print(f'mmcv:      {mmcv.__version__}')
print(f'mmdet:     {mmdet.__version__}')
"
echo ""
echo "====== 下一步：准备 VOC 数据集 ======"
echo "  将 VOCdevkit/ 放到 faster_rcnn_cbam/data/VOCdevkit/"
echo "  目录结构：data/VOCdevkit/VOC2007/ 和 data/VOCdevkit/VOC2012/"
