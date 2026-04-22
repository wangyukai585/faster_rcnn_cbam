# Faster R-CNN + CBAM 目标检测

基于 MMDetection 框架实现 Faster R-CNN，并在 ResNet50 Backbone 中插入 CBAM 注意力模块，
在 PASCAL VOC 2007+2012 数据集上进行目标检测实验。

---

## 1. 环境安装

### 1.1 创建 Conda 环境

```bash
conda create -n faster_rcnn_cbam python=3.8 -y
conda activate faster_rcnn_cbam
```

### 1.2 安装 PyTorch（以 CUDA 11.6 为例，根据实际 GPU 选择版本）

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

> GPU 版本参考：https://pytorch.org/get-started/previous-versions/

### 1.3 安装 MMDetection 依赖链

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
```

### 1.4 安装其余依赖

```bash
pip install -r requirements.txt
```

### 1.5 安装 Jittor（加分项）

```bash
pip install jittor
# 验证安装
python -c "import jittor; jittor.test.test_example()"
```

---

## 2. 数据集下载与放置

### 2.1 下载链接

- VOC 2007 trainval：http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
- VOC 2007 test：http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
- VOC 2012 trainval：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

### 2.2 下载并解压

```bash
cd faster_rcnn_cbam/data

# 下载
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# 解压（三个包解压后都放在 VOCdevkit/ 下）
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
```

### 2.3 目录结构

解压后 `data/VOCdevkit/` 应如下所示：

```
data/VOCdevkit/
├── VOC2007/
│   ├── Annotations/       # XML 标注文件
│   ├── ImageSets/
│   │   └── Main/
│   │       ├── train.txt
│   │       ├── val.txt
│   │       └── test.txt
│   ├── JPEGImages/        # 原始图片
│   └── SegmentationClass/
└── VOC2012/
    ├── Annotations/
    ├── ImageSets/
    │   └── Main/
    │       ├── train.txt
    │       └── val.txt
    └── JPEGImages/
```

---

## 3. PyTorch 版训练

### 3.1 训练基线模型（Baseline Faster R-CNN）

```bash
cd faster_rcnn_cbam
python pytorch_mmdet/tools/train.py \
    pytorch_mmdet/configs/base_faster_rcnn.py \
    --work-dir experiments/results/baseline
```

### 3.2 训练加入 CBAM 的改进模型

```bash
python pytorch_mmdet/tools/train.py \
    pytorch_mmdet/configs/cbam_faster_rcnn.py \
    --work-dir experiments/results/cbam
```

---

## 4. Jittor 版训练

```bash
python jittor_impl/train_jittor.py \
    --data-root data/VOCdevkit \
    --epochs 12 \
    --lr 0.01 \
    --work-dir experiments/results/jittor_cbam
```

---

## 5. 模型评估

```bash
python pytorch_mmdet/tools/evaluate.py \
    pytorch_mmdet/configs/cbam_faster_rcnn.py \
    experiments/results/cbam/epoch_12.pth \
    --eval mAP
```

---

## 6. 消融实验

一键运行所有消融实验（基线 / 仅通道注意力 / 仅空间注意力 / 完整 CBAM）：

```bash
bash pytorch_mmdet/tools/run_ablation.sh
```

实验结果保存在 `experiments/results/` 下各子目录，运行完毕后汇总：

```bash
python analyze_results.py
```

---

## 7. 超参数实验

一键运行所有超参数实验（学习率、Batch Size）：

```bash
bash pytorch_mmdet/tools/run_hyper.sh
```

---

## 8. 实验结果汇总与可视化

```bash
python analyze_results.py
# 结果表格和图表输出到 report/figures/
```

---

## 项目结构

```
faster_rcnn_cbam/
├── pytorch_mmdet/
│   ├── configs/           # 各实验配置文件
│   ├── models/            # CBAM 模块 + ResNetCBAM
│   ├── tools/             # 训练 / 评估 / 实验脚本
│   └── utils/             # 可视化工具
├── jittor_impl/           # Jittor 版本实现
├── data/VOCdevkit/        # 数据集
├── experiments/results/   # 实验日志与权重
├── report/figures/        # 报告图表
├── analyze_results.py     # 结果汇总
└── requirements.txt
```
