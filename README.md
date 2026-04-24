# Faster R-CNN + CBAM 目标检测

基于 MMDetection 框架实现 Faster R-CNN，在 ResNet50 Backbone 中插入 CBAM 注意力模块，
在 PASCAL VOC 2007+2012 数据集上进行目标检测实验。

---

## 1. 环境配置

### 1.1 推荐环境（AutoDL）

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.0.0 |
| CUDA | 11.8 |
| OS | Ubuntu 20.04 |
| mmengine | ≥ 0.8.0 |
| mmcv | 2.0.1 |
| mmdet | ≥ 3.2.0 |

### 1.2 一键安装

```bash
bash install.sh
```

脚本依次完成：pip 镜像配置 → mmengine → mmcv 预编译 wheel → mmdet → 其他依赖 → 下载 ResNet50 预训练权重。

> **注意**：mmcv 必须使用 OpenMMLab 预编译 wheel 安装（与 PyTorch/CUDA 版本严格对应），
> 不能直接 `pip install mmcv`。

### 1.3 验证安装

```bash
python -c "import torch, mmdet, mmcv, mmengine; print(torch.__version__, mmdet.__version__)"
```

---

## 2. 数据集准备

### 2.1 AutoDL 公开数据集（推荐，秒级解压）

AutoDL 提供 VOC 数据集的公开镜像，无需从外网下载：

```bash
cd /root/autodl-tmp/faster_rcnn_cbam/data
tar xzf /root/autodl-pub/VOCdevkit/VOC2007.tar.gz
tar xzf /root/autodl-pub/VOCdevkit/VOC2012.tar.gz
```

### 2.2 从官方地址下载（备用，国内较慢）

```bash
cd faster_rcnn_cbam/data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
```

### 2.3 目录结构

解压后 `data/VOCdevkit/` 应如下所示：

```
data/VOCdevkit/
├── VOC2007/
│   ├── Annotations/
│   ├── ImageSets/Main/   (train.txt / val.txt / test.txt / trainval.txt)
│   └── JPEGImages/
└── VOC2012/
    ├── Annotations/
    ├── ImageSets/Main/   (train.txt / val.txt / trainval.txt)
    └── JPEGImages/
```

---

## 3. 训练

所有命令在 `faster_rcnn_cbam/` 目录下执行。

### 3.1 消融实验（4 组，一键运行）

```bash
bash pytorch_mmdet/tools/run_ablation.sh
```

| # | 配置文件 | 说明 |
|---|----------|------|
| 1 | base_faster_rcnn.py | Baseline（无 CBAM） |
| 2 | ablation_channel.py | 仅通道注意力（CA Only） |
| 3 | ablation_spatial.py | 仅空间注意力（SA Only） |
| 4 | cbam_faster_rcnn.py | 完整 CBAM（r=16, k=7，论文默认） |

结果保存在 `experiments/results/ablation_{1-4}_*/`。

### 3.2 超参数实验（5 组，一键运行）

超参数实验在 **Baseline Faster R-CNN**（无 CBAM）基础上进行，排除注意力模块对结果的干扰，单独分析学习率和批量大小的影响。

```bash
bash pytorch_mmdet/tools/run_hyper.sh
```

| # | lr | batch_size | 说明 |
|---|-----|-----------|------|
| 1 | 0.005 | 4 | 学习率偏低 |
| 2 | 0.010 | 4 | 默认（复用消融实验 1 的结果，即 Baseline） |
| 3 | 0.020 | 4 | 学习率偏高 |
| 4 | 0.005 | 2 | 小批量（线性缩放 lr） |
| 5 | 0.020 | 8 | 大批量（线性缩放 lr） |

结果保存在 `experiments/results/hyper_*/`。

### 3.3 单独训练某个实验

```bash
python pytorch_mmdet/tools/train.py \
    pytorch_mmdet/configs/cbam_faster_rcnn.py \
    --work-dir experiments/results/my_exp \
    --seed 42
```

可选参数：
- `--resume`：从最新 checkpoint 继续训练（断点续训）
- `--cfg-options key=value`：从命令行覆盖配置项

### 3.4 训练输出

每个实验目录 `experiments/results/<exp_name>/` 包含：

| 文件 | 说明 |
|------|------|
| `training_log.csv` | 每 iteration 的 loss + 每 epoch 的 val_mAP |
| `best_pascal_voc_mAP_epoch_X.pth` | 最优 mAP 对应的权重 |
| `epoch_X.pth` | 最近 2 个 epoch 的 checkpoint |

---

## 4. 模型评估

对已训练的模型单独运行完整评估（每类 AP + mAP@0.5 + FPS）：

```bash
python pytorch_mmdet/tools/evaluate.py \
    pytorch_mmdet/configs/cbam_faster_rcnn.py \
    experiments/results/ablation_4_cbam_r16_k7/best_pascal_voc_mAP_epoch_12.pth \
    --out-dir experiments/results/ablation_4_cbam_r16_k7
```

加 `--show-dir` 可保存可视化检测图（每张测试图上画 bbox）：

```bash
python pytorch_mmdet/tools/evaluate.py \
    pytorch_mmdet/configs/cbam_faster_rcnn.py \
    experiments/results/ablation_4_cbam_r16_k7/best_pascal_voc_mAP_epoch_12.pth \
    --out-dir experiments/results/ablation_4_cbam_r16_k7 \
    --show-dir report/figures/detections/cbam
```

评估结果保存为 `eval_results.json`（`analyze_results.py` 会读取此文件）。

---

## 5. 结果汇总与可视化

```bash
python analyze_results.py              # 分析所有实验
python analyze_results.py --mode ablation  # 仅消融实验
python analyze_results.py --mode hyper     # 仅超参数实验
```

生成的表格和图表保存到 `report/figures/`。

> **前提**：需先对每个实验运行 evaluate.py，生成 `eval_results.json` 后才能汇总。

---

## 6. Jittor 版本

`jittor_impl/` 目录包含 Jittor 框架下的等价实现，代码结构与 PyTorch 版对应。
Jittor 版本仅需保证代码正确，无需实际运行训练。

| 文件 / 目录 | 说明 |
|-------------|------|
| `train_jittor.py` | 训练入口（支持 `--no-channel-attn --no-spatial-attn` 以禁用 CBAM，对应 Baseline） |
| `run_hyper_jittor.sh` | 一键运行全部超参数实验（Baseline 模式，与 PyTorch 版 `run_hyper.sh` 对应） |
| `models/` | 模型定义（ResNet50-CBAM、FPN、RPN、ROI Head） |
| `datasets/` | VOC 数据集加载 |
| `utils/` | 评估指标工具 |

---

## 项目结构

```
faster_rcnn_cbam/
├── install.sh                      # 一键安装脚本（AutoDL PyTorch 2.1 + CUDA 11.8）
├── requirements.txt                # Python 依赖清单
├── analyze_results.py              # 结果汇总与可视化
├── pytorch_mmdet/
│   ├── configs/
│   │   ├── base_faster_rcnn.py     # Baseline 配置（数据集/训练策略基准）
│   │   ├── cbam_faster_rcnn.py     # 完整 CBAM（r=16, k=7）
│   │   ├── ablation_channel.py     # 消融：仅通道注意力
│   │   ├── ablation_spatial.py     # 消融：仅空间注意力
│   │   ├── hyper_lr0005.py         # 超参：lr=0.005，bs=4
│   │   ├── hyper_lr002.py          # 超参：lr=0.02，bs=4
│   │   ├── hyper_bs2.py            # 超参：bs=2，lr=0.005
│   │   └── hyper_bs8.py            # 超参：bs=8，lr=0.02
│   ├── models/
│   │   ├── __init__.py             # 注册自定义模块
│   │   ├── cbam.py                 # CBAM 模块（通道注意力 + 空间注意力）
│   │   └── resnet_cbam.py          # 插入 CBAM 的 ResNet50 Backbone
│   ├── tools/
│   │   ├── train.py                # 训练入口（含 CSVLoggerHook）
│   │   ├── evaluate.py             # 评估脚本（mAP + FPS + 可视化）
│   │   ├── run_ablation.sh         # 一键运行全部消融实验
│   │   └── run_hyper.sh            # 一键运行全部超参数实验
│   └── utils/
│       ├── visualize.py            # 训练曲线可视化
│       └── show_results.py         # 检测结果展示
├── jittor_impl/                    # Jittor 框架等价实现
├── data/VOCdevkit/                 # 数据集（VOC2007 + VOC2012）
├── experiments/results/            # 训练输出（权重 + 日志 + eval_results.json）
└── report/figures/                 # 报告图表
```
