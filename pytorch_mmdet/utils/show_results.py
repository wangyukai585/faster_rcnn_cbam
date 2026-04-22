"""
检测结果可视化脚本

从 VOC2007 test 随机选 12 张图，分别用 Baseline 和 CBAM 模型推理，
生成左右对比图保存到 report/figures/detection_comparison.png：
  - 左列：Baseline 检测结果
  - 右列：CBAM 检测结果

使用示例：
    python pytorch_mmdet/utils/show_results.py \\
        --baseline-config pytorch_mmdet/configs/base_faster_rcnn.py \\
        --baseline-ckpt   experiments/results/ablation_1_baseline/best_pascal_voc_mAP.pth \\
        --cbam-config     pytorch_mmdet/configs/cbam_faster_rcnn.py \\
        --cbam-ckpt       experiments/results/ablation_4_cbam_r16_k7/best_pascal_voc_mAP.pth \\
        --out-dir         report/figures \\
        --num-images 12
"""

import argparse
import os
import random
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

# VOC 20 类名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

# 每类对应的颜色（HSV 均匀分布）
_COLORS = [
    tuple(int(c * 255) for c in plt.cm.hsv(i / len(VOC_CLASSES))[:3])
    for i in range(len(VOC_CLASSES))
]


def _build_model(config_path: str, checkpoint_path: str):
    """加载 MMDetection 模型（推理模式）。"""
    from mmengine.config import Config
    from mmdet.apis import init_detector
    cfg = Config.fromfile(config_path)
    model = init_detector(cfg, checkpoint_path, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return model


def _infer(model, img_path: str, score_thr: float = 0.3):
    """对单张图片推理，返回检测结果列表。

    Returns:
        list of dict: [{'bbox': [x1,y1,x2,y2], 'label': int, 'score': float}, ...]
    """
    from mmdet.apis import inference_detector
    result = inference_detector(model, img_path)
    detections = []
    pred_instances = result.pred_instances
    for i in range(len(pred_instances)):
        score = float(pred_instances.scores[i])
        if score < score_thr:
            continue
        bbox = pred_instances.bboxes[i].cpu().numpy().tolist()
        label = int(pred_instances.labels[i])
        detections.append({'bbox': bbox, 'label': label, 'score': score})
    return detections


def _draw_detections(ax, img: np.ndarray, detections: list, title: str = '') -> None:
    """在 matplotlib Axes 上绘制检测框和标签。

    Args:
        ax         : matplotlib 子图
        img        : RGB 图像数组 [H, W, 3]
        detections : [{'bbox', 'label', 'score'}, ...]
        title      : 子图标题
    """
    ax.imshow(img)
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis('off')

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label_id = det['label']
        score = det['score']
        cls_name = VOC_CLASSES[label_id] if label_id < len(VOC_CLASSES) else str(label_id)
        color = tuple(c / 255.0 for c in _COLORS[label_id % len(_COLORS)])

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor='none',
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 2,
            f'{cls_name} {score:.2f}',
            color='white', fontsize=7, fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor='none'),
        )


def _collect_test_images(data_root: str, num_images: int = 12, seed: int = 42) -> list:
    """从 VOC2007 test 集随机采样图片路径。"""
    test_list = os.path.join(data_root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
    img_dir = os.path.join(data_root, 'VOC2007', 'JPEGImages')

    if not os.path.exists(test_list):
        raise FileNotFoundError(f'未找到测试集列表: {test_list}')

    with open(test_list) as f:
        img_names = [line.strip() for line in f if line.strip()]

    random.seed(seed)
    selected = random.sample(img_names, min(num_images, len(img_names)))
    return [os.path.join(img_dir, f'{name}.jpg') for name in selected]


def generate_comparison(
    baseline_config: str,
    baseline_ckpt: str,
    cbam_config: str,
    cbam_ckpt: str,
    data_root: str = 'data/VOCdevkit',
    out_dir: str = 'report/figures',
    num_images: int = 12,
    score_thr: float = 0.35,
    seed: int = 42,
) -> None:
    """主函数：生成 Baseline vs CBAM 的检测结果对比图。

    布局：num_images 行 × 2 列（左：Baseline，右：CBAM）
    """
    print('加载 Baseline 模型...')
    baseline_model = _build_model(baseline_config, baseline_ckpt)

    print('加载 CBAM 模型...')
    cbam_model = _build_model(cbam_config, cbam_ckpt)

    print('采样测试图片...')
    img_paths = _collect_test_images(data_root, num_images, seed)

    nrows = num_images
    fig, axes = plt.subplots(nrows, 2, figsize=(12, nrows * 3.5))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    print(f'推理 {len(img_paths)} 张图片...')
    for row_idx, img_path in enumerate(img_paths):
        img = np.array(Image.open(img_path).convert('RGB'))
        img_name = os.path.basename(img_path)

        # Baseline 推理
        baseline_dets = _infer(baseline_model, img_path, score_thr)
        _draw_detections(
            axes[row_idx, 0], img, baseline_dets,
            title=f'Baseline | {img_name}',
        )

        # CBAM 推理
        cbam_dets = _infer(cbam_model, img_path, score_thr)
        _draw_detections(
            axes[row_idx, 1], img, cbam_dets,
            title=f'CBAM | {img_name}',
        )

    fig.suptitle(
        'Detection Comparison: Baseline vs CBAM (Faster R-CNN, VOC2007)',
        fontsize=13, fontweight='bold', y=1.002,
    )
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'detection_comparison.png')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'对比图已保存: {save_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='检测结果 Baseline vs CBAM 可视化')
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-ckpt',   required=True)
    parser.add_argument('--cbam-config',      required=True)
    parser.add_argument('--cbam-ckpt',        required=True)
    parser.add_argument('--data-root', default='data/VOCdevkit')
    parser.add_argument('--out-dir',   default='report/figures')
    parser.add_argument('--num-images', type=int, default=12)
    parser.add_argument('--score-thr',  type=float, default=0.35)
    parser.add_argument('--seed',       type=int,   default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_comparison(
        baseline_config=args.baseline_config,
        baseline_ckpt=args.baseline_ckpt,
        cbam_config=args.cbam_config,
        cbam_ckpt=args.cbam_ckpt,
        data_root=args.data_root,
        out_dir=args.out_dir,
        num_images=args.num_images,
        score_thr=args.score_thr,
        seed=args.seed,
    )
