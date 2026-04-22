"""
Jittor 版 Faster R-CNN + CBAM 训练脚本

超参数与 PyTorch 版保持一致（便于框架间公平对比）：
  - 数据集：VOC2007 trainval + VOC2012 trainval，测试 VOC2007 test
  - Optimizer：SGD，lr=0.01，momentum=0.9，weight_decay=1e-4
  - LR 调度：MultiStepLR，milestones=[8, 11]，gamma=0.1
  - 训练 12 个 epoch，每 epoch 评估一次 mAP

运行示例：
    python jittor_impl/train_jittor.py \\
        --data-root data/VOCdevkit \\
        --epochs 12 \\
        --lr 0.01 \\
        --batch-size 4 \\
        --work-dir experiments/results/jittor_cbam
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict

# 将项目根目录加入 sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import jittor as jt
import jittor.nn as nn

from jittor_impl.models.faster_rcnn_jittor import FasterRCNNCBAM
from jittor_impl.datasets.voc_dataset_jittor import VOCDataset
from jittor_impl.utils.metrics_jittor import VOCEvaluator


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Jittor 版 Faster R-CNN + CBAM 训练')
    parser.add_argument('--data-root',   default='data/VOCdevkit')
    parser.add_argument('--work-dir',    default='experiments/results/jittor_cbam')
    parser.add_argument('--epochs',      type=int,   default=12)
    parser.add_argument('--lr',          type=float, default=0.01)
    parser.add_argument('--momentum',    type=float, default=0.9)
    parser.add_argument('--weight-decay',type=float, default=1e-4)
    parser.add_argument('--batch-size',  type=int,   default=2,
                        help='Jittor 建议小 batch（Jittor 默认单机）')
    parser.add_argument('--num-workers', type=int,   default=4)
    parser.add_argument('--milestones',  type=int,   nargs='+', default=[8, 11])
    parser.add_argument('--gamma',       type=float, default=0.1)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--use-gpu',     action='store_true', default=True)
    parser.add_argument('--cbam-reduction',    type=int,  default=16)
    parser.add_argument('--cbam-kernel-size',  type=int,  default=7)
    parser.add_argument('--no-channel-attn',   action='store_true')
    parser.add_argument('--no-spatial-attn',   action='store_true')
    parser.add_argument('--pretrained',        default=None,
                        help='torchvision ResNet50 预训练权重路径（.pth）')
    return parser.parse_args()


# ============================================================
# 学习率调度（MultiStepLR）
# ============================================================

def adjust_lr(optimizer: jt.optim.Optimizer, epoch: int, milestones: list, gamma: float, base_lr: float) -> float:
    """按 milestones 调整学习率。

    Args:
        optimizer  : Jittor 优化器
        epoch      : 当前 epoch（1-indexed）
        milestones : 衰减节点列表
        gamma      : 衰减系数
        base_lr    : 初始学习率

    Returns:
        current_lr: 当前学习率
    """
    lr = base_lr
    for m in milestones:
        if epoch >= m:
            lr *= gamma

    # 更新 Jittor 优化器的学习率
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    return lr


# ============================================================
# 评估函数
# ============================================================

def evaluate(model: FasterRCNNCBAM, test_dataset: VOCDataset) -> Dict:
    """在测试集上评估模型，返回 mAP 结果。

    Args:
        model       : 已训练的检测器
        test_dataset: VOC 测试集

    Returns:
        results: {'mAP': float, 'per_class_ap': dict}
    """
    model.eval()
    evaluator = VOCEvaluator(num_classes=20)

    # 逐图推理（不做 batch，避免 padding 影响）
    for idx in range(len(test_dataset)):
        img_tensor, gt_boxes, gt_labels = test_dataset[idx]

        # 扩展 batch 维度: [3, H, W] -> [1, 3, H, W]
        img_batch = img_tensor.unsqueeze(0)

        with jt.no_grad():
            output = model(img_batch)

        dets = output['detections'][0]

        # 转为 numpy（用于 evaluator）
        det_boxes  = dets['boxes'].numpy() if dets['boxes'].shape[0] > 0 else np.zeros((0, 4))
        det_scores = dets['scores'].numpy() if dets['scores'].shape[0] > 0 else np.zeros(0)
        det_labels = dets['labels'].numpy() if dets['labels'].shape[0] > 0 else np.zeros(0, dtype=np.int32)
        gt_boxes_np  = gt_boxes.numpy()
        gt_labels_np = gt_labels.numpy()

        evaluator.update(det_boxes, det_scores, det_labels, gt_boxes_np, gt_labels_np, img_id=idx)

        if (idx + 1) % 500 == 0:
            print(f'  评估进度: {idx + 1}/{len(test_dataset)}')

    return evaluator.compute()


# ============================================================
# CSV 日志工具
# ============================================================

_CSV_FIELDS = [
    'type', 'epoch', 'iter',
    'total_loss', 'rpn_cls_loss', 'rpn_reg_loss',
    'rcnn_cls_loss', 'rcnn_reg_loss', 'val_mAP',
]


def _write_iter_row(csv_path: str, epoch: int, it: int, losses: dict) -> None:
    """写入单个 iteration 的 loss 行。"""
    row = {
        'type': 'iter', 'epoch': epoch, 'iter': it,
        'total_loss':    f'{losses.get("loss", 0):.6f}',
        'rpn_cls_loss':  f'{losses.get("rpn_cls_loss", 0):.6f}',
        'rpn_reg_loss':  f'{losses.get("rpn_reg_loss", 0):.6f}',
        'rcnn_cls_loss': f'{losses.get("rcnn_cls_loss", 0):.6f}',
        'rcnn_reg_loss': f'{losses.get("rcnn_reg_loss", 0):.6f}',
        'val_mAP': '',
    }
    with open(csv_path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)


def _write_epoch_row(csv_path: str, epoch: int, val_map: float) -> None:
    """写入 epoch 结束的 mAP 行。"""
    row = {k: '' for k in _CSV_FIELDS}
    row.update({'type': 'epoch_end', 'epoch': epoch, 'val_mAP': f'{val_map:.4f}'})
    with open(csv_path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)


# ============================================================
# 主训练函数
# ============================================================

def main():
    args = parse_args()

    # 设置随机种子，保证复现性
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)

    # 启用 CUDA（若可用）
    if args.use_gpu and jt.has_cuda:
        jt.flags.use_cuda = 1
        print('[Jittor] 使用 GPU 训练')
    else:
        print('[Jittor] 使用 CPU 训练')

    # 创建输出目录
    os.makedirs(args.work_dir, exist_ok=True)
    csv_path = os.path.join(args.work_dir, 'training_log_jittor.csv')

    # 初始化 CSV 日志
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    # ---- 数据集 ----
    print('\n[1/4] 构建数据集...')
    train_dataset = VOCDataset(
        data_root=args.data_root,
        splits=[('VOC2007', 'trainval'), ('VOC2012', 'trainval')],
        is_train=True,
    )
    test_dataset = VOCDataset(
        data_root=args.data_root,
        splits=[('VOC2007', 'test')],
        is_train=False,
    )
    print(f'  训练集: {len(train_dataset)} 张，测试集: {len(test_dataset)} 张')

    # Jittor DataLoader：通过 set_attrs 配置批量参数
    train_dataset.set_attrs(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_batch=train_dataset.collate_batch,
    )

    # ---- 模型 ----
    print('\n[2/4] 构建模型...')
    model = FasterRCNNCBAM(
        num_classes=20,
        cbam_reduction=args.cbam_reduction,
        cbam_kernel_size=args.cbam_kernel_size,
        use_channel_attn=not args.no_channel_attn,
        use_spatial_attn=not args.no_spatial_attn,
        pretrained=args.pretrained,
    )
    model.train()

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  模型总参数: {total_params / 1e6:.2f}M')

    # ---- 优化器 ----
    print('\n[3/4] 配置优化器...')
    optimizer = jt.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # ---- 训练循环 ----
    print('\n[4/4] 开始训练...')
    best_map = 0.0
    best_ckpt_path = os.path.join(args.work_dir, 'best_model_jittor.pkl')

    for epoch in range(1, args.epochs + 1):
        model.train()

        # 调整学习率
        cur_lr = adjust_lr(optimizer, epoch, args.milestones, args.gamma, args.lr)
        print(f'\n====== Epoch {epoch}/{args.epochs}  lr={cur_lr:.6f} ======')

        epoch_loss_sum = 0.0
        iter_count = 0
        t0 = time.time()

        # 遍历训练批次
        for batch_idx, (images, boxes_list, labels_list) in enumerate(train_dataset):
            # images: [B, 3, H, W]，boxes_list/labels_list: list of jt.Var

            # 前向传播 + 计算 loss
            loss_dict = model(images, boxes_list, labels_list)
            total_loss = loss_dict['loss']

            # 反向传播 + 更新参数
            # Jittor 差异：optimizer.step(loss) 等价于 backward + step
            optimizer.step(total_loss)

            # 记录 loss
            loss_val = float(total_loss.numpy())
            epoch_loss_sum += loss_val
            iter_count += 1
            global_iter = (epoch - 1) * len(train_dataset) + batch_idx + 1

            # 提取各子 loss
            loss_np = {k: float(v.numpy()) for k, v in loss_dict.items()}

            # 每 50 个 iter 打印一次
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f'  [{batch_idx+1}/{len(train_dataset)}] '
                    f'loss={loss_val:.4f} '
                    f'rpn_cls={loss_np.get("rpn_cls_loss",0):.4f} '
                    f'rpn_reg={loss_np.get("rpn_reg_loss",0):.4f} '
                    f'rcnn_cls={loss_np.get("rcnn_cls_loss",0):.4f} '
                    f'rcnn_reg={loss_np.get("rcnn_reg_loss",0):.4f} '
                    f'({elapsed:.1f}s)'
                )

            # 写入 CSV
            _write_iter_row(csv_path, epoch, global_iter, loss_np)

        avg_loss = epoch_loss_sum / max(iter_count, 1)
        print(f'  Epoch {epoch} 平均 loss: {avg_loss:.4f}  耗时: {time.time()-t0:.1f}s')

        # ---- 每 epoch 评估 mAP ----
        print(f'  评估 VOC2007 test mAP...')
        results = evaluate(model, test_dataset)
        val_map = results['mAP']
        print(f'  mAP@0.5 = {val_map * 100:.2f}%')

        # 写入 epoch 结束行
        _write_epoch_row(csv_path, epoch, val_map)

        # 保存最优模型
        if val_map > best_map:
            best_map = val_map
            model.save(best_ckpt_path)
            print(f'  ✓ 新最优 mAP={best_map*100:.2f}%，已保存到 {best_ckpt_path}')

        # 每 epoch 保存一次 checkpoint
        ckpt_path = os.path.join(args.work_dir, f'epoch_{epoch}_jittor.pkl')
        model.save(ckpt_path)

    print('\n训练完成！')
    print(f'  最优 mAP: {best_map * 100:.2f}%')
    print(f'  最优模型: {best_ckpt_path}')
    print(f'  训练日志: {csv_path}')


if __name__ == '__main__':
    main()
