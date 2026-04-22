"""
训练脚本：基于 MMDetection v3.x（mmengine.Runner）的训练入口

使用示例：
    python pytorch_mmdet/tools/train.py \\
        pytorch_mmdet/configs/cbam_faster_rcnn.py \\
        --work-dir experiments/results/cbam \\
        --seed 42

训练过程自动记录到 <work-dir>/training_log.csv：
    - 每 iteration：total_loss、rpn_cls_loss、rpn_bbox_loss、rcnn_cls_loss、rcnn_bbox_loss
    - 每 epoch 结束：val_mAP（写入当前 epoch 最后一行）
"""

import argparse
import csv
import os
import sys

# 将项目根目录加入 sys.path，使 custom_imports 能找到 pytorch_mmdet.models
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mmengine
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


# ============================================================
# 自定义 Hook：将训练 loss 和验证 mAP 写入 CSV
# ============================================================
@HOOKS.register_module()
class CSVLoggerHook(Hook):
    """将每 iteration 的 loss 和每 epoch 的 val_mAP 记录到 CSV 文件。

    CSV 列：epoch, iter, total_loss, rpn_cls_loss, rpn_bbox_loss,
             rcnn_cls_loss, rcnn_bbox_loss, val_mAP

    对于普通训练迭代行，val_mAP 列为空；
    每个 epoch 结束后追加一行 type='epoch_end' 记录该 epoch 的 mAP。
    """

    # 优先级设为 NORMAL，在 LoggerHook（VERY_LOW）之前执行
    priority = 'NORMAL'

    _FIELDS = [
        'type', 'epoch', 'iter',
        'total_loss', 'rpn_cls_loss', 'rpn_bbox_loss',
        'rcnn_cls_loss', 'rcnn_bbox_loss', 'val_mAP',
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def before_run(self, runner) -> None:
        """训练开始前创建 CSV 文件并写入表头。"""
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        with open(self.csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self._FIELDS).writeheader()

    def after_train_iter(self, runner, batch_idx, data_batch, outputs) -> None:
        """每个 iteration 结束后记录 loss。

        在 MMDetection v3.x 中，outputs 是 BaseDetector.train_step 的返回值，
        即 parse_losses 后的 log_vars 字典（float 类型）。
        """
        if not isinstance(outputs, dict):
            return

        def _get(key):
            """安全取值，缺失时返回空字符串。"""
            val = outputs.get(key, None)
            if val is None:
                return ''
            return f'{float(val):.6f}' if isinstance(val, (int, float)) else str(val)

        row = {
            'type': 'iter',
            'epoch': runner.epoch + 1,
            'iter': runner.iter + 1,
            'total_loss': _get('loss'),
            'rpn_cls_loss': _get('loss_rpn_cls'),
            'rpn_bbox_loss': _get('loss_rpn_bbox'),
            'rcnn_cls_loss': _get('loss_cls'),
            'rcnn_bbox_loss': _get('loss_bbox'),
            'val_mAP': '',
        }
        with open(self.csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self._FIELDS).writerow(row)

    def after_val_epoch(self, runner, metrics) -> None:
        """每个 epoch 验证结束后，将 mAP 追加为一行 epoch_end 记录。"""
        val_map = metrics.get('pascal_voc/mAP', metrics.get('mAP', ''))
        row = {
            'type': 'epoch_end',
            'epoch': runner.epoch,          # 此时 epoch 已自增
            'iter': '',
            'total_loss': '', 'rpn_cls_loss': '', 'rpn_bbox_loss': '',
            'rcnn_cls_loss': '', 'rcnn_bbox_loss': '',
            'val_mAP': f'{float(val_map):.4f}' if val_map != '' else '',
        }
        with open(self.csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self._FIELDS).writerow(row)


# ============================================================
# 命令行参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection Faster R-CNN 训练脚本')
    parser.add_argument('config', help='配置文件路径（.py）')
    parser.add_argument(
        '--work-dir',
        help='输出目录（checkpoint、日志），默认 experiments/results/<config名>/',
    )
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='从 checkpoint 继续训练。不带参数时自动找最新 ckpt；可指定 .pth 路径',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子，保证复现性（默认 42）',
    )
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        default=[0],
        help='使用的 GPU id 列表（默认 [0]）',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='从命令行覆盖 config 中的字段，格式：key=value',
    )
    return parser.parse_args()


# ============================================================
# 主函数
# ============================================================
def main():
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(args.config)

    # 命令行覆盖配置
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 设置输出目录
    if args.work_dir is None:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        args.work_dir = os.path.join('experiments', 'results', config_name)
    cfg.work_dir = args.work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)

    # 设置随机种子
    cfg.seed = args.seed

    # 设置 GPU
    cfg.gpu_ids = args.gpu_ids

    # 自动限制 num_workers，避免在 CPU 核心有限的服务器上过载
    import multiprocessing
    max_workers = max(1, min(4, multiprocessing.cpu_count() // 2))
    if hasattr(cfg, 'train_dataloader'):
        cfg.train_dataloader.num_workers = min(
            cfg.train_dataloader.get('num_workers', 4), max_workers
        )
    if hasattr(cfg, 'val_dataloader'):
        cfg.val_dataloader.num_workers = min(
            cfg.val_dataloader.get('num_workers', 2), max_workers
        )

    # 注入 CSVLoggerHook
    csv_path = os.path.join(cfg.work_dir, 'training_log.csv')
    if not hasattr(cfg, 'custom_hooks'):
        cfg.custom_hooks = []
    cfg.custom_hooks.append(dict(type='CSVLoggerHook', csv_path=csv_path))

    # Resume 设置
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # 构建并启动 Runner
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
