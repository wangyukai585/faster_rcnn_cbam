"""
评估脚本：对已训练好的模型在 VOC2007 test 上进行推理和评估

使用示例：
    python pytorch_mmdet/tools/evaluate.py \\
        pytorch_mmdet/configs/cbam_faster_rcnn.py \\
        experiments/results/cbam/best_pascal_voc_mAP_epoch_12.pth \\
        --show-dir report/figures/detections/

输出：
    - 每个类别的 AP 值（20类）+ 总 mAP@0.5
    - 推理速度（FPS）
    - eval_results.json（保存到 checkpoint 所在目录）
"""

import argparse
import json
import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


# ============================================================
# 命令行参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection 模型评估脚本')
    parser.add_argument('config', help='配置文件路径（.py）')
    parser.add_argument('checkpoint', help='模型权重文件路径（.pth）')
    parser.add_argument(
        '--show-dir',
        help='可视化检测结果的保存目录（不指定则不保存可视化）',
    )
    parser.add_argument(
        '--out-dir',
        help='eval_results.json 保存目录（默认与 checkpoint 同级）',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='从命令行覆盖 config 中的字段',
    )
    return parser.parse_args()


# ============================================================
# 推理速度测量
# ============================================================
def measure_fps(runner: Runner, num_warmup: int = 5, num_test: int = 50) -> float:
    """在测试集上测量模型推理 FPS。

    Args:
        runner: 已构建的 MMDetection Runner
        num_warmup: 预热轮数（不计时）
        num_test: 正式计时轮数

    Returns:
        FPS（float）
    """
    model = runner.model
    model.eval()
    dataloader = runner.test_dataloader

    total_time = 0.0
    count = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = model.data_preprocessor(data, False)
            if i < num_warmup:
                # 预热，不计时
                model(**data, mode='predict')
                continue
            t0 = time.perf_counter()
            model(**data, mode='predict')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            count += 1
            if count >= num_test:
                break

    fps = count / total_time if total_time > 0 else 0.0
    return fps


# ============================================================
# 主函数
# ============================================================
def main():
    args = parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # 设置 checkpoint 加载路径
    cfg.load_from = args.checkpoint

    # 若指定可视化目录，开启检测结果可视化
    if args.show_dir:
        os.makedirs(args.show_dir, exist_ok=True)
        cfg.default_hooks['visualization'] = dict(
            type='DetVisualizationHook',
            draw=True,
            interval=1,
            show=False,
        )
        cfg.visualizer = dict(
            type='DetLocalVisualizer',
            vis_backends=[dict(type='LocalVisBackend', save_dir=args.show_dir)],
            name='visualizer',
        )

    # work_dir 用于存放评估日志（与 checkpoint 同级）
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    cfg.work_dir = ckpt_dir

    # 构建 Runner 并执行测试
    runner = Runner.from_cfg(cfg)

    print('\n' + '=' * 60)
    print('开始评估...')
    print('=' * 60)
    metrics = runner.test()

    # ---- 打印每类 AP ----
    print('\n' + '=' * 60)
    print('各类别 AP 值：')
    print('=' * 60)
    per_class_ap = {}
    for key, val in metrics.items():
        if key.startswith('pascal_voc/') and key != 'pascal_voc/mAP':
            cls_name = key.replace('pascal_voc/', '')
            per_class_ap[cls_name] = float(val)
            print(f'  {cls_name:<20s}: {float(val) * 100:.2f}%')

    map_val = float(metrics.get('pascal_voc/mAP', 0))
    print('-' * 60)
    print(f'  {"mAP@0.5":<20s}: {map_val * 100:.2f}%')

    # ---- 推理速度 ----
    print('\n测量推理速度...')
    fps = measure_fps(runner)
    inference_ms = 1000.0 / fps if fps > 0 else 0.0
    print(f'  推理速度: {fps:.1f} FPS ({inference_ms:.1f} ms/image)')

    # ---- 保存结果到 JSON ----
    out_dir = args.out_dir or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'eval_results.json')
    result_dict = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'mAP': map_val,
        'fps': fps,
        'inference_ms': inference_ms,
        'per_class_ap': per_class_ap,
        'raw_metrics': {k: float(v) for k, v in metrics.items()},
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    print(f'\n评估结果已保存到: {json_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
