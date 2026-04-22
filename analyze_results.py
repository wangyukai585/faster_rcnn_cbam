"""
实验结果汇总分析脚本

功能：
  1. 自动扫描 experiments/results/ 下所有实验目录
  2. 读取每个实验的 eval_results.json
  3. 打印消融实验结果表和超参数实验结果表
  4. 调用 visualize.py 生成所有图表
  5. 输出 final_report_data.json 供报告使用

使用示例：
    python analyze_results.py              # 分析所有实验
    python analyze_results.py --mode ablation  # 仅消融实验
    python analyze_results.py --mode hyper     # 仅超参数实验
"""

import argparse
import json
import os
import sys

# 将项目根目录加入 sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ============================================================
# 实验定义
# ============================================================

# 消融实验：6组，按"从无到有"的逻辑排列
_ABLATION_EXPS = [
    {
        'dir':         'ablation_1_baseline',
        'name':        'Baseline（无CBAM）',
        'config':      'base_faster_rcnn.py',
        'cbam_params': 'N/A',
    },
    {
        'dir':         'ablation_2_channel_only',
        'name':        '仅通道注意力（CA Only）',
        'config':      'ablation_channel.py',
        'cbam_params': 'CA only',
    },
    {
        'dir':         'ablation_3_spatial_only',
        'name':        '仅空间注意力（SA Only）',
        'config':      'ablation_spatial.py',
        'cbam_params': 'SA only',
    },
    {
        'dir':         'ablation_4_cbam_r16_k7',
        'name':        '完整CBAM（r=16, k=7）',
        'config':      'cbam_faster_rcnn.py',
        'cbam_params': 'r=16, k=7',
    },
    {
        'dir':         'ablation_5_cbam_r8_k7',
        'name':        'CBAM（r=8, k=7）',
        'config':      'ablation_cbam_r8.py',
        'cbam_params': 'r=8, k=7',
    },
    {
        'dir':         'ablation_6_cbam_r16_k3',
        'name':        'CBAM（r=16, k=3）',
        'config':      'ablation_cbam_k3.py',
        'cbam_params': 'r=16, k=3',
    },
]

# 超参数实验：5组
_HYPER_EXPS = [
    {
        'dir':  'hyper_lr0005_bs4',
        'name': 'lr=0.005, bs=4',
        'lr':   0.005,
        'bs':   4,
    },
    {
        'dir':  'hyper_lr001_bs4',
        'name': 'lr=0.01,  bs=4（默认）',
        'lr':   0.01,
        'bs':   4,
    },
    {
        'dir':  'hyper_lr002_bs4',
        'name': 'lr=0.02,  bs=4',
        'lr':   0.02,
        'bs':   4,
    },
    {
        'dir':  'hyper_lr0005_bs2',
        'name': 'lr=0.005, bs=2',
        'lr':   0.005,
        'bs':   2,
    },
    {
        'dir':  'hyper_lr002_bs8',
        'name': 'lr=0.02,  bs=8',
        'lr':   0.02,
        'bs':   8,
    },
]


# ============================================================
# 辅助函数
# ============================================================

def _load_result(results_dir: str, exp_dir: str) -> dict:
    """读取实验目录下的 eval_results.json，不存在则返回占位值。"""
    json_path = os.path.join(results_dir, exp_dir, 'eval_results.json')
    if not os.path.exists(json_path):
        return {'mAP': None, 'fps': None, 'inference_ms': None}
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_best_epoch(results_dir: str, exp_dir: str) -> int:
    """从 training_log.csv 读取最高 val_mAP 所在 epoch。"""
    csv_path = os.path.join(results_dir, exp_dir, 'training_log.csv')
    if not os.path.exists(csv_path):
        return -1
    try:
        import csv
        best_map, best_epoch = -1.0, -1
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('type') == 'epoch_end':
                    try:
                        val = float(row.get('val_mAP', '0') or '0')
                        ep = int(row.get('epoch', '0') or '0')
                        if val > best_map:
                            best_map, best_epoch = val, ep
                    except ValueError:
                        pass
        return best_epoch
    except Exception:
        return -1


def _estimate_extra_params(cbam_params: str) -> str:
    """根据 CBAM 参数估算额外参数量（相对于标准 ResNet50 的增量）。

    ResNet50 各层 Bottleneck 输出通道：256, 512, 1024, 2048
    每层块数：3, 4, 6, 3  共 16 个 Bottleneck

    ChannelAttention 参数量（per block）：
        2 * (C * C//r + C//r * C) = 4 * C^2 / r

    SpatialAttention 参数量（per block）：
        2 * 1 * k * k = 2 * k^2（卷积核，无 bias）
    """
    if cbam_params == 'N/A':
        return '0'

    configs_map = {
        'r=16, k=7': (16, 7),
        'r=8, k=7':  (8, 7),
        'r=16, k=3': (16, 3),
        'CA only':   (16, 0),  # 无空间注意力
        'SA only':   (0, 7),   # 无通道注意力
    }
    if cbam_params not in configs_map:
        return '?'

    r, k = configs_map[cbam_params]
    channels = [256, 512, 1024, 2048]
    num_blocks = [3, 4, 6, 3]

    total = 0
    for C, n in zip(channels, num_blocks):
        per_block = 0
        # 通道注意力 MLP 参数（两路共享）
        if r > 0:
            per_block += 2 * (C * (C // r) + (C // r) * C)
        # 空间注意力卷积参数
        if k > 0:
            per_block += 2 * k * k   # in_channels=2, out=1, kernel_size=k
        total += per_block * n

    total_k = total / 1000
    return f'+{total_k:.1f}K'


def _fmt(val, fmt=':.1f', suffix='%', none_str='--'):
    """格式化数值，None 时返回占位符。"""
    if val is None:
        return none_str
    return f'{val:{fmt[1:]}}{suffix}'


# ============================================================
# 打印表格
# ============================================================

def _print_separator(widths: list, char: str = '─') -> None:
    parts = [char * (w + 2) for w in widths]
    print('┌' + '┬'.join(parts) + '┐')


def _print_row(cells: list, widths: list) -> None:
    padded = [f' {str(c):<{w}s} ' for c, w in zip(cells, widths)]
    print('│' + '│'.join(padded) + '│')


def _print_table(headers: list, rows: list) -> None:
    """打印带边框的表格。"""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep_top    = '┌' + '┬'.join('─' * (w + 2) for w in widths) + '┐'
    sep_mid    = '├' + '┼'.join('─' * (w + 2) for w in widths) + '┤'
    sep_bottom = '└' + '┴'.join('─' * (w + 2) for w in widths) + '┘'

    print(sep_top)
    _print_row(headers, widths)
    print(sep_mid)
    for row in rows:
        _print_row(row, widths)
    print(sep_bottom)


# ============================================================
# 消融实验分析
# ============================================================

def analyze_ablation(results_dir: str) -> list:
    """汇总消融实验结果，打印表格，返回数据列表。"""
    print('\n' + '=' * 70)
    print('                   【消融实验结果表】')
    print('=' * 70)

    headers = ['实验配置', 'mAP@0.5', '参数量增加', '推理时间(ms)']
    rows = []
    data_list = []

    baseline_map = None

    for exp in _ABLATION_EXPS:
        result = _load_result(results_dir, exp['dir'])
        mAP    = result.get('mAP')
        ms     = result.get('inference_ms')
        params = _estimate_extra_params(exp['cbam_params'])

        if mAP is not None and baseline_map is None:
            baseline_map = mAP

        map_str = _fmt(mAP * 100 if mAP is not None else None, ':.1f', '%')
        ms_str  = _fmt(ms, ':.1f', '')
        rows.append([exp['name'], map_str, params, ms_str])

        data_list.append({
            'experiment': exp['name'],
            'dir': exp['dir'],
            'mAP': mAP,
            'mAP_pct': mAP * 100 if mAP is not None else None,
            'extra_params': params,
            'inference_ms': ms,
            'gain_over_baseline': (
                (mAP - baseline_map) * 100
                if mAP is not None and baseline_map is not None
                else None
            ),
        })

    _print_table(headers, rows)
    return data_list


# ============================================================
# 超参数实验分析
# ============================================================

def analyze_hyper(results_dir: str) -> list:
    """汇总超参数实验结果，打印表格，返回数据列表。"""
    print('\n' + '=' * 60)
    print('                 【超参数实验结果表】')
    print('=' * 60)

    headers = ['超参数配置', 'mAP@0.5', '收敛Epoch']
    rows = []
    data_list = []

    for exp in _HYPER_EXPS:
        result       = _load_result(results_dir, exp['dir'])
        mAP          = result.get('mAP')
        best_epoch   = _load_best_epoch(results_dir, exp['dir'])

        map_str   = _fmt(mAP * 100 if mAP is not None else None, ':.1f', '%')
        epoch_str = str(best_epoch) if best_epoch > 0 else '--'
        rows.append([exp['name'], map_str, epoch_str])

        data_list.append({
            'experiment': exp['name'],
            'dir': exp['dir'],
            'lr': exp['lr'],
            'batch_size': exp['bs'],
            'mAP': mAP,
            'mAP_pct': mAP * 100 if mAP is not None else None,
            'best_epoch': best_epoch if best_epoch > 0 else None,
        })

    _print_table(headers, rows)
    return data_list


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='实验结果汇总分析')
    parser.add_argument(
        '--results-dir', default='experiments/results',
        help='实验结果根目录（默认 experiments/results/）',
    )
    parser.add_argument(
        '--out-dir', default='report/figures',
        help='图表输出目录（默认 report/figures/）',
    )
    parser.add_argument(
        '--mode', choices=['all', 'ablation', 'hyper'], default='all',
        help='分析模式（默认 all）',
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ablation_data, hyper_data = [], []

    if args.mode in ('all', 'ablation'):
        ablation_data = analyze_ablation(args.results_dir)

    if args.mode in ('all', 'hyper'):
        hyper_data = analyze_hyper(args.results_dir)

    # ---- 保存汇总 JSON ----
    report_data = {
        'ablation_results': ablation_data,
        'hyper_results':    hyper_data,
    }
    json_path = os.path.join(args.out_dir, 'final_report_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f'\n[analyze] 汇总数据已保存: {json_path}')

    # ---- 生成所有可视化图表 ----
    try:
        from pytorch_mmdet.utils.visualize import plot_all
        plot_all(results_dir=args.results_dir, out_dir=args.out_dir)
    except Exception as e:
        print(f'[analyze] 图表生成失败（可能是实验数据不完整）: {e}')

    print('\n[analyze] 分析完成！')
    print(f'  - 汇总数据: {json_path}')
    print(f'  - 图表目录: {args.out_dir}/')


if __name__ == '__main__':
    main()
