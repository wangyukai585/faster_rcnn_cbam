"""
训练过程可视化脚本

读取各实验的 training_log.csv，生成以下图表（保存到 report/figures/）：
  - training_loss_curve.png ：Baseline vs CBAM 的训练 loss 曲线
  - val_map_curve.png        ：Baseline vs CBAM 的 val mAP 曲线
  - ablation_bar_chart.png   ：消融实验柱状图
  - hyper_comparison.png     ：超参数对比图（lr + bs 各一个子图）

使用方式：
    from pytorch_mmdet.utils.visualize import plot_all
    plot_all(results_dir='experiments/results', out_dir='report/figures')
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')   # 无显示器环境下不弹窗
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# 统一图表风格
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})

# 各实验的显示名称和颜色
_EXP_STYLES: Dict[str, Tuple[str, str]] = {
    'ablation_1_baseline':     ('Baseline',            '#2196F3'),
    'ablation_2_channel_only': ('CA Only',             '#FF9800'),
    'ablation_3_spatial_only': ('SA Only',             '#9C27B0'),
    'ablation_4_cbam_r16_k7':  ('Full CBAM (r=16,k=7)', '#F44336'),
    'hyper_lr0005_bs4':        ('lr=0.005, bs=4',     '#00BCD4'),
    'hyper_lr001_bs4':         ('lr=0.01,  bs=4 (默认)', '#F44336'),
    'hyper_lr002_bs4':         ('lr=0.02,  bs=4',     '#FF5722'),
    'hyper_lr0005_bs2':        ('lr=0.005, bs=2',     '#8BC34A'),
    'hyper_lr002_bs8':         ('lr=0.02,  bs=8',     '#3F51B5'),
}


# ============================================================
# 数据读取辅助函数
# ============================================================

def _load_csv(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取 training_log.csv，分离 iter 行和 epoch_end 行。

    Returns:
        iter_df  : 按 iteration 的 loss 数据
        epoch_df : 按 epoch 的 val_mAP 数据
    """
    df = pd.read_csv(csv_path)
    iter_df = df[df['type'] == 'iter'].copy()
    epoch_df = df[df['type'] == 'epoch_end'].copy()
    iter_df['epoch'] = iter_df['epoch'].astype(int)
    iter_df['iter'] = iter_df['iter'].astype(int)
    epoch_df['epoch'] = epoch_df['epoch'].astype(int)
    epoch_df['val_mAP'] = pd.to_numeric(epoch_df['val_mAP'], errors='coerce')
    return iter_df, epoch_df


def _load_eval_json(exp_dir: str) -> Optional[Dict]:
    """读取实验目录下的 eval_results.json。"""
    json_path = os.path.join(exp_dir, 'eval_results.json')
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _smooth(values: List[float], window: int = 10) -> List[float]:
    """简单移动平均平滑，用于 loss 曲线降噪。"""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    return list(np.convolve(values, kernel, mode='same'))


# ============================================================
# 图 1：训练 Loss 曲线
# ============================================================

def plot_loss_curve(
    results_dir: str,
    out_dir: str,
    exp_keys: Optional[List[str]] = None,
) -> None:
    """绘制多个实验的训练 loss 曲线（epoch 均值），保存为 training_loss_curve.png。

    Args:
        results_dir: 实验结果根目录
        out_dir    : 图片保存目录
        exp_keys   : 要展示的实验目录名列表（默认展示 baseline 和 cbam）
    """
    if exp_keys is None:
        exp_keys = ['ablation_1_baseline', 'ablation_4_cbam_r16_k7']

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for exp_key in exp_keys:
        csv_path = os.path.join(results_dir, exp_key, 'training_log.csv')
        if not os.path.exists(csv_path):
            continue

        iter_df, _ = _load_csv(csv_path)
        if iter_df.empty:
            continue

        # 按 epoch 分组，计算均值 loss
        loss_col = 'total_loss' if 'total_loss' in iter_df.columns else iter_df.columns[3]
        epoch_loss = iter_df.groupby('epoch')[loss_col].mean().reset_index()

        label, color = _EXP_STYLES.get(exp_key, (exp_key, None))
        ax.plot(
            epoch_loss['epoch'], epoch_loss[loss_col],
            label=label, color=color, linewidth=2, marker='o', markersize=4,
        )
        plotted = True

    if not plotted:
        print('[visualize] 未找到 training_log.csv，跳过 loss 曲线绘制。')
        plt.close(fig)
        return

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curve：Baseline vs CBAM')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'training_loss_curve.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[visualize] Loss 曲线已保存: {save_path}')


# ============================================================
# 图 2：验证集 mAP 曲线
# ============================================================

def plot_map_curve(
    results_dir: str,
    out_dir: str,
    exp_keys: Optional[List[str]] = None,
) -> None:
    """绘制多个实验的 val mAP@0.5 曲线，保存为 val_map_curve.png。"""
    if exp_keys is None:
        exp_keys = ['ablation_1_baseline', 'ablation_4_cbam_r16_k7']

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for exp_key in exp_keys:
        csv_path = os.path.join(results_dir, exp_key, 'training_log.csv')
        if not os.path.exists(csv_path):
            continue

        _, epoch_df = _load_csv(csv_path)
        if epoch_df.empty or epoch_df['val_mAP'].isna().all():
            continue

        label, color = _EXP_STYLES.get(exp_key, (exp_key, None))
        epochs = epoch_df['epoch'].tolist()
        maps = (epoch_df['val_mAP'] * 100).tolist()

        ax.plot(epochs, maps, label=label, color=color, linewidth=2,
                marker='o', markersize=4)

        # 标注最高 mAP 点
        best_idx = int(np.argmax(maps))
        ax.annotate(
            f'{maps[best_idx]:.1f}%',
            xy=(epochs[best_idx], maps[best_idx]),
            xytext=(epochs[best_idx] + 0.3, maps[best_idx] + 0.3),
            fontsize=9, color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
        )
        plotted = True

    if not plotted:
        print('[visualize] 未找到 epoch_end 数据，跳过 mAP 曲线绘制。')
        plt.close(fig)
        return

    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@0.5 (%)')
    ax.set_title('Validation mAP Curve：Baseline vs CBAM')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'val_map_curve.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[visualize] mAP 曲线已保存: {save_path}')


# ============================================================
# 图 3：消融实验柱状图
# ============================================================

_ABLATION_KEYS = [
    'ablation_1_baseline',
    'ablation_2_channel_only',
    'ablation_3_spatial_only',
    'ablation_4_cbam_r16_k7',
]


def plot_ablation_bar(results_dir: str, out_dir: str) -> None:
    """绘制消融实验柱状图，保存为 ablation_bar_chart.png。"""
    labels, maps, colors = [], [], []

    for exp_key in _ABLATION_KEYS:
        data = _load_eval_json(os.path.join(results_dir, exp_key))
        label, color = _EXP_STYLES.get(exp_key, (exp_key, '#9E9E9E'))
        if data is not None:
            maps.append(data.get('mAP', 0) * 100)
        else:
            maps.append(0.0)   # 实验未完成时填充 0
        labels.append(label)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), maps, color=colors, width=0.6, edgecolor='white')

    # 突出显示完整 CBAM 的柱子（实验 4）
    if len(bars) >= 4:
        bars[3].set_edgecolor('#212121')
        bars[3].set_linewidth(2.5)

    # 在每根柱子上标注数值
    for bar, val in zip(bars, maps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('mAP@0.5 (%)')
    ax.set_title('消融实验：各注意力组件对 mAP 的贡献')
    y_max = max(maps) * 1.12 if maps else 100
    ax.set_ylim(0, y_max if y_max > 0 else 100)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'ablation_bar_chart.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[visualize] 消融实验柱状图已保存: {save_path}')


# ============================================================
# 图 4：超参数对比图
# ============================================================

def plot_hyper_comparison(results_dir: str, out_dir: str) -> None:
    """绘制超参数对比图（lr + bs 各一个子图），保存为 hyper_comparison.png。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- 子图 1：学习率对比（bs=4 固定）----
    lr_exps = [
        ('hyper_lr0005_bs4', 'lr=0.005'),
        ('hyper_lr001_bs4',  'lr=0.01 (默认)'),
        ('hyper_lr002_bs4',  'lr=0.02'),
    ]
    lr_vals, lr_maps, lr_colors = [], [], []
    for exp_key, label in lr_exps:
        data = _load_eval_json(os.path.join(results_dir, exp_key))
        _, color = _EXP_STYLES.get(exp_key, (exp_key, '#9E9E9E'))
        lr_vals.append(label)
        lr_maps.append(data['mAP'] * 100 if data else 0.0)
        lr_colors.append(color)

    bars = axes[0].bar(range(len(lr_vals)), lr_maps, color=lr_colors, width=0.5)
    for bar, val in zip(bars, lr_maps):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
        )
    axes[0].set_xticks(range(len(lr_vals)))
    axes[0].set_xticklabels(lr_vals)
    axes[0].set_ylabel('mAP@0.5 (%)')
    axes[0].set_title('学习率对比（batch_size=4）')
    lr_y_max = max(lr_maps) * 1.12 if lr_maps else 100
    axes[0].set_ylim(0, lr_y_max if lr_y_max > 0 else 100)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    # ---- 子图 2：Batch Size 对比----
    bs_exps = [
        ('hyper_lr0005_bs2',  'bs=2 (lr=0.005)'),
        ('hyper_lr001_bs4',   'bs=4 (默认)'),
        ('hyper_lr002_bs8',   'bs=8 (lr=0.02)'),
    ]
    bs_vals, bs_maps, bs_colors = [], [], []
    for exp_key, label in bs_exps:
        data = _load_eval_json(os.path.join(results_dir, exp_key))
        _, color = _EXP_STYLES.get(exp_key, (exp_key, '#9E9E9E'))
        bs_vals.append(label)
        bs_maps.append(data['mAP'] * 100 if data else 0.0)
        bs_colors.append(color)

    bars = axes[1].bar(range(len(bs_vals)), bs_maps, color=bs_colors, width=0.5)
    for bar, val in zip(bars, bs_maps):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
        )
    axes[1].set_xticks(range(len(bs_vals)))
    axes[1].set_xticklabels(bs_vals)
    axes[1].set_ylabel('mAP@0.5 (%)')
    axes[1].set_title('Batch Size 对比（线性缩放 lr）')
    bs_y_max = max(bs_maps) * 1.12 if bs_maps else 100
    axes[1].set_ylim(0, bs_y_max if bs_y_max > 0 else 100)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    fig.suptitle('超参数实验对比', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'hyper_comparison.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[visualize] 超参数对比图已保存: {save_path}')


# ============================================================
# 入口：一次性生成所有图表
# ============================================================

def plot_all(
    results_dir: str = 'experiments/results',
    out_dir: str = 'report/figures',
) -> None:
    """生成本项目所需的所有可视化图表。"""
    print(f'\n[visualize] 从 {results_dir} 读取实验数据...')
    plot_loss_curve(results_dir, out_dir)
    plot_map_curve(results_dir, out_dir)
    plot_ablation_bar(results_dir, out_dir)
    plot_hyper_comparison(results_dir, out_dir)
    print(f'[visualize] 所有图表已保存到 {out_dir}/')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='experiments/results')
    parser.add_argument('--out-dir', default='report/figures')
    args = parser.parse_args()
    plot_all(args.results_dir, args.out_dir)
