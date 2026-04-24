"""
训练过程可视化脚本

读取各实验的 training_log.csv / scalars.json，生成以下图表：

  report/figures/ablation/
    - training_loss_curve.png  : 4 组消融实验训练 loss 曲线
    - ablation_bar_chart.png   : 消融实验最终 mAP 柱状图
    - ablation_map_curves.png  : 4 组消融实验 mAP 收敛曲线
    - speed_accuracy.png       : 推理速度-精度散点图

  report/figures/hyper/
    - hyper_comparison.png     : 超参数最终 mAP 对比柱状图
    - hyper_map_curves.png     : 超参数 mAP 收敛曲线

  report/figures/training/
    - loss_components.png      : Baseline 的 4 个 loss 分量分解图

使用方式：
    from pytorch_mmdet.utils.visualize import plot_all
    plot_all(results_dir='experiments/results', out_dir='report/figures')
"""

import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')   # 无显示器环境下不弹窗
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# 统一图表风格


def _pick_font_family() -> Tuple[str, bool]:
    preferred_fonts = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'Source Han Sans SC',
        'WenQuanYi Zen Hei',
        'SimHei',
        'Microsoft YaHei',
        'PingFang SC',
        'Arial Unicode MS',
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            return font_name, True
    return 'DejaVu Sans', False


_FONT_FAMILY, _HAS_CJK_FONT = _pick_font_family()

plt.rcParams.update({
    'font.family': _FONT_FAMILY,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
    'axes.unicode_minus': False,
})


def _text(zh: str, en: str) -> str:
    return zh if _HAS_CJK_FONT else en

# 各实验的显示名称和颜色
_EXP_STYLES: Dict[str, Tuple[str, str]] = {
    'ablation_1_baseline':     ('Baseline',            '#2196F3'),
    'ablation_2_channel_only': ('CA Only',             '#FF9800'),
    'ablation_3_spatial_only': ('SA Only',             '#9C27B0'),
    'ablation_4_cbam_r16_k7':  ('Full CBAM (r=16,k=7)', '#F44336'),
    'hyper_lr0005_bs4':        ('lr=0.005, bs=4',     '#00BCD4'),
    'hyper_lr001_bs4':         ('lr=0.01, bs=4 (default)', '#F44336'),
    'hyper_lr002_bs4':         ('lr=0.02,  bs=4',     '#FF5722'),
    'hyper_lr0005_bs2':        ('lr=0.005, bs=2',     '#8BC34A'),
    'hyper_lr002_bs8':         ('lr=0.02,  bs=8',     '#3F51B5'),
}


_NUMERIC_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')


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

    def _to_numeric_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors='coerce')

        extracted = series.astype(str).str.extract(_NUMERIC_PATTERN, expand=False)
        return pd.to_numeric(extracted, errors='coerce')

    iter_df['epoch'] = iter_df['epoch'].astype(int)
    iter_df['iter'] = iter_df['iter'].astype(int)
    for col in [
        'total_loss', 'rpn_cls_loss', 'rpn_bbox_loss', 'rcnn_cls_loss', 'rcnn_bbox_loss'
    ]:
        if col in iter_df.columns:
            iter_df[col] = _to_numeric_series(iter_df[col])

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


def _load_scalars_json(exp_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """读取实验目录下 vis_data/scalars.json，返回训练和验证数据。

    scalars.json 每行是一个 JSON 对象：
      训练行: {'loss', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox',
                'acc', 'lr', 'epoch', 'iter', 'step', ...}
      验证行: {'pascal_voc/mAP', 'pascal_voc/AP50', 'step', ...}  (step = epoch)

    Returns:
        train_entries : 训练 iteration 数据（含 loss 分量）
        val_entries   : 验证 epoch 数据（含 mAP）
    """
    exp_path = Path(exp_dir)
    train_entries: List[Dict] = []
    val_entries: List[Dict] = []

    for sf in sorted(exp_path.glob('*/vis_data/scalars.json')):
        try:
            lines = sf.read_text(encoding='utf-8').splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'pascal_voc/mAP' in d:
                val_entries.append(d)
            elif 'loss' in d:
                train_entries.append(d)

    return train_entries, val_entries


# ============================================================
# 图 1：训练 Loss 曲线
# ============================================================

def plot_loss_curve(
    results_dir: str,
    out_dir: str,
    exp_keys: Optional[List[str]] = None,
) -> None:
    """绘制全部 4 组消融实验的训练 loss 曲线（epoch 均值），保存为 training_loss_curve.png。

    Args:
        results_dir: 实验结果根目录
        out_dir    : 图片保存目录
        exp_keys   : 要展示的实验目录名列表（默认 4 组消融实验）
    """
    if exp_keys is None:
        exp_keys = _ABLATION_KEYS

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
        iter_df = iter_df.dropna(subset=[loss_col])
        if iter_df.empty:
            continue
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
    ax.set_title('Ablation Study: Training Loss Curve (All Components)')
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
    ax.set_title('Validation mAP Curve: Baseline vs CBAM')
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
    ax.set_title(_text('消融实验：各注意力组件对 mAP 的贡献', 'Ablation Study: Attention Module Contribution to mAP'))
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
        ('hyper_lr001_bs4',  'lr=0.01 (default)'),
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
    axes[0].set_title(_text('学习率对比（batch_size=4）', 'Learning Rate Comparison (batch_size=4)'))
    lr_y_max = max(lr_maps) * 1.12 if lr_maps else 100
    axes[0].set_ylim(0, lr_y_max if lr_y_max > 0 else 100)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    # ---- 子图 2：Batch Size 对比----
    bs_exps = [
        ('hyper_lr0005_bs2',  'bs=2 (lr=0.005)'),
        ('hyper_lr001_bs4',   'bs=4 (default)'),
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
    axes[1].set_title(_text('Batch Size 对比（线性缩放 lr）', 'Batch Size Comparison (Linear LR Scaling)'))
    bs_y_max = max(bs_maps) * 1.12 if bs_maps else 100
    axes[1].set_ylim(0, bs_y_max if bs_y_max > 0 else 100)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    fig.suptitle(_text('超参数实验对比', 'Hyperparameter Comparison'), fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'hyper_comparison.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[visualize] 超参数对比图已保存: {save_path}')


# ============================================================
# 图 5：消融实验 mAP 收敛曲线（全部 4 组）
# ============================================================

def plot_ablation_map_curves(results_dir: str, out_dir: str) -> None:
    """绘制全部 4 组消融实验的 per-epoch mAP 收敛曲线，保存为 ablation_map_curves.png。

    数据源：vis_data/scalars.json（比 training_log.csv 更精确的原始记录）。
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False

    for exp_key in _ABLATION_KEYS:
        exp_dir = os.path.join(results_dir, exp_key)
        _, val_entries = _load_scalars_json(exp_dir)
        if not val_entries:
            continue

        val_entries.sort(key=lambda x: x['step'])
        epochs = [e['step'] for e in val_entries]
        maps = [e['pascal_voc/mAP'] * 100 for e in val_entries]

        label, color = _EXP_STYLES.get(exp_key, (exp_key, None))
        ax.plot(epochs, maps, label=label, color=color, linewidth=2,
                marker='o', markersize=5)

        # 标注最终 mAP
        ax.annotate(
            f'{maps[-1]:.1f}%',
            xy=(epochs[-1], maps[-1]),
            xytext=(epochs[-1] + 0.25, maps[-1]),
            fontsize=8.5, color=color, va='center',
        )
        plotted = True

    if not plotted:
        print('[visualize] 未找到 scalars.json，跳过消融 mAP 曲线绘制。')
        plt.close(fig)
        return

    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@0.5 (%)')
    ax.set_title('Ablation Study: mAP Convergence (All Components)')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'ablation_map_curves.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[visualize] 消融 mAP 曲线已保存: {save_path}')


# ============================================================
# 图 6：Baseline Loss 分量堆叠图
# ============================================================

def plot_loss_components(results_dir: str, out_dir: str) -> None:
    """绘制 Baseline 模型各 loss 分量随 epoch 均值的堆叠面积图，
    保存为 loss_components.png。

    展示 RPN/RCNN 的分类与回归损失如何随训练进展下降。
    """
    from collections import defaultdict

    exp_dir = os.path.join(results_dir, 'ablation_1_baseline')
    train_entries, _ = _load_scalars_json(exp_dir)
    if not train_entries:
        print('[visualize] 未找到 scalars.json，跳过 loss 分量图绘制。')
        return

    # 按 epoch 分组，计算每个分量的均值
    epoch_buckets: Dict[int, List[Dict]] = defaultdict(list)
    for entry in train_entries:
        ep = int(entry.get('epoch', entry.get('step', 0)))
        epoch_buckets[ep].append(entry)

    epochs = sorted(epoch_buckets.keys())
    components = ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox']
    comp_labels = ['RPN Cls', 'RPN Bbox', 'RCNN Cls', 'RCNN Bbox']
    comp_colors = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350']

    stacks = []
    for comp in components:
        row = []
        for ep in epochs:
            vals = [float(e[comp]) for e in epoch_buckets[ep]
                    if comp in e and not np.isnan(float(e[comp]))]
            row.append(np.mean(vals) if vals else 0.0)
        stacks.append(row)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(epochs, stacks, labels=comp_labels,
                 colors=comp_colors, alpha=0.75)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Baseline: Training Loss Component Breakdown')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'loss_components.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[visualize] Loss 分量图已保存: {save_path}')


# ============================================================
# 图 7：速度-精度权衡散点图（消融实验）
# ============================================================

def plot_speed_accuracy(results_dir: str, out_dir: str) -> None:
    """绘制消融实验的推理速度-精度散点图（FPS vs mAP@0.5），
    保存为 speed_accuracy.png。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False

    for exp_key in _ABLATION_KEYS:
        data = _load_eval_json(os.path.join(results_dir, exp_key))
        if data is None:
            continue
        label, color = _EXP_STYLES.get(exp_key, (exp_key, '#9E9E9E'))
        mAP = data.get('mAP', 0) * 100
        fps = data.get('fps', 0)
        if fps <= 0:
            continue

        ax.scatter(fps, mAP, color=color, s=200, zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.annotate(
            label,
            xy=(fps, mAP),
            xytext=(fps + 0.8, mAP - 1.5),
            fontsize=9, color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
        )
        plotted = True

    if not plotted:
        print('[visualize] 未找到 eval_results.json，跳过速度-精度图绘制。')
        plt.close(fig)
        return

    ax.set_xlabel('Inference Speed (FPS)')
    ax.set_ylabel('mAP@0.5 (%)')
    ax.set_title('Ablation Study: Speed-Accuracy Trade-off')
    ax.grid(True, linestyle='--', alpha=0.5)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'speed_accuracy.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[visualize] 速度-精度图已保存: {save_path}')


# ============================================================
# 图 8：超参数实验 mAP 收敛曲线（lr 组 + bs 组）
# ============================================================

def plot_hyper_map_curves(results_dir: str, out_dir: str) -> None:
    """绘制超参数实验的 per-epoch mAP 收敛曲线（2 子图），
    保存为 hyper_map_curves.png。

    左：学习率对比（batch_size=4 固定）
    右：Batch Size 对比（线性缩放 lr）
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    lr_group = [
        ('hyper_lr0005_bs4', '-'),
        ('hyper_lr001_bs4',  '-'),
        ('hyper_lr002_bs4',  '--'),   # 虚线标识训练发散
    ]
    bs_group = [
        ('hyper_lr0005_bs2', '-'),
        ('hyper_lr001_bs4',  '-'),
        ('hyper_lr002_bs8',  '-'),
    ]

    for ax, group, title in [
        (axes[0], lr_group, 'Learning Rate Comparison (batch_size=4)'),
        (axes[1], bs_group, 'Batch Size Comparison (Linear LR Scaling)'),
    ]:
        for exp_key, ls in group:
            exp_dir = os.path.join(results_dir, exp_key)
            _, val_entries = _load_scalars_json(exp_dir)
            if not val_entries:
                continue
            val_entries.sort(key=lambda x: x['step'])
            epochs = [e['step'] for e in val_entries]
            maps = [e['pascal_voc/mAP'] * 100 for e in val_entries]

            label, color = _EXP_STYLES.get(exp_key, (exp_key, None))
            ax.plot(epochs, maps, label=label, color=color,
                    linewidth=2, marker='o', markersize=4, linestyle=ls)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP@0.5 (%)')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle('Hyperparameter Experiments: mAP Convergence',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'hyper_map_curves.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[visualize] 超参 mAP 曲线已保存: {save_path}')


# ============================================================
# 入口：一次性生成所有图表
# ============================================================

def plot_all(
    results_dir: str = 'experiments/results',
    out_dir: str = 'report/figures',
) -> None:
    """生成本项目所需的所有可视化图表，按类别保存到三个子目录。"""
    ablation_dir = os.path.join(out_dir, 'ablation')
    hyper_dir    = os.path.join(out_dir, 'hyper')
    training_dir = os.path.join(out_dir, 'training')

    print(f'\n[visualize] 从 {results_dir} 读取实验数据...')

    # ablation/ 目录：4 张消融相关图
    plot_loss_curve(results_dir, ablation_dir)         # training_loss_curve.png
    plot_ablation_bar(results_dir, ablation_dir)       # ablation_bar_chart.png
    plot_ablation_map_curves(results_dir, ablation_dir) # ablation_map_curves.png
    plot_speed_accuracy(results_dir, ablation_dir)     # speed_accuracy.png

    # hyper/ 目录：2 张超参图
    plot_hyper_comparison(results_dir, hyper_dir)      # hyper_comparison.png
    plot_hyper_map_curves(results_dir, hyper_dir)      # hyper_map_curves.png

    # training/ 目录：1 张 loss 分量图
    plot_loss_components(results_dir, training_dir)    # loss_components.png

    print(f'\n[visualize] 所有图表已保存到:')
    print(f'  {ablation_dir}/  (4 张)')
    print(f'  {hyper_dir}/     (2 张)')
    print(f'  {training_dir}/  (1 张)')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='experiments/results')
    parser.add_argument('--out-dir', default='report/figures')
    args = parser.parse_args()
    plot_all(args.results_dir, args.out_dir)
