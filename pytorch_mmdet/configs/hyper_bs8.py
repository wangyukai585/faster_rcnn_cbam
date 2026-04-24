# ============================================================
# 超参数实验：batch_size=8，lr=0.02（线性缩放规则）
#
# 目的：验证更大批量对训练速度和最终精度的影响，适用于多卡训练场景。
#       根据线性缩放规则（Linear Scaling Rule, Goyal et al. 2017）：
#         lr ∝ batch_size，故 bs 从 4→8，lr 从 0.01→0.02
#       大 batch 梯度估计更准确，通常收敛更快，但可能损失泛化性。
#
# 与默认配置（base_faster_rcnn.py）的区别：
#   train_dataloader.batch_size: 4 -> 8
#   optimizer.lr: 0.01 -> 0.02（线性缩放）
#   注意：运行此配置需要足够的 GPU 显存（建议 ≥ 2×24GB 或 4×16GB）
# ============================================================

_base_ = ['base_faster_rcnn.py']

train_dataloader = dict(
    batch_size=8,     # 翻倍，需要更多显存
    num_workers=4,
)

optim_wrapper = dict(
    type='OptimWrapper',
    # 线性缩放：lr = 0.01 * (8/4) = 0.02
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=1e-4),
)
