# ============================================================
# 超参数实验：batch_size=2，lr=0.005（线性缩放规则）
#
# 目的：验证更小批量对训练效果的影响，适用于显存受限的场景。
#       根据线性缩放规则（Linear Scaling Rule, Goyal et al. 2017）：
#         lr ∝ batch_size，故 bs 从 4→2，lr 从 0.01→0.005
#       小 batch 的梯度噪声更大，可能有正则化效果，但收敛更慢。
#
# 与默认配置（cbam_faster_rcnn.py）的区别：
#   train_dataloader.batch_size: 4 -> 2
#   optimizer.lr: 0.01 -> 0.005（线性缩放）
# ============================================================

_base_ = ['cbam_faster_rcnn.py']

train_dataloader = dict(
    batch_size=2,     # 减半，适配显存受限环境
    num_workers=2,
)

optim_wrapper = dict(
    type='OptimWrapper',
    # 线性缩放：lr = 0.01 * (2/4) = 0.005
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=1e-4),
)
