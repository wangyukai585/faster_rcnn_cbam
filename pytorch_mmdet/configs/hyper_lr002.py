# ============================================================
# 超参数实验：学习率 lr=0.02（默认 0.01 的两倍）
#
# 目的：验证更激进的学习率对收敛速度和最终 mAP 的影响。
#       更大的 lr 收敛更快，但可能导致训练不稳定或过拟合。
#       根据线性缩放规则（Linear Scaling Rule），lr 与 batch_size 等比。
#
# 与默认配置（base_faster_rcnn.py）的区别：
#   optimizer.lr: 0.01 -> 0.02
# ============================================================

_base_ = ['base_faster_rcnn.py']

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=1e-4),
)
