# ============================================================
# 超参数实验：学习率 lr=0.005（默认 0.01 的一半）
#
# 目的：验证更保守的学习率对收敛速度和最终 mAP 的影响。
#       更小的 lr 通常收敛更慢但训练更稳定，适合小批量或复杂模型。
#
# 与默认配置（cbam_faster_rcnn.py）的区别：
#   optimizer.lr: 0.01 -> 0.005
# ============================================================

_base_ = ['cbam_faster_rcnn.py']

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=1e-4),
)
