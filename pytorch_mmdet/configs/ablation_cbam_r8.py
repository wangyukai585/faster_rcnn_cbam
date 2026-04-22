# ============================================================
# 消融实验 5/6：完整 CBAM，reduction=8（更大压缩前通道数）
#
# 目的：验证通道注意力压缩比 reduction 对性能的影响。
#       reduction=8 时 MLP 隐层通道数增大（C//8 > C//16），
#       模型容量更大，但参数量增加。
#       与默认 reduction=16 对比，确定最优压缩比。
#
# 改动：cbam_reduction 从 16 改为 8，其余与 cbam_faster_rcnn.py 一致。
# ============================================================

_base_ = ['base_faster_rcnn.py']

model = dict(
    backbone=dict(
        type='ResNetCBAM',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        cbam_reduction=8,         # 压缩比改为 8（默认 16）
        cbam_kernel_size=7,
        cbam_use_channel=True,
        cbam_use_spatial=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    )
)
