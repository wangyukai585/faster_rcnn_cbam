# ============================================================
# 消融实验 3/6：仅空间注意力（Spatial Attention Only）
#
# 目的：单独验证空间注意力对检测性能的贡献。
#       与 Baseline 对比可量化 SA 的增益；
#       与完整 CBAM 对比可量化 CA 的额外增益。
#
# 改动：在 ResNetCBAM 中关闭 ChannelAttention（cbam_use_channel=False），
#       只保留 SpatialAttention，其余配置与 Baseline 完全一致。
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
        cbam_reduction=16,
        cbam_kernel_size=7,
        cbam_use_channel=False,   # 关闭通道注意力（消融）
        cbam_use_spatial=True,    # 启用空间注意力
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    )
)
