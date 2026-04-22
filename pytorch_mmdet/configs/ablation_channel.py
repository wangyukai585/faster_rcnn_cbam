# ============================================================
# 消融实验 2/6：仅通道注意力（Channel Attention Only）
#
# 目的：单独验证通道注意力对检测性能的贡献。
#       与 Baseline 对比可量化 CA 的增益；
#       与完整 CBAM 对比可量化 SA 的额外增益。
#
# 改动：在 ResNetCBAM 中关闭 SpatialAttention（cbam_use_spatial=False），
#       只保留 ChannelAttention，其余配置与 Baseline 完全一致。
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
        cbam_use_channel=True,    # 启用通道注意力
        cbam_use_spatial=False,   # 关闭空间注意力（消融）
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    )
)
