# ============================================================
# 完整创新配置：Faster R-CNN + CBAM（完整版，reduction=16，kernel_size=7）
#
# 继承 base_faster_rcnn.py，仅将 backbone 替换为 ResNetCBAM，
# 其余所有设置（数据集、训练策略、超参数）与 Baseline 完全一致，
# 保证实验对比的公平性。
#
# 相比 Baseline 的改动：
#   backbone.type: ResNet -> ResNetCBAM
#   backbone 新增：cbam_reduction=16, cbam_kernel_size=7（论文默认值）
# ============================================================

_base_ = ['base_faster_rcnn.py']

model = dict(
    backbone=dict(
        # 将 type 从 'ResNet' 改为我们注册的 'ResNetCBAM'
        type='ResNetCBAM',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # CBAM 核心超参数（论文推荐默认值）
        cbam_reduction=16,       # 通道注意力 MLP 压缩比
        cbam_kernel_size=7,      # 空间注意力卷积核大小
        cbam_use_channel=True,   # 启用通道注意力
        cbam_use_spatial=True,   # 启用空间注意力
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50',
        ),
    )
)
