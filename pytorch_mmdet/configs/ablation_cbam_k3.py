# ============================================================
# 消融实验 6/6：完整 CBAM，kernel_size=3（更小空间卷积核）
#
# 目的：验证空间注意力卷积核大小对性能的影响。
#       kernel_size=3 感受野更小，参数量更少；
#       kernel_size=7 感受野更大，适合捕获大范围空间依赖。
#       论文推荐 7，此实验验证该结论是否在 VOC 上成立。
#
# 改动：cbam_kernel_size 从 7 改为 3，其余与 cbam_faster_rcnn.py 一致。
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
        cbam_kernel_size=3,       # 空间卷积核改为 3（默认 7）
        cbam_use_channel=True,
        cbam_use_spatial=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    )
)
