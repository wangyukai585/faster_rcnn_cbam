"""
ResNetCBAM：在 ResNet50 每个 Bottleneck 残差块后插入 CBAM 注意力模块

插入位置示意（每个 Bottleneck 内部）：
    原始：conv1→BN→ReLU → conv2→BN→ReLU → conv3→BN → (+shortcut) → ReLU
    改进：conv1→BN→ReLU → conv2→BN→ReLU → conv3→BN → (+shortcut) → CBAM → ReLU

支持的消融参数：
    cbam_use_channel: False 时仅保留空间注意力（消融通道注意力）
    cbam_use_spatial: False 时仅保留通道注意力（消融空间注意力）

向 MMDetection 注册为 'ResNetCBAM'，可在 config 中直接通过字符串名引用：
    backbone=dict(type='ResNetCBAM', depth=50, ...)
"""

import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet, Bottleneck

from .cbam import CBAM


class BottleneckWithCBAM(Bottleneck):
    """在标准 Bottleneck 残差块的残差相加之后、ReLU 之前插入 CBAM 模块。

    继承自 MMDetection 的 Bottleneck，只重写 forward 方法以在正确位置
    插入 CBAM，其余（BN、卷积权重、shortcut 等）完全复用父类实现。

    注：MMDetection 的 ResLayer 会将 make_res_layer 的额外 **kwargs 透传给
    每个 block 的构造函数，因此这里的 CBAM 参数通过 ResNetCBAM.make_res_layer
    自动注入，无需手动为每个 block 传参。

    Args:
        *args: 透传给父类 Bottleneck 的位置参数
        reduction (int): CBAM 通道注意力压缩比，默认 16
        cbam_kernel_size (int): CBAM 空间注意力卷积核大小，默认 7
        use_channel_attn (bool): 是否启用通道注意力（消融实验开关），默认 True
        use_spatial_attn (bool): 是否启用空间注意力（消融实验开关），默认 True
        **kwargs: 透传给父类 Bottleneck 的关键字参数
    """

    def __init__(
        self,
        *args,
        reduction: int = 16,
        cbam_kernel_size: int = 7,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # self.planes * self.expansion 即该 Bottleneck 输出的通道数
        # ResNet50 各 stage 输出：layer1=256, layer2=512, layer3=1024, layer4=2048
        out_channels = self.planes * self.expansion
        self.cbam = CBAM(
            in_channels=out_channels,
            reduction=reduction,
            kernel_size=cbam_kernel_size,
            use_channel_attn=use_channel_attn,
            use_spatial_attn=use_spatial_attn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        复刻父类 forward 逻辑，在残差相加后、最终 ReLU 前插入 CBAM。

        Args:
            x: 输入特征图，形状 [B, C_in, H, W]

        Returns:
            输出特征图，形状 [B, C_out, H', W']
        """
        # shortcut 分支：若通道/步长不匹配则经下采样层对齐维度
        identity = self.downsample(x) if self.downsample is not None else x

        # --- 主路（3层卷积） ---
        # conv1: 1×1 降维  [B, C_in, H, W] -> [B, planes, H, W]
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        # conv2: 3×3 特征提取  [B, planes, H, W] -> [B, planes, H', W']
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        # conv3: 1×1 升维  [B, planes, H', W'] -> [B, planes*4, H', W']
        out = self.conv3(out)
        out = self.norm3(out)

        # 残差相加: [B, planes*4, H', W']
        out = out + identity

        # CBAM 注意力（通道 + 空间双重加权）: [B, planes*4, H', W']
        out = self.cbam(out)

        # 最终激活
        out = self.relu(out)

        return out


@MODELS.register_module()
class ResNetCBAM(ResNet):
    """在 ResNet50 每个 Bottleneck 后插入 CBAM 的改进 Backbone。

    继承自 MMDetection 的 ResNet，通过以下机制实现 CBAM 注入：
      1. 在 arch_settings 中将 block 类型替换为 BottleneckWithCBAM
      2. 重写 make_res_layer，将 CBAM 超参透传至每个 block 构造函数

    MMDetection config 中使用方式示例：
        backbone=dict(
            type='ResNetCBAM',
            depth=50,
            cbam_reduction=16,
            cbam_kernel_size=7,
            cbam_use_channel=True,
            cbam_use_spatial=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        )

    Args:
        depth (int): ResNet 深度，本实验固定 50
        cbam_reduction (int): CBAM 通道注意力压缩比，默认 16
        cbam_kernel_size (int): CBAM 空间注意力卷积核大小，默认 7
        cbam_use_channel (bool): 是否启用通道注意力，默认 True
        cbam_use_spatial (bool): 是否启用空间注意力，默认 True
        **kwargs: 其余参数透传给父类 ResNet（num_stages, out_indices,
                  frozen_stages, norm_cfg, style, init_cfg 等）
    """

    # 用 BottleneckWithCBAM 替换原始 Bottleneck，其他参数与原 arch_settings 相同
    arch_settings = {
        **ResNet.arch_settings,
        50:  (BottleneckWithCBAM, (3, 4, 6, 3)),
        101: (BottleneckWithCBAM, (3, 4, 23, 3)),
        152: (BottleneckWithCBAM, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth: int = 50,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
        cbam_use_channel: bool = True,
        cbam_use_spatial: bool = True,
        **kwargs,
    ):
        # 必须在 super().__init__() 之前保存 CBAM 参数，
        # 因为 super().__init__() 内部会调用 self.make_res_layer()
        self.cbam_reduction = cbam_reduction
        self.cbam_kernel_size = cbam_kernel_size
        self.cbam_use_channel = cbam_use_channel
        self.cbam_use_spatial = cbam_use_spatial

        super().__init__(depth=depth, **kwargs)

    def make_res_layer(self, **kwargs) -> nn.Module:
        """重写 make_res_layer，将 CBAM 超参透传至 BottleneckWithCBAM。

        MMDetection 的 ResLayer 构造函数接受 **kwargs 并将其转发给
        每个 block 的 __init__，因此此处追加的 CBAM 参数会自动注入
        到每个 BottleneckWithCBAM 实例中。
        """
        return super().make_res_layer(
            reduction=self.cbam_reduction,
            cbam_kernel_size=self.cbam_kernel_size,
            use_channel_attn=self.cbam_use_channel,
            use_spatial_attn=self.cbam_use_spatial,
            **kwargs,
        )
