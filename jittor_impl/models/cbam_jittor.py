"""
Jittor 版 CBAM 注意力模块

与 PyTorch 版（pytorch_mmdet/models/cbam.py）功能完全一致，
使用 Jittor 语法重写。主要差异：
  1. 继承 jittor.nn.Module，前向方法命名为 execute（Jittor 原生）
  2. torch.mean → jt.mean（keepdim → keepdims）
  3. torch.max  → jt.max（返回 Var 而非 (values,indices) 元组）
  4. torch.cat  → jt.concat
  5. 无需 .cuda()，Jittor 自动管理设备
"""

import jittor as jt
import jittor.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力模块（Jittor 版）

    原理：双路全局池化 → 共享 MLP → Sigmoid → 通道权重
    与 PyTorch 版结构完全相同。

    Args:
        in_channels (int): 输入通道数 C
        reduction   (int): MLP 压缩比，默认 16
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden = max(in_channels // reduction, 1)
        # 使用 1×1 卷积替代全连接（与 PyTorch 版相同）
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )

    def execute(self, x: jt.Var) -> jt.Var:
        """Jittor 前向传播（等价于 PyTorch 的 forward）

        Args:
            x: [B, C, H, W]
        Returns:
            通道加权后的特征图 [B, C, H, W]
        """
        # 全局平均池化: [B, C, H, W] -> [B, C, 1, 1]
        avg_out = self.avg_pool(x)

        # 全局最大池化: [B, C, H, W] -> [B, C, 1, 1]
        # Jittor 差异：nn.AdaptiveMaxPool2d 与 PyTorch 用法相同
        max_out = self.max_pool(x)

        # 共享 MLP: [B, C, 1, 1] -> [B, C, 1, 1]
        avg_out = self.shared_mlp(avg_out)
        max_out = self.shared_mlp(max_out)

        # Sigmoid 得到通道权重: [B, C, 1, 1]
        # Jittor 差异：jt.sigmoid 与 torch.sigmoid 用法相同
        channel_weight = jt.sigmoid(avg_out + max_out)

        # 广播乘法: [B, C, H, W]
        return x * channel_weight


class SpatialAttention(nn.Module):
    """空间注意力模块（Jittor 版）

    原理：通道维度 avg+max 拼接 → 7×7 卷积 → Sigmoid → 空间权重

    Args:
        kernel_size (int): 空间卷积核大小，默认 7
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def execute(self, x: jt.Var) -> jt.Var:
        """Jittor 前向传播

        Args:
            x: [B, C, H, W]
        Returns:
            空间加权后的特征图 [B, C, H, W]
        """
        # 通道均值: [B, C, H, W] -> [B, 1, H, W]
        # Jittor 差异：keepdims（PyTorch 用 keepdim）
        avg_out = jt.mean(x, dim=1, keepdims=True)

        # 通道最大值: [B, C, H, W] -> [B, 1, H, W]
        # Jittor 差异：jt.max 沿 dim 时直接返回值（不返回 indices 元组）
        max_out = jt.max(x, dim=1, keepdims=True)

        # 拼接: [B, 1, H, W] + [B, 1, H, W] -> [B, 2, H, W]
        # Jittor 差异：jt.concat（PyTorch 用 torch.cat）
        concat = jt.concat([avg_out, max_out], dim=1)

        # 7×7 卷积 + Sigmoid -> [B, 1, H, W]
        spatial_weight = jt.sigmoid(self.conv(concat))

        # 广播乘法: [B, C, H, W]
        return x * spatial_weight


class CBAM(nn.Module):
    """CBAM 完整注意力模块（Jittor 版）

    通道注意力 → 空间注意力，串联结构，支持消融实验开关。

    Args:
        in_channels       (int):  输入通道数
        reduction         (int):  通道注意力压缩比，默认 16
        kernel_size       (int):  空间注意力卷积核，默认 7
        use_channel_attn  (bool): 是否启用通道注意力，默认 True
        use_spatial_attn  (bool): 是否启用空间注意力，默认 True
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
    ):
        super().__init__()
        self.channel_attention = (
            ChannelAttention(in_channels, reduction) if use_channel_attn else None
        )
        self.spatial_attention = (
            SpatialAttention(kernel_size) if use_spatial_attn else None
        )

    def execute(self, x: jt.Var) -> jt.Var:
        """前向传播: [B, C, H, W] -> [B, C, H, W]"""
        if self.channel_attention is not None:
            x = self.channel_attention(x)
        if self.spatial_attention is not None:
            x = self.spatial_attention(x)
        return x


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    jt.flags.use_cuda = 1 if jt.has_cuda else 0

    B, C, H, W = 2, 256, 32, 32
    x = jt.randn(B, C, H, W)

    print('=' * 50)
    print(f'输入形状: {list(x.shape)}')
    print('=' * 50)

    ca = ChannelAttention(C, reduction=16)
    print(f'ChannelAttention 输出: {list(ca(x).shape)}  ✓')

    sa = SpatialAttention(kernel_size=7)
    print(f'SpatialAttention 输出: {list(sa(x).shape)}  ✓')

    cbam = CBAM(C, reduction=16, kernel_size=7)
    print(f'CBAM 完整输出:         {list(cbam(x).shape)}  ✓')

    print('Jittor CBAM 单元测试通过！')
