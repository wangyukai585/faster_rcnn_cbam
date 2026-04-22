"""
CBAM (Convolutional Block Attention Module) 注意力模块实现

论文参考：
  Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
  https://arxiv.org/abs/1807.06521

模块结构：输入特征图 → 通道注意力（可选）→ 空间注意力（可选）→ 加权输出特征图

注：use_channel_attn / use_spatial_attn 两个开关用于消融实验，
    可独立关闭某一子模块，验证各组件对性能的贡献。
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力模块 (Channel Attention Module)

    原理：通过对空间维度做全局池化，提取每个通道的全局信息，
    再经共享 MLP 学习各通道的重要性权重，让网络自动关注更有价值的特征通道。

    Args:
        in_channels (int): 输入特征图的通道数 C
        reduction (int): MLP 瓶颈压缩比，隐层通道数 = C // reduction，默认 16
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        # 全局平均池化：将空间维度压缩为 1×1，捕获通道的全局分布信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全局最大池化：提取每个通道的最显著激活，关注突出特征
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享 MLP：两路池化结果共用同一套权重
        # 使用 1×1 卷积等价于全连接层，避免 Flatten 操作，保持 [B, C, 1, 1] 格式
        hidden_channels = max(in_channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            # 降维：C → C//reduction，压缩通道间冗余信息
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            # 升维：C//reduction → C，恢复原始通道数
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，形状 [B, C, H, W]

        Returns:
            通道加权后的特征图，形状 [B, C, H, W]
        """
        # 全局平均池化: [B, C, H, W] -> [B, C, 1, 1]
        avg_out = self.avg_pool(x)

        # 全局最大池化: [B, C, H, W] -> [B, C, 1, 1]
        max_out = self.max_pool(x)

        # 两路分别经过共享 MLP: [B, C, 1, 1] -> [B, C, 1, 1]
        avg_out = self.shared_mlp(avg_out)
        max_out = self.shared_mlp(max_out)

        # 逐元素相加后 Sigmoid，得到通道权重: [B, C, 1, 1]
        channel_weight = self.sigmoid(avg_out + max_out)

        # 通道权重与输入特征图相乘（广播机制）: [B, C, H, W]
        return x * channel_weight


class SpatialAttention(nn.Module):
    """空间注意力模块 (Spatial Attention Module)

    原理：对通道维度做统计聚合（均值 + 最大值），得到包含空间位置信息的描述，
    再经 7×7 卷积学习各空间位置的重要性权重，让网络聚焦于更关键的区域。

    Args:
        kernel_size (int): 空间卷积核大小，论文推荐 7，也可选 3，默认 7
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        assert kernel_size in (3, 7), "kernel_size 仅支持 3 或 7"
        padding = kernel_size // 2  # 保持 H、W 不变所需的 padding

        # 7×7 卷积：输入 2 通道（avg + max 拼接），输出 1 通道空间权重图
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 经过通道注意力后的特征图，形状 [B, C, H, W]

        Returns:
            空间加权后的特征图，形状 [B, C, H, W]
        """
        # 沿通道维度取均值，捕获全局空间分布: [B, C, H, W] -> [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)

        # 沿通道维度取最大值，捕获显著激活: [B, C, H, W] -> [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 在通道维度拼接两路结果: [B, 1, H, W] cat [B, 1, H, W] -> [B, 2, H, W]
        concat = torch.cat([avg_out, max_out], dim=1)

        # 7×7 卷积 + Sigmoid，生成空间权重图: [B, 2, H, W] -> [B, 1, H, W]
        spatial_weight = self.sigmoid(self.conv(concat))

        # 空间权重与输入特征图相乘（广播机制）: [B, C, H, W]
        return x * spatial_weight


class CBAM(nn.Module):
    """CBAM 完整注意力模块 (Convolutional Block Attention Module)

    将 ChannelAttention 和 SpatialAttention 串联：
        输入 → 通道注意力（可选）→ 空间注意力（可选）→ 输出

    两个注意力模块顺序施加，分别从"哪些通道重要"和"哪些位置重要"两个维度
    对特征图进行自适应加权，在几乎不增加计算量的前提下提升表征能力。

    Args:
        in_channels (int): 输入特征图的通道数 C
        reduction (int): 通道注意力 MLP 压缩比，默认 16
        kernel_size (int): 空间注意力卷积核大小，默认 7
        use_channel_attn (bool): 是否启用通道注意力，消融实验时可关闭，默认 True
        use_spatial_attn (bool): 是否启用空间注意力，消融实验时可关闭，默认 True
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

        # 根据消融开关决定是否构建子模块（None 表示跳过该模块）
        self.channel_attention = (
            ChannelAttention(in_channels, reduction) if use_channel_attn else None
        )
        self.spatial_attention = (
            SpatialAttention(kernel_size) if use_spatial_attn else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，形状 [B, C, H, W]

        Returns:
            经注意力加权后的特征图，形状 [B, C, H, W]（与输入同形）
        """
        # 通道注意力（若启用）: [B, C, H, W] -> [B, C, H, W]
        if self.channel_attention is not None:
            x = self.channel_attention(x)

        # 空间注意力（若启用）: [B, C, H, W] -> [B, C, H, W]
        if self.spatial_attention is not None:
            x = self.spatial_attention(x)

        return x


# ---------------------------------------------------------------------------
# 单元测试：验证各子模块的输入输出维度是否正确
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 256, 32, 32
    x = torch.randn(B, C, H, W)

    print("=" * 55)
    print(f"输入特征图形状: {list(x.shape)}")
    print("=" * 55)

    # 测试通道注意力
    ca = ChannelAttention(in_channels=C, reduction=16)
    ca_out = ca(x)
    print(f"ChannelAttention 输出: {list(ca_out.shape)}  ✓")
    assert ca_out.shape == x.shape

    # 测试空间注意力
    sa = SpatialAttention(kernel_size=7)
    sa_out = sa(x)
    print(f"SpatialAttention 输出: {list(sa_out.shape)}  ✓")
    assert sa_out.shape == x.shape

    # 测试完整 CBAM
    cbam_full = CBAM(in_channels=C, reduction=16, kernel_size=7)
    cbam_out = cbam_full(x)
    print(f"CBAM(完整) 输出:       {list(cbam_out.shape)}  ✓")
    assert cbam_out.shape == x.shape

    # 测试消融：仅通道注意力
    cbam_ca_only = CBAM(C, use_channel_attn=True, use_spatial_attn=False)
    print(f"CBAM(仅CA) 输出:       {list(cbam_ca_only(x).shape)}  ✓")

    # 测试消融：仅空间注意力
    cbam_sa_only = CBAM(C, use_channel_attn=False, use_spatial_attn=True)
    print(f"CBAM(仅SA) 输出:       {list(cbam_sa_only(x).shape)}  ✓")

    print("=" * 55)
    print("所有形状验证通过！")
