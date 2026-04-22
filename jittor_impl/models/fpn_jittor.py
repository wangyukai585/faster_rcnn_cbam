"""
Jittor 版 FPN（Feature Pyramid Network）

接收 ResNet50 的 [C2, C3, C4, C5]，通过自顶向下融合和侧连接，
输出统一 256 通道的多尺度特征 [P2, P3, P4, P5, P6]。

结构（自顶向下）：
  C5 → lateral_conv5(1×1) → P5
  C4 → lateral_conv4(1×1) → +upsample(P5) → P4
  C3 → lateral_conv3(1×1) → +upsample(P4) → P3
  C2 → lateral_conv2(1×1) → +upsample(P3) → P2
  P5 → MaxPool(2×2, s2) → P6（用于检测大目标）

每个 Pi 再经过 output_conv(3×3) 平滑。
"""

from typing import List

import jittor as jt
import jittor.nn as nn


class FPN(nn.Module):
    """特征金字塔网络（Jittor 版）

    Args:
        in_channels  (List[int]): 各 stage 输入通道 [256, 512, 1024, 2048]
        out_channels (int):       所有输出特征的统一通道数，默认 256
        num_outs     (int):       输出特征层数，默认 5（P2~P6）
    """

    def __init__(
        self,
        in_channels: List[int] = (256, 512, 1024, 2048),
        out_channels: int = 256,
        num_outs: int = 5,
    ):
        super().__init__()
        assert len(in_channels) == 4, "FPN 需要 4 个输入 stage"

        self.out_channels = out_channels
        self.num_outs = num_outs

        # 侧连接：1×1 卷积统一通道数，对应 C2~C5
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])

        # 输出平滑卷积：3×3 卷积消除上采样伪影，对应 P2~P5
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

        # P6 由 P5 经最大池化得到（用于检测超大目标）
        if num_outs == 5:
            self.p6_pool = nn.MaxPool2d(kernel_size=1, stride=2)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier 均匀初始化所有卷积权重。"""
        for conv in list(self.lateral_convs) + list(self.output_convs):
            jt.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                conv.bias = jt.zeros(conv.bias.shape)

    def execute(self, features: List[jt.Var]) -> List[jt.Var]:
        """自顶向下融合 + 侧连接，生成多尺度特征图。

        Args:
            features: [C2, C3, C4, C5]，各 [B, C_i, H_i, W_i]

        Returns:
            [P2, P3, P4, P5] 或 [P2, P3, P4, P5, P6]，均为 [B, 256, ...]
        """
        assert len(features) == 4

        # 步骤 1：侧连接，统一通道数
        # laterals[i] 对应 C_{i+2} 经过 1×1 卷积的结果
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # 步骤 2：自顶向下融合（从 C5 到 C2）
        # laterals[-1] 是最顶层（步长最大，分辨率最低），直接作为 P5 基础
        for i in range(len(laterals) - 2, -1, -1):
            # 上采样高一层的特征到当前层尺寸，再相加
            top = laterals[i + 1]
            # Jittor 差异：nn.interpolate 与 PyTorch F.interpolate 用法一致
            upsampled = nn.interpolate(
                top,
                size=(laterals[i].shape[2], laterals[i].shape[3]),
                mode='nearest',
            )
            laterals[i] = laterals[i] + upsampled

        # 步骤 3：输出平滑卷积，得到 P2~P5
        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        # outs[0]=P2, outs[1]=P3, outs[2]=P4, outs[3]=P5

        # 步骤 4：P6 由 P5 下采样得到（用于 anchor stride=64）
        if self.num_outs == 5:
            outs.append(self.p6_pool(outs[-1]))  # P6: [B, 256, H/64, W/64]

        return outs  # [P2, P3, P4, P5, P6]
