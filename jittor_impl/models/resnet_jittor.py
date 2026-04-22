"""
Jittor 版 ResNet50 + CBAM Backbone

从零实现 ResNet50，在每个 Bottleneck 残差相加后插入 CBAM，
输出 layer1~layer4 的特征图供 FPN 使用。

网络结构：
  conv1(7×7, s2) → BN → ReLU → MaxPool(3×3, s2)
  → layer1(3×Bottleneck, 256ch)
  → layer2(4×Bottleneck, 512ch)
  → layer3(6×Bottleneck, 1024ch)
  → layer4(3×Bottleneck, 2048ch)
  → 输出 [C2, C3, C4, C5]（对应 layer1~layer4）

支持从 torchvision ResNet50 权重文件（.pth）转换加载预训练权重。
"""

import os
import math
from typing import List, Optional

import jittor as jt
import jittor.nn as nn

from .cbam_jittor import CBAM


class Bottleneck(nn.Module):
    """ResNet50 Bottleneck 残差块，插入 CBAM 于残差相加之后、ReLU 之前。

    结构：
      conv1(1×1) → BN → ReLU
      → conv2(3×3) → BN → ReLU
      → conv3(1×1) → BN
      → (+shortcut) → CBAM → ReLU

    Args:
        inplanes        (int):  输入通道数
        planes          (int):  中间通道数（输出通道 = planes * 4）
        stride          (int):  conv2 的步长，默认 1
        downsample      (nn.Module): shortcut 下采样层（可为 None）
        cbam_reduction  (int):  CBAM 压缩比
        cbam_kernel_size(int):  CBAM 空间卷积核大小
        use_channel_attn(bool): 是否启用 CA
        use_spatial_attn(bool): 是否启用 SA
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
    ):
        super().__init__()

        out_channels = planes * self.expansion

        # 1×1 降维卷积
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3×3 特征提取卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 1×1 升维卷积
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsample = downsample

        # CBAM 注意力模块（残差相加后插入）
        self.cbam = CBAM(
            in_channels=out_channels,
            reduction=cbam_reduction,
            kernel_size=cbam_kernel_size,
            use_channel_attn=use_channel_attn,
            use_spatial_attn=use_spatial_attn,
        )

    def execute(self, x: jt.Var) -> jt.Var:
        """前向传播

        Args:
            x: [B, C_in, H, W]
        Returns:
            [B, planes*4, H', W']
        """
        # shortcut 分支
        identity = self.downsample(x) if self.downsample is not None else x

        # --- 主路 ---
        # conv1: [B, C_in, H, W] -> [B, planes, H, W]
        out = self.relu(self.bn1(self.conv1(x)))

        # conv2: [B, planes, H, W] -> [B, planes, H', W']
        out = self.relu(self.bn2(self.conv2(out)))

        # conv3: [B, planes, H', W'] -> [B, planes*4, H', W']
        out = self.bn3(self.conv3(out))

        # 残差相加: [B, planes*4, H', W']
        out = out + identity

        # CBAM 双重注意力: [B, planes*4, H', W']
        out = self.cbam(out)

        # 最终激活
        out = self.relu(out)
        return out


def _make_layer(
    inplanes: int,
    planes: int,
    num_blocks: int,
    stride: int = 1,
    cbam_reduction: int = 16,
    cbam_kernel_size: int = 7,
    use_channel_attn: bool = True,
    use_spatial_attn: bool = True,
) -> nn.Sequential:
    """构建一个 stage（多个 Bottleneck 的顺序组合）。

    第一个 block 负责调整 stride 和通道数（可能含 downsample），
    后续 block stride=1，inplanes = planes*4。
    """
    downsample = None
    out_channels = planes * Bottleneck.expansion

    # 当步长或通道数不匹配时需要下采样分支
    if stride != 1 or inplanes != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    cbam_kwargs = dict(
        cbam_reduction=cbam_reduction,
        cbam_kernel_size=cbam_kernel_size,
        use_channel_attn=use_channel_attn,
        use_spatial_attn=use_spatial_attn,
    )

    layers = [
        Bottleneck(inplanes, planes, stride=stride, downsample=downsample, **cbam_kwargs)
    ]
    inplanes = out_channels
    for _ in range(1, num_blocks):
        layers.append(Bottleneck(inplanes, planes, stride=1, **cbam_kwargs))

    return nn.Sequential(*layers)


class ResNet50CBAM(nn.Module):
    """ResNet50 + CBAM Backbone（Jittor 版）

    输出 layer1~layer4 的特征图，通道数为 [256, 512, 1024, 2048]，
    步长分别为 [4, 8, 16, 32]（相对于输入图像）。

    Args:
        cbam_reduction   (int):  CBAM 通道注意力压缩比，默认 16
        cbam_kernel_size (int):  CBAM 空间注意力卷积核，默认 7
        use_channel_attn (bool): 是否启用通道注意力，默认 True
        use_spatial_attn (bool): 是否启用空间注意力，默认 True
        frozen_stages    (int):  冻结前几个 stage 的参数（0 = 不冻结），默认 1
    """

    def __init__(
        self,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
        frozen_stages: int = 1,
    ):
        super().__init__()

        cbam_kwargs = dict(
            cbam_reduction=cbam_reduction,
            cbam_kernel_size=cbam_kernel_size,
            use_channel_attn=use_channel_attn,
            use_spatial_attn=use_spatial_attn,
        )

        # ---- Stem 层 ----
        # conv1: [B, 3, H, W] -> [B, 64, H/2, W/2]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # maxpool: [B, 64, H/2, W/2] -> [B, 64, H/4, W/4]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- 4 个 stage ----
        # layer1: inplanes=64  -> 256,  stride=1, 3 blocks
        self.layer1 = _make_layer(64,   64,  3, stride=1, **cbam_kwargs)
        # layer2: inplanes=256 -> 512,  stride=2, 4 blocks
        self.layer2 = _make_layer(256,  128, 4, stride=2, **cbam_kwargs)
        # layer3: inplanes=512 -> 1024, stride=2, 6 blocks
        self.layer3 = _make_layer(512,  256, 6, stride=2, **cbam_kwargs)
        # layer4: inplanes=1024 -> 2048, stride=2, 3 blocks
        self.layer4 = _make_layer(1024, 512, 3, stride=2, **cbam_kwargs)

        # 初始化权重
        self._init_weights()

        # 冻结前 frozen_stages 个 stage（stem + layer1 = stage 1）
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _init_weights(self) -> None:
        """Kaiming 初始化卷积权重，BN 初始化为 1/0。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 正态初始化
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight = jt.init.gauss(m.weight.shape, mean=0.0, std=math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight = jt.ones(m.weight.shape)
                m.bias = jt.zeros(m.bias.shape)

    def _freeze_stages(self) -> None:
        """冻结指定 stage 的参数（stopped_grad + eval BN）。"""
        if self.frozen_stages >= 1:
            # 冻结 stem
            self.bn1.eval()
            for param in [self.conv1.weight, self.bn1.weight, self.bn1.bias]:
                param.stop_grad()
        if self.frozen_stages >= 2:
            self.layer1.eval()
            for param in self.layer1.parameters():
                param.stop_grad()

    def load_pretrained(self, pth_path: str) -> None:
        """从 PyTorch torchvision ResNet50 权重文件加载预训练参数。

        只加载 Backbone 部分（忽略分类头 fc.*），
        权重键名需与本模型的层名对齐。

        Args:
            pth_path: torchvision ResNet50 权重文件路径（.pth）
        """
        try:
            import torch
            state_dict = torch.load(pth_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # 过滤掉分类头权重
            backbone_dict = {
                k: v.numpy()
                for k, v in state_dict.items()
                if not k.startswith('fc.')
            }

            # 将 numpy 数组转为 jt.Var 并加载
            jt_state = {k: jt.array(v) for k, v in backbone_dict.items()}
            self.load_state_dict(jt_state, strict=False)
            print(f'[ResNet50CBAM] 预训练权重加载完成: {pth_path}')
        except ImportError:
            print('[ResNet50CBAM] 警告：torch 未安装，跳过预训练权重加载。')
        except Exception as e:
            print(f'[ResNet50CBAM] 警告：权重加载失败（{e}），使用随机初始化。')

    def execute(self, x: jt.Var) -> List[jt.Var]:
        """前向传播，返回 4 个 stage 的特征图。

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            [C2, C3, C4, C5]，对应 layer1~layer4 的输出
            通道数：[256, 512, 1024, 2048]
            步长：  [4,   8,   16,   32]
        """
        # Stem: [B, 3, H, W] -> [B, 64, H/4, W/4]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Stage 1~4
        c2 = self.layer1(x)    # [B, 256,  H/4,  W/4 ]
        c3 = self.layer2(c2)   # [B, 512,  H/8,  W/8 ]
        c4 = self.layer3(c3)   # [B, 1024, H/16, W/16]
        c5 = self.layer4(c4)   # [B, 2048, H/32, W/32]

        return [c2, c3, c4, c5]
