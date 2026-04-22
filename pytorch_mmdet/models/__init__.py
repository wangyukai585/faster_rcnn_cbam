"""
自定义模型模块入口

导入顺序：先导入基础模块（cbam），再导入依赖它的模块（resnet_cbam），
确保 MMDetection 注册表在 import 时完成注册。
"""

from .cbam import CBAM, ChannelAttention, SpatialAttention
from .resnet_cbam import ResNetCBAM, BottleneckWithCBAM

__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "BottleneckWithCBAM",
    "ResNetCBAM",
]
