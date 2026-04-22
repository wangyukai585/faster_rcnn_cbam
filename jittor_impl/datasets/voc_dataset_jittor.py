"""
Jittor 版 PASCAL VOC 数据集

解析 VOC XML 标注文件，提供训练/测试数据加载，
支持标准数据增强（随机翻转、Resize）和归一化。

VOC 20 类别（0-indexed）：
  0:aeroplane 1:bicycle 2:bird 3:boat 4:bottle
  5:bus 6:car 7:cat 8:chair 9:cow
  10:diningtable 11:dog 12:horse 13:motorbike 14:person
  15:pottedplant 16:sheep 17:sofa 18:train 19:tvmonitor
"""

import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import jittor as jt
from jittor.dataset import Dataset

# VOC 类别名到索引的映射
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
CLASS2IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

# ImageNet 归一化均值和标准差（RGB 格式）
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)


def parse_voc_annotation(xml_path: str) -> Dict:
    """解析单个 VOC XML 标注文件。

    Args:
        xml_path: XML 文件路径

    Returns:
        dict 包含：
          'boxes':  [[x1,y1,x2,y2], ...] (float，图像坐标系)
          'labels': [class_idx, ...]      (int，0-indexed)
          'difficult': [bool, ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes, labels, difficult = [], [], []

    for obj in root.findall('object'):
        name = obj.find('name').text.strip().lower()
        if name not in CLASS2IDX:
            continue  # 跳过不在 VOC 20 类中的对象

        diff = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bndbox = obj.find('bndbox')
        x1 = float(bndbox.find('xmin').text)
        y1 = float(bndbox.find('ymin').text)
        x2 = float(bndbox.find('xmax').text)
        y2 = float(bndbox.find('ymax').text)

        boxes.append([x1, y1, x2, y2])
        labels.append(CLASS2IDX[name])
        difficult.append(bool(diff))

    return {
        'boxes': np.array(boxes, dtype=np.float32).reshape(-1, 4),
        'labels': np.array(labels, dtype=np.int32),
        'difficult': np.array(difficult, dtype=bool),
    }


class VOCDataset(Dataset):
    """PASCAL VOC 数据集（Jittor 版）

    支持合并多个 split（如 VOC2007 trainval + VOC2012 trainval）。

    Args:
        data_root (str):  VOCdevkit 根目录（含 VOC2007/ 和 VOC2012/）
        splits    (list): [('VOC2007', 'trainval'), ('VOC2012', 'trainval')] 等
        img_scale (tuple):(max_long_edge, max_short_edge)，默认 (1000, 600)
        is_train  (bool): 是否为训练模式（开启随机翻转）
        use_difficult (bool): 是否使用 difficult 标注，默认 False
    """

    def __init__(
        self,
        data_root: str,
        splits: List[Tuple[str, str]] = (('VOC2007', 'trainval'),),
        img_scale: Tuple[int, int] = (1000, 600),
        is_train: bool = True,
        use_difficult: bool = False,
    ):
        super().__init__()

        self.data_root = data_root
        self.img_scale = img_scale
        self.is_train = is_train
        self.use_difficult = use_difficult

        # 收集所有图像的（图像路径，标注路径）
        self.samples: List[Tuple[str, str]] = []
        for year, split in splits:
            split_file = os.path.join(data_root, year, 'ImageSets', 'Main', f'{split}.txt')
            img_dir  = os.path.join(data_root, year, 'JPEGImages')
            ann_dir  = os.path.join(data_root, year, 'Annotations')

            with open(split_file) as f:
                img_ids = [line.strip() for line in f if line.strip()]

            for img_id in img_ids:
                img_path = os.path.join(img_dir, f'{img_id}.jpg')
                ann_path = os.path.join(ann_dir, f'{img_id}.xml')
                if os.path.exists(img_path) and os.path.exists(ann_path):
                    self.samples.append((img_path, ann_path))

        self.total_len = len(self.samples)

    def __len__(self) -> int:
        return self.total_len

    def _resize(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """等比例 Resize 到指定最大尺寸。

        以最长边和最短边分别不超过 img_scale[0] 和 img_scale[1] 为约束。

        Args:
            img   : [H, W, 3]
            boxes : [N, 4]，格式 x1y1x2y2

        Returns:
            img_resized, boxes_resized
        """
        H, W = img.shape[:2]
        max_long, max_short = self.img_scale

        # 计算缩放比例
        scale = min(max_long / max(H, W), max_short / min(H, W))
        if scale == 1.0:
            return img, boxes

        new_W = int(W * scale)
        new_H = int(H * scale)

        img_pil = Image.fromarray(img).resize((new_W, new_H), Image.BILINEAR)
        img_resized = np.array(img_pil)

        if boxes.shape[0] > 0:
            boxes_resized = boxes * scale
        else:
            boxes_resized = boxes

        return img_resized, boxes_resized

    def _random_flip(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """随机水平翻转图像和边界框（训练时 50% 概率触发）。

        Args:
            img  : [H, W, 3]
            boxes: [N, 4]，格式 x1y1x2y2

        Returns:
            img_flipped, boxes_flipped
        """
        if random.random() < 0.5:
            W = img.shape[1]
            # 翻转图像
            img = img[:, ::-1, :].copy()
            # 翻转框的 x 坐标
            if boxes.shape[0] > 0:
                x1 = W - boxes[:, 2]
                x2 = W - boxes[:, 0]
                boxes = boxes.copy()
                boxes[:, 0] = x1
                boxes[:, 2] = x2
        return img, boxes

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """ImageNet 归一化（减均值除标准差）。

        Args:
            img: [H, W, 3]，uint8 或 float32（RGB）

        Returns:
            img_norm: [H, W, 3]，float32
        """
        img = img.astype(np.float32)
        img = (img - _MEAN) / _STD
        return img

    def __getitem__(self, idx: int):
        """获取单个样本。

        Returns:
            img_tensor: jt.Var [3, H, W]，float32
            boxes     : jt.Var [N, 4]，float32，格式 x1y1x2y2
            labels    : jt.Var [N]，int32，0-indexed
        """
        img_path, ann_path = self.samples[idx]

        # 加载图像（BGR → RGB）
        img = np.array(Image.open(img_path).convert('RGB'))

        # 解析标注
        ann = parse_voc_annotation(ann_path)
        boxes  = ann['boxes'].copy()    # [N, 4]
        labels = ann['labels'].copy()   # [N]
        diff   = ann['difficult']

        # 过滤 difficult 样本（训练时可选）
        if not self.use_difficult and self.is_train:
            keep = ~diff
            boxes  = boxes[keep]
            labels = labels[keep]

        # 数据增强（仅训练时）
        if self.is_train:
            img, boxes = self._random_flip(img, boxes)

        # Resize
        img, boxes = self._resize(img, boxes)

        # 归一化
        img = self._normalize(img)

        # [H, W, 3] → [3, H, W]
        img = img.transpose(2, 0, 1)

        # 转为 Jittor 张量
        img_tensor = jt.array(img, dtype=jt.float32)
        boxes_var  = jt.array(boxes, dtype=jt.float32)
        labels_var = jt.array(labels, dtype=jt.int32)

        return img_tensor, boxes_var, labels_var

    def collate_batch(self, batch):
        """自定义 batch 拼接：图像 pad 到相同大小，框保持列表形式。

        Jittor 差异：Dataset 的 collate_batch 替代 PyTorch 的 collate_fn。

        Args:
            batch: list of (img_tensor, boxes, labels)

        Returns:
            images      : jt.Var [B, 3, H_max, W_max]（pad 0）
            boxes_list  : list of jt.Var [N_i, 4]
            labels_list : list of jt.Var [N_i]
        """
        imgs, boxes_list, labels_list = zip(*batch)

        # 找到最大高宽，用 0 padding（归一化后 0 对应 -mean/std，不影响 BN）
        max_H = max(img.shape[1] for img in imgs)
        max_W = max(img.shape[2] for img in imgs)

        padded = []
        for img in imgs:
            C, H, W = img.shape
            pad = jt.zeros((C, max_H, max_W), dtype=jt.float32)
            pad[:, :H, :W] = img
            padded.append(pad)

        images = jt.stack(padded, dim=0)  # [B, 3, H_max, W_max]
        return images, list(boxes_list), list(labels_list)
