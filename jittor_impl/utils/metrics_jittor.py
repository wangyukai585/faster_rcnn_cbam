"""
Jittor 版 mAP 计算工具

实现 PASCAL VOC 标准的 mAP@0.5 评估（11点插值法）。

使用流程：
  1. 实例化 VOCEvaluator
  2. 训练结束后对测试集逐图调用 update(detections, gt_annotations)
  3. 调用 compute() 得到每类 AP 和总 mAP
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class VOCEvaluator:
    """PASCAL VOC mAP 评估器（纯 NumPy 实现，与 Jittor 兼容）

    Args:
        num_classes  (int):   类别数，VOC 为 20
        iou_threshold(float): IoU 阈值，默认 0.5
        class_names  (list):  类别名称列表（可选，用于打印）
    """

    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    ]

    def __init__(
        self,
        num_classes: int = 20,
        iou_threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.class_names = class_names or self.VOC_CLASSES[:num_classes]

        # 每类存储所有图像的检测结果和 GT
        # detections[cls] = list of (score, img_id, box)
        # gt[cls][img_id] = list of (box, detected_flag)
        self._detections: Dict[int, List] = {c: [] for c in range(num_classes)}
        self._gt: Dict[int, Dict] = {c: {} for c in range(num_classes)}
        self._img_count = 0

    def reset(self) -> None:
        """清空所有已积累的检测结果和 GT。"""
        self._detections = {c: [] for c in range(self.num_classes)}
        self._gt = {c: {} for c in range(self.num_classes)}
        self._img_count = 0

    def update(
        self,
        det_boxes: np.ndarray,
        det_scores: np.ndarray,
        det_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        img_id: Optional[int] = None,
    ) -> None:
        """更新单张图的检测结果和 GT 标注。

        Args:
            det_boxes  : [N, 4] 检测框（x1y1x2y2）
            det_scores : [N] 置信度分数
            det_labels : [N] 预测类别（0-indexed）
            gt_boxes   : [M, 4] GT 框
            gt_labels  : [M] GT 类别（0-indexed）
            img_id     : 图像唯一标识符（可选，默认自增）
        """
        if img_id is None:
            img_id = self._img_count
        self._img_count += 1

        # 记录 GT
        for cls_id in range(self.num_classes):
            gt_mask = gt_labels == cls_id
            gt_cls_boxes = gt_boxes[gt_mask] if gt_mask.any() else np.zeros((0, 4))
            self._gt[cls_id][img_id] = {
                'boxes': gt_cls_boxes,
                'detected': [False] * len(gt_cls_boxes),
            }

        # 记录检测结果
        for i in range(len(det_boxes)):
            cls_id = int(det_labels[i])
            if 0 <= cls_id < self.num_classes:
                self._detections[cls_id].append((
                    float(det_scores[i]), img_id, det_boxes[i].copy()
                ))

    def _compute_ap_single_class(self, cls_id: int) -> float:
        """计算单类别的 AP（11点插值法，VOC 标准）。

        Args:
            cls_id: 类别索引

        Returns:
            ap: float，该类别的平均精度（0~1）
        """
        detections = self._detections[cls_id]

        # 按置信度降序排列
        detections.sort(key=lambda x: -x[0])

        # 统计该类 GT 总数（用于计算 recall）
        num_gt = sum(
            len(v['boxes']) for v in self._gt[cls_id].values()
        )
        if num_gt == 0:
            return 0.0

        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))

        for i, (score, img_id, det_box) in enumerate(detections):
            gt_info = self._gt[cls_id].get(img_id, {'boxes': np.zeros((0, 4)), 'detected': []})
            gt_boxes = gt_info['boxes']
            detected = gt_info['detected']

            if len(gt_boxes) == 0:
                fp[i] = 1
                continue

            # 计算该检测框与该图所有 GT 框的 IoU
            iou = self._box_iou_single(det_box, gt_boxes)
            max_iou_idx = np.argmax(iou)
            max_iou = iou[max_iou_idx]

            if max_iou >= self.iou_threshold and not detected[max_iou_idx]:
                tp[i] = 1
                gt_info['detected'][max_iou_idx] = True
            else:
                fp[i] = 1

        # 计算累积精确率和召回率
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall    = cum_tp / (num_gt + 1e-8)
        precision = cum_tp / (cum_tp + cum_fp + 1e-8)

        # 11点插值法（VOC 2007 标准）
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            p = precision[recall >= thr].max() if (recall >= thr).any() else 0.0
            ap += p / 11.0

        return float(ap)

    @staticmethod
    def _box_iou_single(box: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
        """计算单个检测框与多个 GT 框的 IoU。

        Args:
            box     : [4]，格式 x1y1x2y2
            gt_boxes: [M, 4]

        Returns:
            iou: [M]
        """
        x1 = np.maximum(box[0], gt_boxes[:, 0])
        y1 = np.maximum(box[1], gt_boxes[:, 1])
        x2 = np.minimum(box[2], gt_boxes[:, 2])
        y2 = np.minimum(box[3], gt_boxes[:, 3])

        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_gt  = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        union = area_box + area_gt - inter
        return inter / (union + 1e-8)

    def compute(self) -> Dict:
        """计算所有类别的 AP 和总 mAP。

        Returns:
            dict 包含：
              'mAP'    : 总 mAP（float）
              'per_class_ap': {class_name: ap} 字典
        """
        per_class_ap = {}
        aps = []

        for cls_id in range(self.num_classes):
            ap = self._compute_ap_single_class(cls_id)
            cls_name = self.class_names[cls_id]
            per_class_ap[cls_name] = ap
            aps.append(ap)

        mAP = float(np.mean(aps))
        return {'mAP': mAP, 'per_class_ap': per_class_ap}

    def print_results(self, results: Optional[Dict] = None) -> None:
        """打印每类 AP 和总 mAP。"""
        if results is None:
            results = self.compute()

        print('\n' + '=' * 50)
        print('VOC mAP 评估结果（11点插值法，IoU≥0.5）')
        print('=' * 50)
        for cls_name, ap in results['per_class_ap'].items():
            print(f'  {cls_name:<20s}: {ap * 100:.2f}%')
        print('-' * 50)
        print(f'  {"mAP@0.5":<20s}: {results["mAP"] * 100:.2f}%')
        print('=' * 50)
