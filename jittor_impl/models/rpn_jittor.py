"""
Jittor 版 RPN（Region Proposal Network）

对每个 FPN 输出特征图（P2~P6）：
  - 生成 anchor（3 种比例 × 1 种尺度 = 3 个/位置）
  - 3×3 卷积提取特征
  - 预测前景/背景分类分数和位置偏移
  - 推理时：解码偏移 → 剪裁到图像边界 → NMS → 输出 proposals

训练时同时计算：
  - RPN 分类损失（BCE）：判断 anchor 是否含目标
  - RPN 回归损失（SmoothL1）：预测前景 anchor 的位置偏移
"""

import math
from typing import Dict, List, Optional, Tuple

import jittor as jt
import jittor.nn as nn


# ============================================================
# Anchor 生成
# ============================================================

def generate_anchors(
    base_size: int,
    ratios: List[float],
    scales: List[float],
) -> jt.Var:
    """为单个位置生成基础 anchor（相对于该位置的偏移，以中心为原点）。

    Args:
        base_size: anchor 基础边长（等于 FPN 步长 × scale）
        ratios:    宽高比列表 [0.5, 1.0, 2.0]
        scales:    尺度列表 [1.0]（尺度已融入 base_size）

    Returns:
        anchors: [num_anchors, 4]，格式 [x1, y1, x2, y2]（相对于图像左上角）
    """
    anchors = []
    for scale in scales:
        for ratio in ratios:
            # 保持面积不变，调整宽高比
            area = (base_size * scale) ** 2
            w = math.sqrt(area / ratio)  # width
            h = w * ratio                # height
            # 以原点为中心的 anchor 坐标
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return jt.array(anchors, dtype=jt.float32)  # [num_anchors, 4]


def get_all_anchors(
    feat_sizes: List[Tuple[int, int]],
    strides: List[int],
    base_sizes: List[int],
    ratios: List[float] = (0.5, 1.0, 2.0),
) -> List[jt.Var]:
    """为所有 FPN 特征层生成 anchor。

    Args:
        feat_sizes : 各层特征图尺寸 [(H2,W2), ..., (H6,W6)]
        strides    : 各层步长 [4, 8, 16, 32, 64]
        base_sizes : 各层 anchor 基础边长 [32, 64, 128, 256, 512]
        ratios     : 宽高比

    Returns:
        all_anchors: 每层 [H_i*W_i*num_anchors, 4]（绝对坐标，格式 x1y1x2y2）
    """
    all_anchors = []
    num_anchors = len(ratios)

    for (H, W), stride, base_size in zip(feat_sizes, strides, base_sizes):
        # 基础 anchor（相对于中心）: [num_anchors, 4]
        base = generate_anchors(base_size, ratios, [1.0])  # [3, 4]

        # 特征图上每个位置的中心坐标（图像坐标系）
        ys = jt.arange(H, dtype=jt.float32) * stride + stride / 2
        xs = jt.arange(W, dtype=jt.float32) * stride + stride / 2

        # 生成网格: [H, W, 2]
        grid_y, grid_x = jt.meshgrid(ys, xs)
        grid_y = grid_y.reshape(-1)   # [H*W]
        grid_x = grid_x.reshape(-1)   # [H*W]

        # 中心坐标扩展为 [H*W, 1, 4] 偏移量（只平移 x1y1x2y2 的中心）
        shifts = jt.stack([grid_x, grid_y, grid_x, grid_y], dim=1)  # [H*W, 4]
        shifts = shifts.reshape(-1, 1, 4)                             # [H*W, 1, 4]
        base = base.reshape(1, num_anchors, 4)                        # [1, NA, 4]

        # 广播相加，得到所有位置的所有 anchor
        anchors = (shifts + base).reshape(-1, 4)  # [H*W*NA, 4]
        all_anchors.append(anchors)

    return all_anchors


# ============================================================
# Box 编解码
# ============================================================

def decode_boxes(anchors: jt.Var, deltas: jt.Var) -> jt.Var:
    """将 RPN 预测的偏移量应用到 anchors，解码出预测框。

    标准 Faster R-CNN 解码公式：
      pred_cx = delta_x * anchor_w + anchor_cx
      pred_cy = delta_y * anchor_h + anchor_cy
      pred_w  = exp(delta_w) * anchor_w
      pred_h  = exp(delta_h) * anchor_h

    Args:
        anchors: [N, 4]，格式 x1y1x2y2
        deltas:  [N, 4]，格式 dx dy dw dh

    Returns:
        boxes: [N, 4]，格式 x1y1x2y2
    """
    aw = anchors[:, 2] - anchors[:, 0]   # anchor width
    ah = anchors[:, 3] - anchors[:, 1]   # anchor height
    acx = anchors[:, 0] + 0.5 * aw       # anchor center x
    acy = anchors[:, 1] + 0.5 * ah       # anchor center y

    # clamp delta_w/h 防止 exp 溢出
    dx, dy = deltas[:, 0], deltas[:, 1]
    dw = jt.clamp(deltas[:, 2], max_v=4.0)
    dh = jt.clamp(deltas[:, 3], max_v=4.0)

    pred_cx = dx * aw + acx
    pred_cy = dy * ah + acy
    pred_w = jt.exp(dw) * aw
    pred_h = jt.exp(dh) * ah

    x1 = pred_cx - 0.5 * pred_w
    y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w
    y2 = pred_cy + 0.5 * pred_h

    return jt.stack([x1, y1, x2, y2], dim=1)


def encode_boxes(anchors: jt.Var, gt_boxes: jt.Var) -> jt.Var:
    """将 GT 框编码为相对于 anchor 的偏移量（训练时使用）。

    Args:
        anchors: [N, 4]，格式 x1y1x2y2
        gt_boxes:[N, 4]，格式 x1y1x2y2

    Returns:
        deltas: [N, 4]，格式 dx dy dw dh
    """
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + 0.5 * aw
    acy = anchors[:, 1] + 0.5 * ah

    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]
    gcx = gt_boxes[:, 0] + 0.5 * gw
    gcy = gt_boxes[:, 1] + 0.5 * gh

    dx = (gcx - acx) / (aw + 1e-8)
    dy = (gcy - acy) / (ah + 1e-8)
    dw = jt.log(gw / (aw + 1e-8) + 1e-8)
    dh = jt.log(gh / (ah + 1e-8) + 1e-8)

    return jt.stack([dx, dy, dw, dh], dim=1)


# ============================================================
# IoU 计算
# ============================================================

def box_iou(boxes_a: jt.Var, boxes_b: jt.Var) -> jt.Var:
    """计算两组框的 IoU 矩阵。

    Args:
        boxes_a: [M, 4]，格式 x1y1x2y2
        boxes_b: [N, 4]，格式 x1y1x2y2

    Returns:
        iou: [M, N]
    """
    M = boxes_a.shape[0]
    N = boxes_b.shape[0]

    # 扩展维度用于广播
    a = boxes_a.reshape(M, 1, 4)  # [M, 1, 4]
    b = boxes_b.reshape(1, N, 4)  # [1, N, 4]

    inter_x1 = jt.maximum(a[:, :, 0], b[:, :, 0])
    inter_y1 = jt.maximum(a[:, :, 1], b[:, :, 1])
    inter_x2 = jt.minimum(a[:, :, 2], b[:, :, 2])
    inter_y2 = jt.minimum(a[:, :, 3], b[:, :, 3])

    inter_w = jt.clamp(inter_x2 - inter_x1, min_v=0)
    inter_h = jt.clamp(inter_y2 - inter_y1, min_v=0)
    inter_area = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a.reshape(M, 1) + area_b.reshape(1, N) - inter_area
    iou = inter_area / (union + 1e-8)
    return iou


# ============================================================
# RPN 主模块
# ============================================================

class RPN(nn.Module):
    """区域候选网络（Jittor 版）

    对每个 FPN 特征层：
      1. 3×3 共享卷积
      2. 分类头（判断 anchor 是否含目标）
      3. 回归头（预测 anchor 偏移量）

    Args:
        in_channels    (int):   FPN 输出通道数，默认 256
        num_anchors    (int):   每个位置的 anchor 数，默认 3（3 种比例）
        pos_iou_thr    (float): 正样本 IoU 阈值，默认 0.7
        neg_iou_thr    (float): 负样本 IoU 阈值，默认 0.3
        num_sample     (int):   每张图采样 anchor 数，默认 256
        pos_fraction   (float): 正样本占比，默认 0.5
        nms_pre        (int):   NMS 前保留的最大候选数，默认 2000
        max_proposals  (int):   NMS 后保留的最大候选数，默认 1000
        nms_iou_thr    (float): NMS IoU 阈值，默认 0.7
    """

    # FPN 各层步长和 anchor 基础边长
    _STRIDES    = [4, 8, 16, 32, 64]
    _BASE_SIZES = [32, 64, 128, 256, 512]
    _RATIOS     = [0.5, 1.0, 2.0]

    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 3,
        pos_iou_thr: float = 0.7,
        neg_iou_thr: float = 0.3,
        num_sample: int = 256,
        pos_fraction: float = 0.5,
        nms_pre: int = 2000,
        max_proposals: int = 1000,
        nms_iou_thr: float = 0.7,
    ):
        super().__init__()

        self.num_anchors = num_anchors
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.num_sample = num_sample
        self.pos_fraction = pos_fraction
        self.nms_pre = nms_pre
        self.max_proposals = max_proposals
        self.nms_iou_thr = nms_iou_thr

        # 3×3 共享卷积（各 FPN 层共享参数）
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # 分类头：每个位置输出 num_anchors 个前景分数（二分类用 sigmoid）
        self.cls_head = nn.Conv2d(in_channels, num_anchors, kernel_size=1)

        # 回归头：每个位置输出 num_anchors × 4 个偏移量
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        # 权重初始化
        for layer in [self.conv, self.cls_head, self.reg_head]:
            jt.init.gauss_(layer.weight, mean=0.0, std=0.01)
            if layer.bias is not None:
                layer.bias = jt.zeros(layer.bias.shape)

    def _forward_single(self, feat: jt.Var) -> Tuple[jt.Var, jt.Var]:
        """对单个 FPN 特征层做 RPN 前向。

        Args:
            feat: [B, 256, H, W]

        Returns:
            cls_scores: [B, num_anchors*H*W]（sigmoid 前）
            reg_preds:  [B, num_anchors*H*W, 4]
        """
        B = feat.shape[0]
        feat = self.relu(self.conv(feat))

        # 分类：[B, A, H, W] -> [B, A*H*W]
        cls = self.cls_head(feat)
        cls = cls.permute(0, 2, 3, 1).reshape(B, -1)

        # 回归：[B, A*4, H, W] -> [B, A*H*W, 4]
        reg = self.reg_head(feat)
        reg = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)

        return cls, reg

    def execute(
        self,
        fpn_feats: List[jt.Var],
        img_size: Tuple[int, int],
        gt_boxes_list: Optional[List[jt.Var]] = None,
    ) -> Dict:
        """RPN 前向传播。

        Args:
            fpn_feats:    FPN 输出 [P2,...,P6]，各 [B, 256, Hi, Wi]
            img_size:     (H, W) 图像尺寸（用于剪裁 proposal 到图像内）
            gt_boxes_list:训练时提供，每张图的 GT 框 [num_gt, 4]

        Returns:
            dict 包含：
              'proposals': 每张图的候选框列表 [List[jt.Var]]
              'rpn_cls_loss'（训练时）
              'rpn_reg_loss'（训练时）
        """
        B = fpn_feats[0].shape[0]

        # 步骤 1：对每个 FPN 层做前向
        all_cls, all_reg = [], []
        feat_sizes = [(f.shape[2], f.shape[3]) for f in fpn_feats]

        for feat in fpn_feats:
            cls, reg = self._forward_single(feat)
            all_cls.append(cls)   # [B, Hi*Wi*A]
            all_reg.append(reg)   # [B, Hi*Wi*A, 4]

        # 步骤 2：拼接所有层的预测
        cat_cls = jt.concat(all_cls, dim=1)  # [B, total_anchors]
        cat_reg = jt.concat(all_reg, dim=1)  # [B, total_anchors, 4]

        # 步骤 3：生成所有 anchor
        all_anchors = get_all_anchors(feat_sizes, self._STRIDES, self._BASE_SIZES, self._RATIOS)
        # cat_anchors: [total_anchors, 4]
        cat_anchors = jt.concat(all_anchors, dim=0)

        # 步骤 4：为每张图生成 proposals（推理时用 NMS）
        proposals_list = []
        H_img, W_img = img_size

        for b in range(B):
            cls_b = jt.sigmoid(cat_cls[b])       # [total_anchors]，前景概率
            reg_b = cat_reg[b]                    # [total_anchors, 4]

            # 按分数取 top-k（加速 NMS）
            if cls_b.shape[0] > self.nms_pre:
                topk_scores, topk_idx = jt.argsort(cls_b, descending=True)
                topk_idx = topk_idx[:self.nms_pre]
                cls_b = cls_b[topk_idx]
                reg_b = reg_b[topk_idx]
                anchors_b = cat_anchors[topk_idx]
            else:
                anchors_b = cat_anchors

            # 解码 proposal
            proposals = decode_boxes(anchors_b, reg_b)

            # 剪裁到图像边界
            proposals[:, 0] = jt.clamp(proposals[:, 0], min_v=0, max_v=W_img)
            proposals[:, 1] = jt.clamp(proposals[:, 1], min_v=0, max_v=H_img)
            proposals[:, 2] = jt.clamp(proposals[:, 2], min_v=0, max_v=W_img)
            proposals[:, 3] = jt.clamp(proposals[:, 3], min_v=0, max_v=H_img)

            # 过滤太小的框
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            keep = (w >= 1) & (h >= 1)
            proposals = proposals[keep]
            scores = cls_b[keep]

            # NMS
            if proposals.shape[0] > 0:
                # Jittor 提供 nms 函数
                keep_idx = jt.nms(
                    jt.concat([proposals, scores.reshape(-1, 1)], dim=1),
                    self.nms_iou_thr,
                )
                keep_idx = keep_idx[:self.max_proposals]
                proposals = proposals[keep_idx]

            proposals_list.append(proposals)

        result = {'proposals': proposals_list}

        # 步骤 5：训练时计算 RPN 损失
        if self.training and gt_boxes_list is not None:
            rpn_cls_loss, rpn_reg_loss = self._compute_loss(
                cat_cls, cat_reg, cat_anchors, gt_boxes_list, img_size
            )
            result['rpn_cls_loss'] = rpn_cls_loss
            result['rpn_reg_loss'] = rpn_reg_loss

        return result

    def _compute_loss(
        self,
        cls_preds: jt.Var,
        reg_preds: jt.Var,
        anchors: jt.Var,
        gt_boxes_list: List[jt.Var],
        img_size: Tuple[int, int],
    ) -> Tuple[jt.Var, jt.Var]:
        """计算 RPN 分类损失和回归损失。

        分配策略：
          IoU >= pos_iou_thr → 正样本（label=1）
          IoU <  neg_iou_thr → 负样本（label=0）
          其余 → 忽略（label=-1）

        Args:
            cls_preds:     [B, N]，sigmoid 前
            reg_preds:     [B, N, 4]
            anchors:       [N, 4]
            gt_boxes_list: [B] × [num_gt, 4]
            img_size:      (H, W)

        Returns:
            cls_loss, reg_loss
        """
        B = cls_preds.shape[0]
        H_img, W_img = img_size
        num_sample = self.num_sample
        num_pos = int(num_sample * self.pos_fraction)

        # 过滤超出图像边界的 anchor
        valid_mask = (
            (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= W_img) & (anchors[:, 3] <= H_img)
        )
        valid_anchors = anchors[valid_mask]

        all_cls_loss = jt.zeros(1)
        all_reg_loss = jt.zeros(1)

        for b in range(B):
            gt_boxes = gt_boxes_list[b]  # [num_gt, 4]
            if gt_boxes.shape[0] == 0:
                continue

            # IoU 矩阵: [num_valid_anchors, num_gt]
            iou = box_iou(valid_anchors, gt_boxes)
            max_iou = jt.max(iou, dim=1)            # [num_valid_anchors]
            gt_idx  = jt.argmax(iou, dim=1)[0]      # [num_valid_anchors]

            # 分配标签
            labels = jt.full((valid_anchors.shape[0],), -1, dtype=jt.int32)
            labels = jt.where(max_iou >= self.pos_iou_thr, jt.ones_like(labels), labels)
            labels = jt.where(max_iou < self.neg_iou_thr, jt.zeros_like(labels), labels)

            # 采样：正样本不超过 num_pos，负样本补齐到 num_sample
            pos_idx = (labels == 1).numpy().nonzero()[0]
            neg_idx = (labels == 0).numpy().nonzero()[0]
            import numpy as np
            if len(pos_idx) > num_pos:
                pos_idx = np.random.choice(pos_idx, num_pos, replace=False)
            num_neg = min(num_sample - len(pos_idx), len(neg_idx))
            if len(neg_idx) > num_neg:
                neg_idx = np.random.choice(neg_idx, num_neg, replace=False)

            sample_idx = np.concatenate([pos_idx, neg_idx])
            if len(sample_idx) == 0:
                continue

            sample_idx = jt.array(sample_idx, dtype=jt.int32)

            # 将 valid_mask 下标映射回全量 anchor 下标
            valid_indices = valid_mask.numpy().nonzero()[0]
            full_sample_idx = jt.array(valid_indices[sample_idx.numpy()])

            # 分类损失（BCE）
            cls_target = (labels[sample_idx] == 1).float()
            cls_pred   = jt.sigmoid(cls_preds[b][full_sample_idx])
            cls_loss   = nn.binary_cross_entropy(cls_pred, cls_target)

            # 回归损失（仅对正样本）
            pos_mask = (labels[sample_idx] == 1)
            if pos_mask.sum() > 0:
                pos_full_idx = full_sample_idx[pos_mask]
                reg_pred_pos = reg_preds[b][pos_full_idx]
                pos_anchor   = valid_anchors[sample_idx[pos_mask]]
                matched_gt   = gt_boxes[gt_idx[sample_idx[pos_mask]]]
                reg_target   = encode_boxes(pos_anchor, matched_gt)
                reg_loss     = nn.smooth_l1_loss(reg_pred_pos, reg_target)
                all_reg_loss = all_reg_loss + reg_loss

            all_cls_loss = all_cls_loss + cls_loss

        return all_cls_loss / B, all_reg_loss / B
