"""
Jittor 版完整 Faster R-CNN + CBAM 检测器

模块串联关系：
  输入图像
    └─→ ResNet50-CBAM（Backbone）→ [C2, C3, C4, C5]
          └─→ FPN（Neck）         → [P2, P3, P4, P5, P6]
                └─→ RPN           → proposals（候选框）
                      └─→ ROI Align（7×7）
                            └─→ FC1(1024) → ReLU
                                  └─→ FC2(1024) → ReLU
                                        ├─→ cls_head → 分类分数（21类，含背景）
                                        └─→ reg_head → 位置偏移（20类×4）

训练时返回 loss 字典：
  {'rpn_cls_loss', 'rpn_reg_loss', 'rcnn_cls_loss', 'rcnn_reg_loss', 'loss'}

推理时返回检测结果列表（每张图）：
  [{'boxes': [N,4], 'scores': [N], 'labels': [N]}, ...]
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import jittor as jt
import jittor.nn as nn

from .resnet_jittor import ResNet50CBAM
from .fpn_jittor import FPN
from .rpn_jittor import RPN, box_iou, decode_boxes, encode_boxes


# ============================================================
# ROI Head（ROI Align + 分类 + 回归）
# ============================================================

class ROIHead(nn.Module):
    """ROI 检测头（Jittor 版）

    结构：
      proposals → ROI Align(7×7) → Flatten → FC1 → ReLU → FC2 → ReLU
               → 分类头（num_classes+1）
               → 回归头（num_classes×4）

    Args:
        in_channels    (int):   ROI Align 输入通道（FPN 输出通道），默认 256
        roi_size       (int):   ROI Align 输出尺寸，默认 7
        fc_out_channels(int):   全连接层宽度，默认 1024
        num_classes    (int):   前景类别数（不含背景），VOC 为 20
        pos_iou_thr    (float): 正样本 IoU 阈值
        neg_iou_thr    (float): 负样本 IoU 阈值
        num_sample     (int):   每张图采样 proposal 数
        pos_fraction   (float): 正样本占比
        score_thr      (float): 推理时保留的最低置信度阈值
        nms_iou_thr    (float): 推理时 NMS 阈值
    """

    def __init__(
        self,
        in_channels: int = 256,
        roi_size: int = 7,
        fc_out_channels: int = 1024,
        num_classes: int = 20,
        pos_iou_thr: float = 0.5,
        neg_iou_thr: float = 0.5,
        num_sample: int = 512,
        pos_fraction: float = 0.25,
        score_thr: float = 0.05,
        nms_iou_thr: float = 0.5,
        featmap_strides: List[int] = (4, 8, 16, 32),
    ):
        super().__init__()

        self.roi_size = roi_size
        self.num_classes = num_classes
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.num_sample = num_sample
        self.pos_fraction = pos_fraction
        self.score_thr = score_thr
        self.nms_iou_thr = nms_iou_thr
        self.featmap_strides = featmap_strides

        flat_dim = in_channels * roi_size * roi_size

        # 共享全连接层
        self.fc1 = nn.Linear(flat_dim, fc_out_channels)
        self.fc2 = nn.Linear(fc_out_channels, fc_out_channels)
        self.relu = nn.ReLU()

        # 分类头：num_classes + 1（含背景）
        self.cls_head = nn.Linear(fc_out_channels, num_classes + 1)
        # 回归头：num_classes × 4（类别相关回归）
        self.reg_head = nn.Linear(fc_out_channels, num_classes * 4)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.cls_head, self.reg_head]:
            jt.init.gauss_(layer.weight, mean=0.0, std=0.01)
            layer.bias = jt.zeros(layer.bias.shape)
        for layer in [self.fc1, self.fc2]:
            jt.init.gauss_(layer.weight, mean=0.0, std=0.01)
            layer.bias = jt.zeros(layer.bias.shape)

    def _assign_and_sample(
        self,
        proposals: jt.Var,
        gt_boxes: jt.Var,
        gt_labels: jt.Var,
    ) -> Tuple[jt.Var, jt.Var, jt.Var]:
        """将 GT 框分配给 proposal，并采样正负样本。

        Returns:
            sampled_proposals : [K, 4]
            cls_targets       : [K]     0=背景，1~num_classes=前景
            reg_targets       : [K, 4]
        """
        num_pos = int(self.num_sample * self.pos_fraction)

        # 合并 proposals 和 GT 框作为候选
        all_boxes = jt.concat([proposals, gt_boxes], dim=0)

        iou = box_iou(all_boxes, gt_boxes)          # [M, num_gt]
        max_iou = jt.max(iou, dim=1)                # [M]
        gt_idx  = jt.argmax(iou, dim=1)[0]          # [M]

        # 分配标签
        labels = jt.full((all_boxes.shape[0],), -1, dtype=jt.int32)
        pos_mask_np = (max_iou >= self.pos_iou_thr).numpy().astype(bool)
        neg_mask_np = (max_iou < self.neg_iou_thr).numpy().astype(bool)
        labels_np = labels.numpy()
        labels_np[pos_mask_np] = 1
        labels_np[neg_mask_np] = 0

        pos_idx = np.where(labels_np == 1)[0]
        neg_idx = np.where(labels_np == 0)[0]
        if len(pos_idx) > num_pos:
            pos_idx = np.random.choice(pos_idx, num_pos, replace=False)
        num_neg = min(self.num_sample - len(pos_idx), len(neg_idx))
        if len(neg_idx) > num_neg:
            neg_idx = np.random.choice(neg_idx, num_neg, replace=False)

        sample_idx = np.concatenate([pos_idx, neg_idx])
        sample_idx = jt.array(sample_idx, dtype=jt.int32)

        sampled_boxes = all_boxes[sample_idx]

        # 分类目标：背景=0，前景类别从 gt_labels 读取（1-indexed）
        matched_gt_labels = gt_labels[gt_idx[sample_idx].numpy()]
        cls_target = jt.zeros(sample_idx.shape[0], dtype=jt.int32)
        pos_sample_mask = jt.array(labels_np[sample_idx.numpy()] == 1)
        cls_target[pos_sample_mask] = matched_gt_labels[pos_sample_mask].int32() + 1

        # 回归目标（仅正样本）
        matched_gt_boxes = gt_boxes[gt_idx[sample_idx].numpy()]
        reg_target = encode_boxes(sampled_boxes, matched_gt_boxes)

        return sampled_boxes, cls_target, reg_target

    def _roi_align(
        self,
        fpn_feats: List[jt.Var],
        proposals: jt.Var,
        batch_idx: int,
    ) -> jt.Var:
        """对单张图的 proposals 做 ROI Align（按框大小选择 FPN 层级）。

        FPN 层分配规则（Mask R-CNN 论文）：
          k = floor(k0 + log2(sqrt(w*h) / 224))，k0=4

        Args:
            fpn_feats : [P2, P3, P4, P5]，各 [B, 256, Hi, Wi]
            proposals : [N, 4]，单张图的候选框
            batch_idx : 图像索引（取第 batch_idx 张图的特征）

        Returns:
            roi_feats: [N, 256, roi_size, roi_size]
        """
        N = proposals.shape[0]
        roi_feats = jt.zeros((N, fpn_feats[0].shape[1], self.roi_size, self.roi_size))

        w = proposals[:, 2] - proposals[:, 0]
        h = proposals[:, 3] - proposals[:, 1]
        area = jt.sqrt(w * h + 1e-8)

        # 计算每个 proposal 应使用的 FPN 层
        k0 = 4
        levels = jt.clamp(
            (jt.log(area / 224.0) / math.log(2) + k0).int32(),
            min_v=2, max_v=5,
        )

        for level_idx, stride in enumerate(self.featmap_strides):
            level = level_idx + 2  # P2=2, P3=3, P4=4, P5=5
            mask = (levels == level)
            if mask.sum() == 0:
                continue

            level_proposals = proposals[mask]
            feat = fpn_feats[level_idx][batch_idx:batch_idx+1]  # [1, C, H, W]

            # 为 ROI Align 格式化 proposals：[N, 5]，第 0 列为 batch_idx
            rois = jt.concat([
                jt.zeros((level_proposals.shape[0], 1)),
                level_proposals,
            ], dim=1)

            # Jittor ROI Align
            roi_out = jt.nn.roi_align(
                feat, rois,
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=1.0 / stride,
                sampling_ratio=0,
            )  # [N_level, C, roi_size, roi_size]

            roi_feats[mask] = roi_out

        return roi_feats

    def execute(
        self,
        fpn_feats: List[jt.Var],
        proposals_list: List[jt.Var],
        gt_boxes_list: Optional[List[jt.Var]] = None,
        gt_labels_list: Optional[List[jt.Var]] = None,
    ) -> Dict:
        """ROI Head 前向传播。

        Args:
            fpn_feats       : FPN 输出 [P2~P5]，各 [B, 256, Hi, Wi]
            proposals_list  : RPN 输出，每张图的候选框 [List of [N_i, 4]]
            gt_boxes_list   : 训练时提供，每张图的 GT 框 [List of [M_i, 4]]
            gt_labels_list  : 训练时提供，每张图的 GT 标签 [List of [M_i]]（0-indexed）

        Returns:
            训练时：{'rcnn_cls_loss', 'rcnn_reg_loss'}
            推理时：{'detections': [List of {'boxes','scores','labels'}]}
        """
        B = len(proposals_list)
        # 仅使用 P2~P5（不含 P6）做 ROI Align
        roi_fpn_feats = fpn_feats[:4]

        if self.training and gt_boxes_list is not None:
            # ---- 训练模式 ----
            all_cls_loss = jt.zeros(1)
            all_reg_loss = jt.zeros(1)

            for b in range(B):
                proposals = proposals_list[b]
                gt_boxes  = gt_boxes_list[b]
                gt_labels = gt_labels_list[b]

                if gt_boxes.shape[0] == 0 or proposals.shape[0] == 0:
                    continue

                # 分配并采样
                sampled_props, cls_targets, reg_targets = self._assign_and_sample(
                    proposals, gt_boxes, gt_labels
                )

                # ROI Align: [K, 256, 7, 7]
                roi_feats = self._roi_align(roi_fpn_feats, sampled_props, b)

                # FC 头: [K, 1024]
                x = roi_feats.reshape(roi_feats.shape[0], -1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))

                # 分类损失
                cls_logits = self.cls_head(x)             # [K, num_classes+1]
                cls_loss = nn.cross_entropy_loss(cls_logits, cls_targets)

                # 回归损失（仅对正样本）
                pos_mask = (cls_targets > 0)
                if pos_mask.sum() > 0:
                    cls_target_pos = cls_targets[pos_mask] - 1  # 转回 0-indexed
                    reg_logits = self.reg_head(x[pos_mask])     # [P, num_classes*4]
                    reg_logits = reg_logits.reshape(-1, self.num_classes, 4)
                    # 取对应类别的回归输出
                    batch_idx_for_gather = cls_target_pos.reshape(-1, 1, 1).expand(-1, 1, 4)
                    reg_pred = reg_logits.gather(1, batch_idx_for_gather).squeeze(1)
                    reg_loss = nn.smooth_l1_loss(reg_pred, reg_targets[pos_mask])
                    all_reg_loss = all_reg_loss + reg_loss

                all_cls_loss = all_cls_loss + cls_loss

            return {
                'rcnn_cls_loss': all_cls_loss / max(B, 1),
                'rcnn_reg_loss': all_reg_loss / max(B, 1),
            }

        else:
            # ---- 推理模式 ----
            detections = []

            for b in range(B):
                proposals = proposals_list[b]
                if proposals.shape[0] == 0:
                    detections.append({'boxes': jt.zeros((0, 4)), 'scores': jt.zeros(0), 'labels': jt.zeros(0, dtype=jt.int32)})
                    continue

                # ROI Align
                roi_feats = self._roi_align(roi_fpn_feats, proposals, b)

                # FC 头
                x = roi_feats.reshape(roi_feats.shape[0], -1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))

                # 分类：softmax → [N, num_classes+1]
                cls_scores = nn.softmax(self.cls_head(x), dim=1)
                # 回归：[N, num_classes*4]
                reg_deltas = self.reg_head(x)

                # 对每个前景类别做 NMS
                all_boxes, all_scores, all_labels = [], [], []

                for cls_id in range(self.num_classes):
                    # 该类别的分数（类别 0 为背景，前景从 1 开始）
                    scores = cls_scores[:, cls_id + 1]

                    # 过滤低置信度
                    keep = scores > self.score_thr
                    if keep.sum() == 0:
                        continue

                    scores = scores[keep]
                    props  = proposals[keep]
                    deltas = reg_deltas[keep, cls_id * 4: cls_id * 4 + 4]

                    # 解码位置
                    boxes = decode_boxes(props, deltas)

                    # NMS
                    nms_input = jt.concat([boxes, scores.reshape(-1, 1)], dim=1)
                    keep_idx  = jt.nms(nms_input, self.nms_iou_thr)
                    boxes  = boxes[keep_idx]
                    scores = scores[keep_idx]

                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_labels.append(jt.full((boxes.shape[0],), cls_id, dtype=jt.int32))

                if all_boxes:
                    detections.append({
                        'boxes':  jt.concat(all_boxes, dim=0),
                        'scores': jt.concat(all_scores, dim=0),
                        'labels': jt.concat(all_labels, dim=0),
                    })
                else:
                    detections.append({
                        'boxes': jt.zeros((0, 4)),
                        'scores': jt.zeros(0),
                        'labels': jt.zeros(0, dtype=jt.int32),
                    })

            return {'detections': detections}


# ============================================================
# 完整检测器
# ============================================================

class FasterRCNNCBAM(nn.Module):
    """Faster R-CNN + CBAM 完整检测器（Jittor 版）

    串联：ResNet50CBAM → FPN → RPN → ROIHead

    模块串联关系（训练时）：
      image → Backbone → [C2,C3,C4,C5]
            → FPN → [P2,P3,P4,P5,P6]
            → RPN(loss) → proposals
            → ROIHead(loss) → {'rcnn_cls_loss', 'rcnn_reg_loss'}
      total_loss = rpn_cls + rpn_reg + rcnn_cls + rcnn_reg

    Args:
        num_classes      (int):  前景类别数（VOC=20），默认 20
        cbam_reduction   (int):  CBAM 压缩比，默认 16
        cbam_kernel_size (int):  CBAM 空间卷积核，默认 7
        use_channel_attn (bool): 是否启用通道注意力
        use_spatial_attn (bool): 是否启用空间注意力
        pretrained       (str):  预训练权重路径（可选）
    """

    def __init__(
        self,
        num_classes: int = 20,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
        pretrained: Optional[str] = None,
    ):
        super().__init__()

        # ---- Backbone：ResNet50 + CBAM ----
        self.backbone = ResNet50CBAM(
            cbam_reduction=cbam_reduction,
            cbam_kernel_size=cbam_kernel_size,
            use_channel_attn=use_channel_attn,
            use_spatial_attn=use_spatial_attn,
            frozen_stages=1,
        )

        # ---- Neck：FPN（P2~P6）----
        self.neck = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
        )

        # ---- RPN ----
        self.rpn = RPN(
            in_channels=256,
            num_anchors=3,
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            num_sample=256,
            pos_fraction=0.5,
            nms_pre=2000,
            max_proposals=1000,
        )

        # ---- ROI Head ----
        self.roi_head = ROIHead(
            in_channels=256,
            roi_size=7,
            fc_out_channels=1024,
            num_classes=num_classes,
        )

        # 加载预训练权重
        if pretrained is not None:
            self.backbone.load_pretrained(pretrained)

    def execute(
        self,
        images: jt.Var,
        gt_boxes_list: Optional[List[jt.Var]] = None,
        gt_labels_list: Optional[List[jt.Var]] = None,
    ) -> Dict:
        """前向传播

        Args:
            images         : [B, 3, H, W]，已归一化
            gt_boxes_list  : 训练时提供，每张图的 GT 框 [List of Var [M,4]]
            gt_labels_list : 训练时提供，每张图的 GT 标签 [List of Var [M]]（0-indexed）

        Returns:
            训练时：{'loss', 'rpn_cls_loss', 'rpn_reg_loss', 'rcnn_cls_loss', 'rcnn_reg_loss'}
            推理时：{'detections': [List of dict]}
        """
        B, _, H, W = images.shape
        img_size = (H, W)

        # 1. Backbone: [B,3,H,W] → [C2, C3, C4, C5]
        backbone_feats = self.backbone(images)

        # 2. FPN: [C2,C3,C4,C5] → [P2, P3, P4, P5, P6]
        fpn_feats = self.neck(backbone_feats)

        # 3. RPN: → proposals（推理），或 proposals + loss（训练）
        rpn_out = self.rpn(
            fpn_feats, img_size,
            gt_boxes_list if self.training else None,
        )
        proposals_list = rpn_out['proposals']

        # 4. ROI Head: proposals → 检测结果（推理）或 loss（训练）
        roi_out = self.roi_head(
            fpn_feats[:4],  # P2~P5（P6 不用于 ROI）
            proposals_list,
            gt_boxes_list if self.training else None,
            gt_labels_list if self.training else None,
        )

        if self.training:
            # 汇总所有损失
            rpn_cls = rpn_out.get('rpn_cls_loss', jt.zeros(1))
            rpn_reg = rpn_out.get('rpn_reg_loss', jt.zeros(1))
            rcnn_cls = roi_out.get('rcnn_cls_loss', jt.zeros(1))
            rcnn_reg = roi_out.get('rcnn_reg_loss', jt.zeros(1))
            total_loss = rpn_cls + rpn_reg + rcnn_cls + rcnn_reg
            return {
                'loss':          total_loss,
                'rpn_cls_loss':  rpn_cls,
                'rpn_reg_loss':  rpn_reg,
                'rcnn_cls_loss': rcnn_cls,
                'rcnn_reg_loss': rcnn_reg,
            }
        else:
            return roi_out
