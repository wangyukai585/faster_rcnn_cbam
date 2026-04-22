#!/usr/bin/env bash
# ============================================================
# 一键运行所有消融实验（6组）
#
# 消融实验设计：从无到有，逐步添加注意力组件，验证每个组件的贡献
#   1. 无CBAM（Baseline）
#   2. 仅通道注意力（CA Only）
#   3. 仅空间注意力（SA Only）
#   4. 完整CBAM（CA + SA，reduction=16，kernel_size=7）
#   5. CBAM（reduction=8，更大MLP容量）
#   6. CBAM（kernel_size=3，更小感受野）
#
# 用法（在项目根目录 faster_rcnn_cbam/ 下运行）：
#   bash pytorch_mmdet/tools/run_ablation.sh
#   bash pytorch_mmdet/tools/run_ablation.sh --gpu 0 1  # 多卡
# ============================================================

set -e  # 任何命令失败则立即退出

# ---- 解析参数 ----
GPU_IDS="0"
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) shift; GPU_IDS="$@"; break ;;
        --seed) SEED="$2"; shift 2 ;;
        *) shift ;;
    esac
done

PYTHON="python"
TRAIN="pytorch_mmdet/tools/train.py"
EVAL="pytorch_mmdet/tools/evaluate.py"
RESULTS_DIR="experiments/results"

echo "========================================================"
echo "  消融实验开始"
echo "  结果目录: ${RESULTS_DIR}/"
echo "  随机种子: ${SEED}"
echo "========================================================"

# ============================================================
# 实验 1/6：Baseline（无注意力）
# ============================================================
EXP_NAME="ablation_1_baseline"
echo ""
echo "[1/6] 训练 Baseline（无CBAM）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/base_faster_rcnn.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

echo "[1/6] 评估 Baseline..."
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/base_faster_rcnn.py \
    "${RESULTS_DIR}/${EXP_NAME}/best_pascal_voc_mAP.pth" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 实验 2/6：仅通道注意力
# ============================================================
EXP_NAME="ablation_2_channel_only"
echo ""
echo "[2/6] 训练 仅通道注意力（CA Only）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/ablation_channel.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

echo "[2/6] 评估..."
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/ablation_channel.py \
    "${RESULTS_DIR}/${EXP_NAME}/best_pascal_voc_mAP.pth" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 实验 3/6：仅空间注意力
# ============================================================
EXP_NAME="ablation_3_spatial_only"
echo ""
echo "[3/6] 训练 仅空间注意力（SA Only）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/ablation_spatial.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

echo "[3/6] 评估..."
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/ablation_spatial.py \
    "${RESULTS_DIR}/${EXP_NAME}/best_pascal_voc_mAP.pth" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 实验 4/6：完整 CBAM（reduction=16，kernel_size=7，论文默认）
# ============================================================
EXP_NAME="ablation_4_cbam_r16_k7"
echo ""
echo "[4/6] 训练 完整CBAM（r=16, k=7）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/cbam_faster_rcnn.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

echo "[4/6] 评估..."
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/cbam_faster_rcnn.py \
    "${RESULTS_DIR}/${EXP_NAME}/best_pascal_voc_mAP.pth" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 实验 5/6：CBAM reduction=8
# ============================================================
EXP_NAME="ablation_5_cbam_r8_k7"
echo ""
echo "[5/6] 训练 CBAM（r=8, k=7）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/ablation_cbam_r8.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

echo "[5/6] 评估..."
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/ablation_cbam_r8.py \
    "${RESULTS_DIR}/${EXP_NAME}/best_pascal_voc_mAP.pth" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 实验 6/6：CBAM kernel_size=3
# ============================================================
EXP_NAME="ablation_6_cbam_r16_k3"
echo ""
echo "[6/6] 训练 CBAM（r=16, k=3）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/ablation_cbam_k3.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

echo "[6/6] 评估..."
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/ablation_cbam_k3.py \
    "${RESULTS_DIR}/${EXP_NAME}/best_pascal_voc_mAP.pth" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 汇总结果
# ============================================================
echo ""
echo "========================================================"
echo "  所有消融实验完成！正在汇总结果..."
echo "========================================================"
${PYTHON} analyze_results.py --mode ablation

echo ""
echo "  结果表格和图表已保存到 report/figures/"
echo "========================================================"
