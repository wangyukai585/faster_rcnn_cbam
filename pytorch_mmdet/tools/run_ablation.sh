#!/usr/bin/env bash
# ============================================================
# 一键运行所有消融实验（4组）
#
# 消融实验设计：从无到有，逐步添加注意力组件，验证每个组件的贡献
#   1. 无CBAM（Baseline）
#   2. 仅通道注意力（CA Only）
#   3. 仅空间注意力（SA Only）
#   4. 完整CBAM（CA + SA，reduction=16，kernel_size=7）
#
# 用法（在项目根目录 faster_rcnn_cbam/ 下运行）：
#   bash pytorch_mmdet/tools/run_ablation.sh
# ============================================================

set -e

SEED=42
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        *) shift ;;
    esac
done

PYTHON="python"
TRAIN="pytorch_mmdet/tools/train.py"
EVAL="pytorch_mmdet/tools/evaluate.py"
RESULTS_DIR="experiments/results"

# mmengine 保存最优权重文件名格式：best_pascal_voc_mAP_epoch_N.pth
# 此函数找到目录下最新的最优权重文件
find_best_ckpt() {
    ls "$1"/best_pascal_voc_mAP_epoch_*.pth 2>/dev/null | sort -V | tail -1
}

echo "========================================================"
echo "  消融实验开始（共 4 组）"
echo "  结果目录: ${RESULTS_DIR}/"
echo "  随机种子: ${SEED}"
echo "========================================================"

# ============================================================
# 实验 1/4：Baseline（无注意力）
# ============================================================
EXP_NAME="ablation_1_baseline"
echo ""
echo "[1/4] 训练 Baseline（无CBAM）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/base_faster_rcnn.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

echo "[1/4] 评估 Baseline..."
CKPT=$(find_best_ckpt "${RESULTS_DIR}/${EXP_NAME}")
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/base_faster_rcnn.py \
    "${CKPT}" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 实验 2/4：仅通道注意力
# ============================================================
EXP_NAME="ablation_2_channel_only"
echo ""
echo "[2/4] 训练 仅通道注意力（CA Only）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/ablation_channel.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

echo "[2/4] 评估..."
CKPT=$(find_best_ckpt "${RESULTS_DIR}/${EXP_NAME}")
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/ablation_channel.py \
    "${CKPT}" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 实验 3/4：仅空间注意力
# ============================================================
EXP_NAME="ablation_3_spatial_only"
echo ""
echo "[3/4] 训练 仅空间注意力（SA Only）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/ablation_spatial.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

echo "[3/4] 评估..."
CKPT=$(find_best_ckpt "${RESULTS_DIR}/${EXP_NAME}")
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/ablation_spatial.py \
    "${CKPT}" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 实验 4/4：完整 CBAM（reduction=16，kernel_size=7，论文默认）
# ============================================================
EXP_NAME="ablation_4_cbam_r16_k7"
echo ""
echo "[4/4] 训练 完整CBAM（r=16, k=7）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/cbam_faster_rcnn.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

echo "[4/4] 评估..."
CKPT=$(find_best_ckpt "${RESULTS_DIR}/${EXP_NAME}")
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/cbam_faster_rcnn.py \
    "${CKPT}" \
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
