#!/usr/bin/env bash
# ============================================================
# 一键运行所有超参数实验（5组）
#
# 实验设计：在完整 CBAM 模型基础上，分别验证学习率和批量大小的影响。
#
#   实验矩阵：
#   ┌─────────┬────────────┬──────────────┐
#   │ 实验     │ lr         │ batch_size   │
#   ├─────────┼────────────┼──────────────┤
#   │ 1       │ 0.005      │ 4            │
#   │ 2（默认）│ 0.01       │ 4            │
#   │ 3       │ 0.02       │ 4            │
#   │ 4       │ 0.005(缩放)│ 2            │
#   │ 5       │ 0.02(缩放) │ 8            │
#   └─────────┴────────────┴──────────────┘
#
# 用法（在项目根目录 faster_rcnn_cbam/ 下运行）：
#   bash pytorch_mmdet/tools/run_hyper.sh
# ============================================================

set -e

SEED=42
PYTHON="python"
TRAIN="pytorch_mmdet/tools/train.py"
EVAL="pytorch_mmdet/tools/evaluate.py"
RESULTS_DIR="experiments/results"

# mmengine 保存最优权重文件名格式：best_pascal_voc_mAP_epoch_N.pth
find_best_ckpt() {
    ls "$1"/best_pascal_voc_mAP_epoch_*.pth 2>/dev/null | sort -V | tail -1
}

echo "========================================================"
echo "  超参数实验开始（共 5 组）"
echo "  结果目录: ${RESULTS_DIR}/"
echo "========================================================"

# ============================================================
# 实验 1/5：lr=0.005，bs=4
# ============================================================
EXP_NAME="hyper_lr0005_bs4"
echo ""
echo "[1/5] lr=0.005, batch_size=4..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/hyper_lr0005.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

CKPT=$(find_best_ckpt "${RESULTS_DIR}/${EXP_NAME}")
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/hyper_lr0005.py \
    "${CKPT}" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 实验 2/5：lr=0.01，bs=4（默认，若已有结果则跳过训练）
# ============================================================
EXP_NAME="hyper_lr001_bs4"
echo ""
echo "[2/5] lr=0.01（默认）, batch_size=4..."
if [ ! -f "${RESULTS_DIR}/${EXP_NAME}/eval_results.json" ]; then
    ${PYTHON} ${TRAIN} \
        pytorch_mmdet/configs/cbam_faster_rcnn.py \
        --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
        --seed ${SEED}
    CKPT=$(find_best_ckpt "${RESULTS_DIR}/${EXP_NAME}")
    ${PYTHON} ${EVAL} \
        pytorch_mmdet/configs/cbam_faster_rcnn.py \
        "${CKPT}" \
        --out-dir "${RESULTS_DIR}/${EXP_NAME}"
else
    echo "  已存在结果，跳过训练。"
fi

# ============================================================
# 实验 3/5：lr=0.02，bs=4
# ============================================================
EXP_NAME="hyper_lr002_bs4"
echo ""
echo "[3/5] lr=0.02, batch_size=4..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/hyper_lr002.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

CKPT=$(find_best_ckpt "${RESULTS_DIR}/${EXP_NAME}")
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/hyper_lr002.py \
    "${CKPT}" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 实验 4/5：bs=2，lr=0.005（线性缩放）
# ============================================================
EXP_NAME="hyper_lr0005_bs2"
echo ""
echo "[4/5] lr=0.005, batch_size=2（线性缩放）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/hyper_bs2.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

CKPT=$(find_best_ckpt "${RESULTS_DIR}/${EXP_NAME}")
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/hyper_bs2.py \
    "${CKPT}" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 实验 5/5：bs=8，lr=0.02（线性缩放）
# ============================================================
EXP_NAME="hyper_lr002_bs8"
echo ""
echo "[5/5] lr=0.02, batch_size=8（线性缩放）..."
${PYTHON} ${TRAIN} \
    pytorch_mmdet/configs/hyper_bs8.py \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --seed ${SEED}

CKPT=$(find_best_ckpt "${RESULTS_DIR}/${EXP_NAME}")
${PYTHON} ${EVAL} \
    pytorch_mmdet/configs/hyper_bs8.py \
    "${CKPT}" \
    --out-dir "${RESULTS_DIR}/${EXP_NAME}"

# ============================================================
# 汇总结果
# ============================================================
echo ""
echo "========================================================"
echo "  所有超参数实验完成！正在汇总结果..."
echo "========================================================"
${PYTHON} analyze_results.py

echo ""
echo "  结果表格和图表已保存到 report/figures/"
echo "========================================================"
