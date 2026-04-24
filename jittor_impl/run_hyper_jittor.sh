#!/usr/bin/env bash
# ============================================================
# Jittor 版超参数实验一键运行脚本（5 组）
#
# 实验设计：在 Baseline Faster R-CNN（无 CBAM）基础上，
# 验证学习率和批量大小的影响，与 PyTorch 版 run_hyper.sh 对应。
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
# 注意：超参数实验使用 Baseline（无 CBAM），通过 --no-channel-attn
#       --no-spatial-attn 禁用注意力模块，与 PyTorch 版保持一致。
#
# 用法（在项目根目录 faster_rcnn_cbam/ 下运行）：
#   bash jittor_impl/run_hyper_jittor.sh
# ============================================================

set -e

SEED=42
PYTHON="python"
TRAIN="jittor_impl/train_jittor.py"
RESULTS_DIR="experiments/results"

# Baseline 标志：禁用 CBAM 全部注意力模块
BASELINE_FLAGS="--no-channel-attn --no-spatial-attn"

echo "========================================================"
echo "  Jittor 超参数实验开始（共 5 组，Baseline 模式）"
echo "  结果目录: ${RESULTS_DIR}/"
echo "========================================================"

# ============================================================
# 实验 1/5：lr=0.005，bs=4
# ============================================================
EXP_NAME="jittor_hyper_lr0005_bs4"
echo ""
echo "[1/5] lr=0.005, batch_size=4..."
${PYTHON} ${TRAIN} \
    --data-root data/VOCdevkit \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --epochs 12 \
    --lr 0.005 \
    --batch-size 4 \
    --seed ${SEED} \
    ${BASELINE_FLAGS}

# ============================================================
# 实验 2/5：lr=0.01，bs=4（默认）
# ============================================================
EXP_NAME="jittor_hyper_lr001_bs4"
echo ""
echo "[2/5] lr=0.01（默认）, batch_size=4..."
${PYTHON} ${TRAIN} \
    --data-root data/VOCdevkit \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --epochs 12 \
    --lr 0.01 \
    --batch-size 4 \
    --seed ${SEED} \
    ${BASELINE_FLAGS}

# ============================================================
# 实验 3/5：lr=0.02，bs=4
# ============================================================
EXP_NAME="jittor_hyper_lr002_bs4"
echo ""
echo "[3/5] lr=0.02, batch_size=4..."
${PYTHON} ${TRAIN} \
    --data-root data/VOCdevkit \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --epochs 12 \
    --lr 0.02 \
    --batch-size 4 \
    --seed ${SEED} \
    ${BASELINE_FLAGS}

# ============================================================
# 实验 4/5：bs=2，lr=0.005（线性缩放）
# ============================================================
EXP_NAME="jittor_hyper_lr0005_bs2"
echo ""
echo "[4/5] lr=0.005, batch_size=2（线性缩放）..."
${PYTHON} ${TRAIN} \
    --data-root data/VOCdevkit \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --epochs 12 \
    --lr 0.005 \
    --batch-size 2 \
    --seed ${SEED} \
    ${BASELINE_FLAGS}

# ============================================================
# 实验 5/5：bs=8，lr=0.02（线性缩放）
# ============================================================
EXP_NAME="jittor_hyper_lr002_bs8"
echo ""
echo "[5/5] lr=0.02, batch_size=8（线性缩放）..."
${PYTHON} ${TRAIN} \
    --data-root data/VOCdevkit \
    --work-dir "${RESULTS_DIR}/${EXP_NAME}" \
    --epochs 12 \
    --lr 0.02 \
    --batch-size 8 \
    --seed ${SEED} \
    ${BASELINE_FLAGS}

echo ""
echo "========================================================"
echo "  全部 Jittor 超参数实验完成！"
echo "  结果保存在 ${RESULTS_DIR}/jittor_hyper_*/"
echo "========================================================"
