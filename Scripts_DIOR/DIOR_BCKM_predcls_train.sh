#!/bin/bash
set -e
set -o pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_CMD=(python3)
if ! "${PYTHON_CMD[@]}" -c "import torch, mmcv" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run -n sgg python)
  fi
fi

ENV_PREFIX="$("${PYTHON_CMD[@]}" -c "import sys; print(sys.prefix)")"
TORCH_LIB_DIR="$("${PYTHON_CMD[@]}" -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"

PRETRAIN_WEIGHTS="/gz-data/mmrotate/work_dirs/oriented_rcnn_swin-l_fpn_1x_star_le90/best_mAP_epoch_21.pth"
MMCONFIG="configs/RSOBB/DIOR_obb_predcls_sgcls_swinl_800.py"

if [ ! -f "$PRETRAIN_WEIGHTS" ]; then
  echo "Error: mmrotate checkpoint not found: $PRETRAIN_WEIGHTS"
  exit 1
fi

if [ ! -d "glove" ] || [ -z "$(ls -A glove)" ]; then
  echo "Warning: glove directory seems empty or missing."
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GUP=1
export SEED="${SEED:-42}"

START_TIME="$(date "+%Y%m%d_%H%M%S")"
MODEL_NAME="DIOR_BCKMv2_predcls_train_seed${SEED}_${START_TIME}"
path="./Checkpoints/${MODEL_NAME}/"
mkdir -p "$path"

TB_LOGDIR="${path}/tb"
mkdir -p "$TB_LOGDIR"
TB_HOST="${TB_HOST:-127.0.0.1}"
TB_PORT="${TB_PORT:-$((6006 + (SEED % 1000)))}"
TB_PID=""
if [ "${AUTO_TENSORBOARD:-1}" != "0" ]; then
  if "${PYTHON_CMD[@]}" -c "import tensorboard" >/dev/null 2>&1; then
    nohup "${PYTHON_CMD[@]}" -m tensorboard --logdir "$TB_LOGDIR" --host "$TB_HOST" --port "$TB_PORT" \
      >"${path}/tensorboard.log" 2>&1 &
    TB_PID="$!"
    echo "$TB_PID" > "${path}/tensorboard.pid"
    echo "TensorBoard: http://${TB_HOST}:${TB_PORT}/ (pid=${TB_PID})"
  elif command -v conda >/dev/null 2>&1; then
    nohup conda run -n sgg python -m tensorboard --logdir "$TB_LOGDIR" --host "$TB_HOST" --port "$TB_PORT" \
      >"${path}/tensorboard.log" 2>&1 &
    TB_PID="$!"
    echo "$TB_PID" > "${path}/tensorboard.pid"
    echo "TensorBoard: http://${TB_HOST}:${TB_PORT}/ (pid=${TB_PID})"
  else
    echo "TensorBoard not started: tensorboard module not found."
  fi
fi
cleanup_tensorboard() {
  if [ -n "${TB_PID}" ] && kill -0 "${TB_PID}" >/dev/null 2>&1; then
    kill "${TB_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_tensorboard EXIT

"${PYTHON_CMD[@]}" -u \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_trans_custom_bckm.yaml" \
  --mm_config "$MMCONFIG" \
  --mm_weight "$PRETRAIN_WEIGHTS" \
  SEED $SEED \
  DATASETS.TRAIN "('DIOR_with_attribute_train',)" \
  DATASETS.VAL "('DIOR_with_attribute_val',)" \
  DATASETS.TEST "('DIOR_with_attribute_test',)" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR BCKM \
  MODEL.ROI_BOX_HEAD.NUM_CLASSES 21 \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 23 \
  MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 2 \
  SOLVER.WARMUP_ITERS 500 \
  DTYPE "float32" \
  GLOVE_DIR glove \
  SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER 20000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(11000, 18000)" SOLVER.VAL_PERIOD 1000 \
  SOLVER.CHECKPOINT_PERIOD 1000 \
  val_outpath "$path/inference/val" \
  test_outpath "$path/inference/test" \
  OUTPUT_DIR "$path" \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  AUTO_LONGTAIL_IDS True \
  Type "Large_RS_OBB" \
  filter_method "PPG" \
  INFERENCE.COMPRESS_OUTPUT True \
  INFERENCE.COMPRESS_LEVEL 1 \
  INFERENCE.COMPRESS_MIN_SIZE_MB 100 \
  2>&1 | tee -a "${path}/console.log"
