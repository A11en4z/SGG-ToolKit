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

MODEL_NAME='DIOR_RPCM_predcls_train'
path="./Checkpoints/${MODEL_NAME}/"
mkdir -p "$path"

"${PYTHON_CMD[@]}" -u \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_trans_custom_rpcm.yaml" \
  --mm_config "$MMCONFIG" \
  --mm_weight "$PRETRAIN_WEIGHTS" \
  DATASETS.TRAIN "('DIOR_with_attribute_train',)" \
  DATASETS.VAL "('DIOR_with_attribute_val',)" \
  DATASETS.TEST "('DIOR_with_attribute_test',)" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR RPCM \
  MODEL.ROI_BOX_HEAD.NUM_CLASSES 21 \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 23 \
  MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 2 \
  SOLVER.WARMUP_ITERS 500 \
  DTYPE "float32" \
  GLOVE_DIR glove \
  SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER 10000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(6000, 8500)" SOLVER.VAL_PERIOD 2000 \
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

