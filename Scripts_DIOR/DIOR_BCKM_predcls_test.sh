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

WEIGHTS="/gz-data/SGG-ToolKit/Checkpoints/STAR-Ship/predcls/DIOR_BCKMv2_predcls_train_seed9891_20260301_175244_nobias/16000.pth"
MMCONFIG="configs/RSOBB/DIOR_obb_predcls_sgcls_swinl_800.py"

if [ ! -f "$WEIGHTS" ]; then
  echo "Error: trained checkpoint not found: $WEIGHTS"
  exit 1
fi

if [ ! -d "glove" ] || [ -z "$(ls -A glove)" ]; then
  echo "Warning: glove directory seems empty or missing."
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GUP=1
export SEED="${SEED:-1029}"

MODEL_NAME='DIOR_BCKM_predcls_test'
path="./Checkpoints/${MODEL_NAME}/"
mkdir -p "$path"

"${PYTHON_CMD[@]}" -u \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_trans_custom_rpcm.yaml" \
  --mm_config "$MMCONFIG" \
  --mm_weight "$WEIGHTS" \
  SEED $SEED \
  DATASETS.TEST "('DIOR_with_attribute_test',)" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR BCKM \
  DTYPE "float32" \
  GLOVE_DIR glove \
  SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GUP \
  OUTPUT_DIR "$path" \
  Type "Large_RS_OBB" \
  filter_method "PPG" \
  INFERENCE.COMPRESS_OUTPUT True \
  INFERENCE.COMPRESS_LEVEL 1 \
  INFERENCE.COMPRESS_MIN_SIZE_MB 100 \
  Only_test True \
  test_outpath "$path" \
  2>&1 | tee -a "${path}/console.log"

