#!/bin/bash
set -e
set -o pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_CMD=(python3)
if ! "${PYTHON_CMD[@]}" -c "import torch" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run -n sgg python)
  fi
fi

ENV_PREFIX="$("${PYTHON_CMD[@]}" -c "import sys; print(sys.prefix)")"
TORCH_LIB_DIR="$("${PYTHON_CMD[@]}" -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"

if [ ! -f "Pretrained_Obj/OBB_swin_L_OBD.pth" ] && [ ! -f "Pretrained_Obj/HBB_swin_L_OBD.pth" ]; then
    echo "Error: Pretrained weights not found in Pretrained_Obj/"
    echo "Please download them from https://huggingface.co/Zhuzi24/STAR_OBJ_REL_WEIGHTS"
    echo "and place them in the Pretrained_Obj/ directory."
    exit 1
fi

if [ ! -d "glove" ] || [ -z "$(ls -A glove)" ]; then
    echo "Warning: glove directory seems empty or missing."
    echo "Please download Glove embeddings from https://huggingface.co/Zhuzi24/STAR_OBJ_REL_WEIGHTS"
    echo "and extract them into the glove/ directory."
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GUP=1
MODEL_NAME='LOBB_RPCM_predcls_train'

path="./Checkpoints/${MODEL_NAME}/"
mkdir -p "$path"

WEIGHTS="Pretrained_Obj/OBB_swin_L_OBD.pth"

echo "Starting training..."
echo "Model: $MODEL_NAME"
echo "Output Path: $path"
echo "Weights: $WEIGHTS"

"${PYTHON_CMD[@]}" -u \
 tools/relation_train_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x_trans_base.yaml" \
--mm_config "configs/RSOBB/STAR_obb_predcls_sgcls.py" \
--mm_weight "$WEIGHTS" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
MODEL.ROI_RELATION_HEAD.PREDICTOR RPCM \
SOLVER.WARMUP_ITERS 500 \
DTYPE "float32" \
GLOVE_DIR glove \
SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH $NUM_GUP \
SOLVER.MAX_ITER 10000 SOLVER.BASE_LR 1e-3 \
SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
SOLVER.STEPS "(6000, 8500)" SOLVER.VAL_PERIOD 2000 \
SOLVER.CHECKPOINT_PERIOD 1000  \
OUTPUT_DIR "$path" \
SOLVER.PRE_VAL False \
SOLVER.GRAD_NORM_CLIP 5.0 \
Type "Large_RS_OBB" \
filter_method "PPG" \
 2>&1 | tee -a "${path}/console.log"
