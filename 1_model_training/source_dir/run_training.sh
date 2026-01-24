#!/bin/bash
set -e

MODEL_DIR=${SM_HP_MODEL_DIR}
PIPELINE_CONFIG_PATH=${SM_HP_PIPELINE_CONFIG_PATH}
NUM_TRAIN_STEPS=${SM_HP_NUM_TRAIN_STEPS}
SAMPLE_1_OF_N_EVAL_EXAMPLES=${SM_HP_SAMPLE_1_OF_N_EVAL_EXAMPLES}

if [ "${SM_NUM_GPUS}" -gt 0 ]; then
  NUM_WORKERS=${SM_NUM_GPUS}
else
  NUM_WORKERS=1
fi

echo "=== TRAINING THE MODEL ==="
python model_main_tf2.py \
  --pipeline_config_path "${PIPELINE_CONFIG_PATH}" \
  --model_dir "${MODEL_DIR}" \
  --num_train_steps "${NUM_TRAIN_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --sample_1_of_n_eval_examples "${SAMPLE_1_OF_N_EVAL_EXAMPLES}" \
  --alsologtostderr

CHECKPOINT_DIR="${MODEL_DIR}"
EXPORT_DIR="/opt/ml/model"

echo "=== CHECKING CHECKPOINTS ==="
if ! ls ${CHECKPOINT_DIR}/ckpt-*.index 1> /dev/null 2>&1; then
  echo "ERROR: No checkpoints found in ${CHECKPOINT_DIR}"
  exit 1
fi

echo "=== EXPORTING THE MODEL ==="
python exporter_main_v2.py \
  --input_type image_tensor \
  --pipeline_config_path "${PIPELINE_CONFIG_PATH}" \
  --trained_checkpoint_dir "${MODEL_DIR}" \
  --output_directory "/opt/ml/model"

echo "=== EXPORT COMPLETE ==="
