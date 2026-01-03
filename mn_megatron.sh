#!/bin/bash
# Launch script for 80a3b Megatron backend training

set -x

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# NCCL settings for multi-node/multi-GPU training
export NCCL_TIMEOUT=1800  # 30 min for large MoE models
export NCCL_DEBUG=INFO

# H100 Performance Optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=1     # Megatron communication/computation overlap
export VLLM_USE_V1=1                     # vLLM v1 engine
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # Reduce memory fragmentation
export NCCL_NVLS_ENABLE=0                # Stability
export NVTE_FRAMEWORK=pytorch            # TransformerEngine
export NVTE_FLASH_ATTN=1                 # Flash attention in TE
export HYDRA_FULL_ERROR=1                # Full error traces

# ============================================================================
# LOGGING SETUP
# ============================================================================

current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")

rm -rf logs
mkdir logs

# Start execution server in background
python execution/server.py > $WD/reward_seeker/verl/execution/server.log 2>&1 &

# ============================================================================
# CONFIG PATHS
# ============================================================================

CONFIG_PATH="/workspace/verl_with_logging/configs"
CONFIG_FILE="80a3b_megatron.yaml"
LOGGING_DIR=/data/console_logs/${CONFIG_FILE}/${current_date}/${current_time}
echo "Logging to: $LOGGING_DIR"
mkdir -p $LOGGING_DIR
LOGGING_PATH=${LOGGING_DIR}/log.log

# Copy config to logging directory for reproducibility
cp ${CONFIG_PATH}/${CONFIG_FILE} ${LOGGING_DIR}/${CONFIG_FILE}

# ============================================================================
# RAY JOB SUBMISSION
# ============================================================================

RAY_ADDRESS="http://localhost:8265"
RUNTIME_ENV="${CONFIG_PATH}/runtime_env.yaml"
WORKING_DIR="/workspace/verl_with_logging"

ray job submit --address="${RAY_ADDRESS}" \
    --runtime-env="${RUNTIME_ENV}" \
    --working-dir="${WORKING_DIR}" \
    --no-wait \
    -- \
    python -m recipe.fully_async_policy.fully_async_main \
       --config-path $CONFIG_PATH \
       --config-name $CONFIG_FILE \
       2>&1 | tee $LOGGING_PATH

