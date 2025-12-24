#!/bin/bash

export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO
set -a
source ~/.env
set +a

# Get current date and time
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")

rm -rf logs
mkdir logs

python execution/server.py > $WD/reward_seeker/verl/execution/server.log 2>&1 &
pid=$!
# Ensure cmd1 is killed on script exit
trap "kill $pid 2>/dev/null" EXIT

CONFIG_PATH="$WD/verl_with_logging/configs"
#CONFIG_FILE="30a3b.yaml"
CONFIG_FILE="8b.yaml"
LOGGING_DIR=console_logs/${CONFIG_FILE}/${current_date}/${current_time}
echo $LOGGING_DIR
mkdir -p $LOGGING_DIR
LOGGING_PATH=${LOGGING_DIR}/log.log

cp ${CONFIG_PATH}/${CONFIG_FILE} ${LOGGING_DIR}/${CONFIG_FILE}

export HYDRA_FULL_ERROR=1;
export VLLM_USE_V1=1
python -m recipe.fully_async_policy.fully_async_main \
   --config-path  $CONFIG_PATH \
   --config-name $CONFIG_FILE \
   2>&1 | tee $LOGGING_PATH


