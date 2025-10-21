rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*400)))
test_freq=10
staleness_threshold=0
trigger_parameter_sync_step=16
partial_rollout=False
use_dynamic_bsz=True

NNODES_TRAIN=1
NNODES_ROLLOUT=1
NGPUS_PER_NODE=1


export CUDA_VISIBLE_DEVICES=3,4
python -m recipe.fully_async_policy.fully_async_main \
    data.train_batch_size=${train_prompt_bsz} \
    data.train_files="/data2/Users/aghyad/data/rlhf/gsm8k/train.parquet" \
    data.val_files="/data2/Users/aghyad/data/rlhf/gsm8k/test.parquet" \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    critic.enable=false \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6b \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=50288 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=16 \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.partial_rollout="${partial_rollout}" \
    algorithm.adv_estimator=grpo
