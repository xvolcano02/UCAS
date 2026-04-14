set -x

export WANDB_API_KEY=YOUR_WANDB_API_KEY
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1

PROJECT_NAME="ucas_math"
EXPERIMENT_NAME="ucas_qwen_math_1.5b"
CHECKPOINT_DIR="/path/to/your/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

mkdir -p "${CHECKPOINT_DIR}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ucas \
    data.train_files=/path/to/your/data/math12k/train.parquet \
    data.val_files=/path/to/your/data/math12k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/path/to/your/model/Qwen2.5-Math-1.5B \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.engine_seed=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.resume_mode='auto' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 2>&1 | tee ucas_qwen_math_1.5b.log
