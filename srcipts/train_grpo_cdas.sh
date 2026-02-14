#!/bin/bash

# ====================================================================================
# Environment Configuration
# ====================================================================================
export PROJECT_NAME=${PROJECT_NAME:-"verl_ppo_training"}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-"XFORMERS"}

# Path Configuration - Users should modify these paths according to their setup
export DATA_PATH=${DATA_PATH:-"./data"}
export MODEL_PATH=${MODEL_PATH:-"./models"}
export CHECKPOINT_PATH=${CHECKPOINT_PATH:-"./checkpoints"}
export LOG_PATH=${LOG_PATH:-"./logs"}

# ====================================================================================
# Training Hyperparameters with Default Values
# ====================================================================================

# Data Configuration
TRAIN_BATCH_SIZE=1024
VAL_BATCH_SIZE=500
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=4096

# Model Configuration
MODEL_NAME="Qwen2.5-7B"
DATASET_NAME="math_dataset"

# Optimization Configuration
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=256
PPO_MICRO_BATCH_SIZE=4  # per GPU

# Loss Configuration
CLIP_RATIO=0.2
KL_LOSS_COEF=0.001
ENTROPY_COEF=0.001
KL_LOSS_TYPE="low_var_kl"
KL_COEF=0.001

# Rollout Configuration
TEMPERATURE=1.0
ROLLOUT_N=8
LOG_PROB_MICRO_BATCH_SIZE=160
ROLLOUT_GPU_MEMORY_UTIL=0.8

# Training Configuration
TOTAL_EPOCHS=500
SAVE_FREQ=5
TEST_FREQ=5

# Reward Configuration
REWARD_FUNCTION_TYPE="mix"
FORMAT_PENALTY_VALUE=-1.0

# Distributed Training Configuration
N_GPUS_PER_NODE=8
N_NODES=4

# CDAS Configuration
CDAS_ENABLE=true
CDAS_WARMUP_STEPS=1
CDAS_SKIP_WARMUP=false
CDAS_EXTRA_SAMPLES=0
CDAS_SAVE_VERSIONED=false

# ====================================================================================
# Argument Parsing
# ====================================================================================

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Training Configuration:
  --train_batch_size NUM          Training batch size (default: $TRAIN_BATCH_SIZE)
  --val_batch_size NUM            Validation batch size (default: $VAL_BATCH_SIZE)
  --max_prompt_length NUM         Maximum prompt length (default: $MAX_PROMPT_LENGTH)
  --max_response_length NUM       Maximum response length (default: $MAX_RESPONSE_LENGTH)
  --learning_rate NUM             Learning rate (default: $LEARNING_RATE)
  --ppo_mini_batch_size NUM       PPO mini batch size (default: $PPO_MINI_BATCH_SIZE)
  --ppo_micro_batch_size NUM      PPO micro batch size per GPU (default: $PPO_MICRO_BATCH_SIZE)
  --clip_ratio NUM                PPO clip ratio (default: $CLIP_RATIO)
  --kl_loss_coef NUM              KL loss coefficient (default: $KL_LOSS_COEF)
  --entropy_coef NUM              Entropy coefficient (default: $ENTROPY_COEF)
  --kl_loss_type TYPE             KL loss type (default: $KL_LOSS_TYPE)
  --temperature NUM               Sampling temperature (default: $TEMPERATURE)
  --rollout_n NUM                 Number of rollouts per prompt (default: $ROLLOUT_N)
  --kl_coef NUM                   KL controller coefficient (default: $KL_COEF)
  --total_epochs NUM              Total training epochs (default: $TOTAL_EPOCHS)
  --save_freq NUM                 Checkpoint save frequency (default: $SAVE_FREQ)
  --test_freq NUM                 Test frequency (default: $TEST_FREQ)

Model & Data:
  --model_name NAME               Model name (default: $MODEL_NAME)
  --dataset_name NAME             Dataset name (default: $DATASET_NAME)

CDAS Configuration:
  --cdas_enable BOOL              Enable CDAS system (default: $CDAS_ENABLE)
  --cdas_warmup_steps NUM         CDAS warmup steps (default: $CDAS_WARMUP_STEPS)
  --cdas_skip_warmup BOOL         Skip CDAS warmup (default: $CDAS_SKIP_WARMUP)
  --cdas_extra_samples NUM        Extra samples for CDAS (default: $CDAS_EXTRA_SAMPLES)
  --cdas_save_versioned BOOL      Save versioned CDAS checkpoints (default: $CDAS_SAVE_VERSIONED)

Distributed Training:
  --n_gpus_per_node NUM           Number of GPUs per node (default: $N_GPUS_PER_NODE)
  --n_nodes NUM                   Number of nodes (default: $N_NODES)

Other:
  --run_name NAME                 Experiment run name (auto-generated if not specified)
  --suffix TEXT                   Additional suffix for run name
  --help                          Display this help message

Example:
  $0 --train_batch_size 512 --learning_rate 1e-6 --cdas_enable true
EOF
}

# Parse command line arguments
RUN_NAME=""
SUFFIX=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --help) print_usage; exit 0 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --val_batch_size) VAL_BATCH_SIZE="$2"; shift 2 ;;
        --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
        --ppo_micro_batch_size) PPO_MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
        --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
        --entropy_coef) ENTROPY_COEF="$2"; shift 2 ;;
        --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
        --log_prob_micro_batch_size) LOG_PROB_MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
        --kl_coef) KL_COEF="$2"; shift 2 ;;
        --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        --save_freq) SAVE_FREQ="$2"; shift 2 ;;
        --test_freq) TEST_FREQ="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --dataset_name) DATASET_NAME="$2"; shift 2 ;;
        --reward_function_type) REWARD_FUNCTION_TYPE="$2"; shift 2 ;;
        --format_penalty_value) FORMAT_PENALTY_VALUE="$2"; shift 2 ;;
        --cdas_enable) CDAS_ENABLE="$2"; shift 2 ;;
        --cdas_warmup_steps) CDAS_WARMUP_STEPS="$2"; shift 2 ;;
        --cdas_skip_warmup) CDAS_SKIP_WARMUP="$2"; shift 2 ;;
        --cdas_extra_samples) CDAS_EXTRA_SAMPLES="$2"; shift 2 ;;
        --cdas_save_versioned) CDAS_SAVE_VERSIONED="$2"; shift 2 ;;
        --n_gpus_per_node) N_GPUS_PER_NODE="$2"; shift 2 ;;
        --n_nodes) N_NODES="$2"; shift 2 ;;
        --run_name) RUN_NAME="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# ====================================================================================
# Generate Run Name
# ====================================================================================

if [ -z "$RUN_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="ppo_${MODEL_NAME}_${DATASET_NAME}"
    
    if [ "$CDAS_ENABLE" = "true" ]; then
        RUN_NAME="${RUN_NAME}_cdas"
    fi
    
    RUN_NAME="${RUN_NAME}_bs${TRAIN_BATCH_SIZE}_lr${LEARNING_RATE}"
    RUN_NAME="${RUN_NAME}_rollout${ROLLOUT_N}_temp${TEMPERATURE}"
    
    if [ -n "$SUFFIX" ]; then
        RUN_NAME="${RUN_NAME}_${SUFFIX}"
    fi
    
    RUN_NAME="${RUN_NAME}_${TIMESTAMP}"
fi

# Setup paths
CHECKPOINT_DIR="${CHECKPOINT_PATH}/${RUN_NAME}"
LOG_FILE="${LOG_PATH}/${RUN_NAME}.log"

# Create directories
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$CHECKPOINT_DIR"

# ====================================================================================
# Print Configuration
# ====================================================================================

cat << EOF | tee -a "$LOG_FILE"
====================================================================================
Training Configuration
====================================================================================
Run Name: $RUN_NAME
Checkpoint Directory: $CHECKPOINT_DIR
Log File: $LOG_FILE

Model & Data:
  Model: $MODEL_NAME
  Dataset: $DATASET_NAME
  Train Batch Size: $TRAIN_BATCH_SIZE
  Val Batch Size: $VAL_BATCH_SIZE
  Max Prompt Length: $MAX_PROMPT_LENGTH
  Max Response Length: $MAX_RESPONSE_LENGTH

Optimization:
  Learning Rate: $LEARNING_RATE
  PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE
  PPO Micro Batch Size (per GPU): $PPO_MICRO_BATCH_SIZE
  Clip Ratio: $CLIP_RATIO
  KL Loss Coefficient: $KL_LOSS_COEF
  Entropy Coefficient: $ENTROPY_COEF
  KL Loss Type: $KL_LOSS_TYPE

Rollout:
  Temperature: $TEMPERATURE
  Rollout N: $ROLLOUT_N
  Log Prob Micro Batch Size: $LOG_PROB_MICRO_BATCH_SIZE
  GPU Memory Utilization: $ROLLOUT_GPU_MEMORY_UTIL

Training:
  Total Epochs: $TOTAL_EPOCHS
  Save Frequency: $SAVE_FREQ
  Test Frequency: $TEST_FREQ

CDAS Configuration:
  Enable: $CDAS_ENABLE
  Warmup Steps: $CDAS_WARMUP_STEPS
  Skip Warmup: $CDAS_SKIP_WARMUP
  Extra Samples: $CDAS_EXTRA_SAMPLES
  Save Versioned Checkpoints: $CDAS_SAVE_VERSIONED

Distributed:
  GPUs per Node: $N_GPUS_PER_NODE
  Number of Nodes: $N_NODES
====================================================================================
EOF

# ====================================================================================
# Calculate Dynamic Parameters
# ====================================================================================

MAX_NUM_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
TOTAL_TRAINING_STEPS=$((TOTAL_EPOCHS * 10))  # Approximate, will be overridden by actual data

# Export reward function configuration
export REWARD_FUNCTION_TYPE=$REWARD_FUNCTION_TYPE
export FORMAT_PENALTY_VALUE=$FORMAT_PENALTY_VALUE

# ====================================================================================
# Start Training
# ====================================================================================

echo "Starting training..." | tee -a "$LOG_FILE"
echo "Command: python -m verl.trainer.main_ppo [with hydra overrides]" | tee -a "$LOG_FILE"
echo "===================================================================================="

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_PATH}/${DATASET_NAME}/train.parquet \
    data.val_files=${DATA_PATH}/${DATASET_NAME}/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=${MODEL_PATH}/${MODEL_NAME} \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEF \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    critic.ppo_micro_batch_size_per_gpu=4 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    trainer.remove_clip=False \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.total_epochs=$TOTAL_EPOCHS \
    cdas.enable=$CDAS_ENABLE \
    cdas.warmup_steps=$CDAS_WARMUP_STEPS \
    cdas.skip_warmup=$CDAS_SKIP_WARMUP \
    cdas.extra_samples=$CDAS_EXTRA_SAMPLES \
    cdas.save_versioned_checkpoints=$CDAS_SAVE_VERSIONED \
    2>&1 | tee -a "$LOG_FILE"

# ====================================================================================
# Training Complete
# ====================================================================================

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!" | tee -a "$LOG_FILE"
else
    echo "Training failed with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
fi

echo "Log file saved to: $LOG_FILE"
echo "Checkpoints saved to: $CHECKPOINT_DIR"

exit $EXIT_CODE