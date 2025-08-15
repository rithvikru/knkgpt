#!/bin/bash
# Launch script for distributed training on 8xB200 GPUs

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it with:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
    echo "Installing dependencies..."
    uv pip install -r requirements.txt
else
    echo "Using existing virtual environment..."
fi

# Activate virtual environment if not already activated
if [[ "$VIRTUAL_ENV" != *".venv"* ]]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Set environment variables for better performance
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0

# Path to the data file
DATA_PATH="data/n_2.jsonl"

# Training hyperparameters optimized for 8xB200 GPUs
BATCH_SIZE=1024  # Per GPU batch size (8k total)
MAX_EPOCHS=5     # Adjust based on convergence
LEARNING_RATE=6e-4
N_LAYER=8       # Larger model for 100M dataset
N_HEAD=8
N_EMBD=512

# Wandb configuration
WANDB_PROJECT="knights-knaves-gpt"
WANDB_RUN_NAME=""  # Leave empty for auto-generated name
# Set WANDB_DISABLED=1 to disable wandb logging
WANDB_DISABLED=${WANDB_DISABLED:-0}

# For testing with smaller dataset first
# Add --max-games 1000000 to use only 1M puzzles for initial testing

echo "Starting distributed training on 8 GPUs..."
echo "Total batch size: $((BATCH_SIZE * 8))"
echo "Dataset: ${DATA_PATH}"

# Prepare wandb arguments
WANDB_ARGS=""
if [ "${WANDB_DISABLED}" = "1" ]; then
    WANDB_ARGS="--wandb-disabled"
    echo "Wandb logging is disabled"
else
    WANDB_ARGS="--wandb-project ${WANDB_PROJECT}"
    if [ -n "${WANDB_RUN_NAME}" ]; then
        WANDB_ARGS="${WANDB_ARGS} --wandb-run-name ${WANDB_RUN_NAME}"
    fi
    echo "Wandb project: ${WANDB_PROJECT}"
fi

# Launch with torchrun for distributed training
# Using uv run ensures the correct environment is used
uv run torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_gpt_knights_knaves.py \
    --data-path "${DATA_PATH}" \
    --batch-size ${BATCH_SIZE} \
    --max-epochs ${MAX_EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --n-layer ${N_LAYER} \
    --n-head ${N_HEAD} \
    --n-embd ${N_EMBD} \
    --checkpoint-dir ./ckpts \
    --validate-every 1000 \
    ${WANDB_ARGS}

echo "Training completed!"
