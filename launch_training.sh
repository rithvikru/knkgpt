#!/bin/bash

# Launch training script for KnightKnaves GPT

# Default configuration
DATA_PATH="./data/n_2.jsonl"
PRETOKENIZED_DIR=""
N_LAYER=8
N_HEAD=8
N_EMBD=512
BATCH_SIZE=64
MAX_EPOCHS=10
LEARNING_RATE=6e-4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --small)
            echo "Using small model configuration"
            N_LAYER=4
            N_HEAD=4
            N_EMBD=256
            BATCH_SIZE=128
            shift
            ;;
        --large)
            echo "Using large model configuration"
            N_LAYER=12
            N_HEAD=12
            N_EMBD=768
            BATCH_SIZE=32
            shift
            ;;
        --debug)
            echo "Debug mode - using subset of data"
            N_PUZZLES=10000
            MAX_EPOCHS=2
            shift
            ;;
        --pretokenized)
            echo "Using pre-tokenized data"
            PRETOKENIZED_DIR="./data/tokenized"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--small|--large] [--debug] [--pretokenized]"
            exit 1
            ;;
    esac
done

# Create checkpoint directory
mkdir -p ./ckpts/knkgpt

# Launch training
echo "Starting KnightKnaves GPT training..."
echo "Model config: ${N_LAYER} layers, ${N_HEAD} heads, ${N_EMBD} embedding dim"
echo "Batch size: ${BATCH_SIZE}"
echo "Max epochs: ${MAX_EPOCHS}"

if [ -n "${PRETOKENIZED_DIR}" ]; then
    echo "Using pre-tokenized data from: ${PRETOKENIZED_DIR}"
    python train_gpt_knights_knaves.py \
        --pretokenized_dir ${PRETOKENIZED_DIR} \
        --n_layer ${N_LAYER} \
        --n_head ${N_HEAD} \
        --n_embd ${N_EMBD} \
        --batch_size ${BATCH_SIZE} \
        --max_epochs ${MAX_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --wandb_project knkgpt
else
    echo "Using raw data from: ${DATA_PATH}"
    python train_gpt_knights_knaves.py \
        --data_path ${DATA_PATH} \
        --n_layer ${N_LAYER} \
        --n_head ${N_HEAD} \
        --n_embd ${N_EMBD} \
        --batch_size ${BATCH_SIZE} \
        --max_epochs ${MAX_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --wandb_project knkgpt \
        ${N_PUZZLES:+--n_puzzles ${N_PUZZLES}}
fi
