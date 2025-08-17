#!/bin/bash
# Script to pre-tokenize dataset and launch training with optimized data loading

set -e  # Exit on error

# Configuration
DATA_PATH="${DATA_PATH:-./data/n_2.jsonl}"
TOKENIZED_DIR="${TOKENIZED_DIR:-./data/tokenized}"
N_PUZZLES="${N_PUZZLES:-}"  # Empty means use all puzzles

echo "Knights and Knaves GPT - Pre-tokenization and Training Pipeline"
echo "=============================================================="

# Step 1: Check if pre-tokenized data exists
if [ ! -d "$TOKENIZED_DIR" ]; then
    echo "Pre-tokenized data not found at $TOKENIZED_DIR"
    echo "Running pre-tokenization..."
    
    # Create tokenized data directory
    mkdir -p "$TOKENIZED_DIR"
    
    # Run pre-tokenization
    if [ -n "$N_PUZZLES" ]; then
        echo "Pre-tokenizing first $N_PUZZLES puzzles..."
        python pretokenize_dataset.py \
            --input "$DATA_PATH" \
            --output "$TOKENIZED_DIR" \
            --n_puzzles "$N_PUZZLES"
    else
        echo "Pre-tokenizing all puzzles (this may take a while for 100M puzzles)..."
        python pretokenize_dataset.py \
            --input "$DATA_PATH" \
            --output "$TOKENIZED_DIR"
    fi
    
    echo "Pre-tokenization complete!"
else
    echo "Found pre-tokenized data at $TOKENIZED_DIR"
fi

# Step 2: Launch training with pre-tokenized data
echo ""
echo "Starting training with pre-tokenized data..."
echo ""

# Default training configuration
N_LAYER="${N_LAYER:-8}"
N_HEAD="${N_HEAD:-8}"
N_EMBD="${N_EMBD:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "Model configuration:"
echo "  Layers: $N_LAYER"
echo "  Heads: $N_HEAD"
echo "  Embedding dim: $N_EMBD"
echo "  Batch size: $BATCH_SIZE"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Data workers: $NUM_WORKERS"
echo ""

python train_gpt_knights_knaves.py \
    --pretokenized_dir "$TOKENIZED_DIR" \
    --n_layer "$N_LAYER" \
    --n_head "$N_HEAD" \
    --n_embd "$N_EMBD" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --num_workers "$NUM_WORKERS" \
    --wandb_project knkgpt \
    --wandb_name "knkgpt_pretokenized_${N_LAYER}L_${N_HEAD}H_${N_EMBD}D"
