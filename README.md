# KnightKnaves GPT (knkgpt)

A GPT model trained on Knights and Knaves logical puzzles. This project adapts the minGPT architecture to learn to solve Knights and Knaves puzzles, where some islanders always tell the truth (Knights) and others always lie (Knaves).

## Setup

This project uses [uv](https://astral.sh/uv) for dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip sync

# Or install in editable mode
uv pip install -e .
```

## Project Structure

```
knkgpt/
├── data/
│   ├── knights_knaves/     # Tokenizer for K&K puzzles
│   └── n_2.jsonl          # Dataset of 100M puzzles with 2 islanders
├── mingpt/                # Model architecture and training
│   ├── model.py          # GPT model
│   ├── dataset.py        # Dataset and data loading
│   ├── trainer.py        # Training loop
│   ├── utils.py          # Utilities
│   └── wandb_utils.py    # Weights & Biases integration
├── train_gpt_knights_knaves.py    # Training script
└── train_gpt_knights_knaves.ipynb # Training notebook
```

## Training

### Quick Start: Pre-tokenized Training (Recommended)

For optimal performance, especially with large datasets, use pre-tokenized data:

```bash
# Pre-tokenize and train in one command
./scripts/pretokenize_and_train.sh
```

This script will:
1. Pre-tokenize the dataset if not already done
2. Launch training with optimized data loading

### Manual Pre-tokenization

```bash
# Step 1: Pre-tokenize the dataset (do this once)
python pretokenize_dataset.py \
    --input ./data/n_2.jsonl \
    --output ./data/tokenized \
    --max_length 512

# Step 2: Train with pre-tokenized data
python train_gpt_knights_knaves.py \
    --pretokenized_dir ./data/tokenized \
    --n_layer 8 \
    --n_head 8 \
    --n_embd 512 \
    --batch_size 64 \
    --max_epochs 10 \
    --wandb_project knkgpt
```

### Traditional Training (without pre-tokenization)

```bash
python train_gpt_knights_knaves.py \
    --data_path ./data/n_2.jsonl \
    --n_layer 8 \
    --n_head 8 \
    --n_embd 512 \
    --batch_size 64 \
    --max_epochs 10 \
    --wandb_project knkgpt
```

### Benchmarking Data Loading

Compare performance between original and pre-tokenized datasets:

```bash
python benchmark_data_loading.py \
    --batch_size 64 \
    --num_workers 4 \
    --n_batches 100
```

## Custom Tokenizer

The project includes a custom tokenizer specifically designed for Knights and Knaves puzzles. It tokenizes:
- Logical operators: `and`, `or`, `not`, `iff`, `imp`, `tt`, `ff`
- Functions: `isKnight`, `isKnave`, `says`
- Islander identifiers: `0`, `1`, etc.
- Solutions: `K` (Knight), `N` (Knave)

## Model Architecture

- **Architecture**: GPT-style transformer
- **Default config**: 8 layers, 8 attention heads, 512 embedding dimensions
- **Context length**: 512 tokens
- **Vocabulary size**: 29 tokens

## Dataset Format

The dataset is in JSONL format with each line containing:
```json
{
  "puzzle": "says 0 (iff (isKnight 0) (isKnave 0)), says 1 (tt), says 0 (ff)",
  "solution": "NK"
}
```

## Pre-tokenization Benefits

Pre-tokenizing your dataset provides significant performance improvements:

- **Faster Training**: Eliminates CPU tokenization bottleneck during training
- **Reduced Memory Usage**: Data is memory-mapped from disk instead of loaded entirely into RAM
- **Better GPU Utilization**: Optimized data loading with pinned memory for faster CPU-GPU transfers
- **Scalability**: Can handle datasets larger than available RAM (e.g., 100M puzzles)

For a 100M puzzle dataset:
- Pre-tokenization time: ~30-60 minutes (one-time cost)
- Storage: ~50-100 GB on disk (depending on sequence length)
- Training speedup: 5-10x faster data loading

## Monitoring Training

Training progress is logged to [Weights & Biases](https://wandb.ai). The training script logs:
- Training/validation loss
- Puzzle solving accuracy
- Per-solution-type accuracy (e.g., KK, KN, NK, NN)
- Example predictions

## Checkpoints

Model checkpoints are saved to `./ckpts/knkgpt/` including:
- `best_model.pt`: Best model by validation loss
- `checkpoint_STEP.pt`: Regular checkpoints during training
- `final_model.pt`: Final model after training

## License

This project is based on minGPT by Andrej Karpathy and the othello_world repository architecture.
