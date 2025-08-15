# Knights and Knaves GPT

This project trains a GPT model on Knights and Knaves logic puzzles, adapted from the Othello World architecture.

## Overview

Knights and Knaves puzzles are logic puzzles where:
- Knights always tell the truth
- Knaves always lie
- Given statements from agents, deduce who is a Knight (K) and who is a Knave (N)

## Dataset

The dataset contains 100M Knights and Knaves puzzles in JSONL format:
- Location: `../data/n_2.jsonl`
- Format: `{"puzzle": "says 0 (...)", "solution": "KN"}`
- Each puzzle contains logical statements and the solution indicates Knight/Knave for each agent

## Model Architecture

- Based on GPT architecture from Othello World
- Character-level tokenization
- Autoregressive training to predict solutions given puzzles
- Default configuration for 100M dataset:
  - 12 transformer layers
  - 12 attention heads  
  - 768 embedding dimensions
  - ~85M parameters

## Training

### Quick Start (Jupyter Notebook)

For interactive training and experimentation:
```bash
jupyter notebook train_gpt_knights_knaves.ipynb
```

### Distributed Training (8xB200 GPUs)

For full-scale training on 8 B200 GPUs:
```bash
./launch_training.sh
```

This will:
- Use distributed data parallel training across 8 GPUs
- Train with batch size 8192 (1024 per GPU)
- Save checkpoints to `./ckpts/`
- Validate every 1000 steps

### Single GPU Training

For testing or smaller datasets:
```bash
python train_gpt_knights_knaves.py \
    --data-path ../data/n_2.jsonl \
    --max-games 1000000 \
    --batch-size 64 \
    --n-layer 8 \
    --n-head 8 \
    --n-embd 512
```

## File Structure

```
knights_knaves/
├── data/
│   ├── __init__.py
│   └── knights_knaves.py      # Dataset loader and tokenizer
├── mingpt/                    # GPT model implementation (from Othello World)
│   ├── model.py
│   ├── trainer.py
│   ├── dataset.py
│   └── ...
├── ckpts/                     # Saved model checkpoints
├── train_gpt_knights_knaves.py    # Main training script
├── train_gpt_knights_knaves.ipynb # Interactive training notebook
├── launch_training.sh         # Multi-GPU launch script
└── README.md
```

## Validation

The model is validated by:
1. Parsing puzzles to extract the puzzle and solution
2. Feeding the puzzle to the model with " => " separator
3. Generating the solution (sequence of K/N characters)
4. Comparing predicted vs actual solutions

Expected performance:
- The model should learn to solve Knights and Knaves puzzles with high accuracy
- Validation accuracy should improve throughout training

## Customization

To adapt for different puzzle formats or datasets:
1. Modify the tokenizer in `data/knights_knaves.py`
2. Adjust the data loading logic for your format
3. Update validation logic if solution format differs

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### Setup Project

1. Clone the repository:
```bash
git clone <repository-url>
cd knkgpt
```

2. Install dependencies with uv:
```bash
# Create virtual environment and install dependencies
uv sync

# Or install with specific Python version
uv python install 3.10
uv sync
```

3. Activate the environment:
```bash
# The virtual environment is created at .venv
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Alternative: Install with pip

If you prefer traditional pip:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.9+
- PyTorch with CUDA support
- 8xB200 GPUs for full training (or adjust batch size for fewer GPUs)
- ~25GB disk space for 100M puzzles dataset

## Citation

Based on Othello World by Li et al.:
```
@article{li2022emergent,
  title={Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task},
  author={Li, Kenneth and others},
  year={2022}
}
```