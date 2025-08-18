# KnightKnaves GPT (knkgpt)

A GPT model trained on Knights and Knaves logical puzzles with automatic single/multi-GPU support. This project adapts the minGPT architecture to learn to solve Knights and Knaves puzzles, where some islanders always tell the truth (Knights) and others always lie (Knaves).

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/knkgpt.git
cd knkgpt

# Install dependencies
pip install -e .

# Run training (automatically detects and uses all available GPUs)
python run_training.py

# Or use specific configurations
python run_training.py --config small --gpus 2  # Small model on 2 GPUs
python run_training.py --config debug           # Quick debug run
python run_training.py --pretokenized           # Use pre-tokenized data
```

## 🎯 Features

- **Automatic GPU Detection**: Seamlessly runs on single GPU, multi-GPU, or CPU
- **Distributed Training**: Built-in support for multi-GPU training with proper wandb logging
- **Smart Launcher**: `run_training.py` automatically configures optimal settings
- **Pre-tokenization Support**: 5-10x faster training with pre-tokenized datasets
- **Wandb Integration**: Automatic experiment tracking and visualization

## 📦 Installation

### Option 1: pip (Recommended for SSH machines)
```bash
pip install -e .
```

### Option 2: uv (Fast dependency management)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip sync
# Or
uv pip install -e .
```

## 🏃 Training

### Easiest Way: Smart Launcher
```bash
# Automatically detects GPUs and runs optimal configuration
python run_training.py

# With options
python run_training.py --config large --gpus 4
python run_training.py --pretokenized --resume checkpoints/latest.pt
```

### Manual Launch (Single GPU)
```bash
python train_gpt_knights_knaves.py \
    --data_path ./data/n_2.jsonl \
    --n_layer 8 \
    --n_head 8 \
    --n_embd 512 \
    --batch_size 64 \
    --max_epochs 10
```

### Manual Launch (Multi-GPU)
```bash
# For 4 GPUs
torchrun --nproc_per_node=4 train_gpt_knights_knaves.py \
    --data_path ./data/n_2.jsonl \
    --n_layer 8 \
    --n_head 8 \
    --n_embd 512 \
    --batch_size 64 \
    --max_epochs 10
```

### Pre-tokenized Training (Faster)
```bash
# Step 1: Pre-tokenize the dataset (one-time)
python pretokenize_dataset.py \
    --input ./data/n_2.jsonl \
    --output ./data/tokenized \
    --max_length 512

# Step 2: Train with pre-tokenized data
python run_training.py --pretokenized
```

## 🔧 Configuration Presets

The `run_training.py` script includes several presets:

- **`default`**: Standard configuration (8 layers, 8 heads, 512 dim)
- **`small`**: Smaller model for testing (4 layers, 4 heads, 256 dim)
- **`large`**: Larger model (12 layers, 12 heads, 768 dim)
- **`debug`**: Quick debugging run (10k samples, 2 epochs)

## 📊 Project Structure

```
knkgpt/
├── run_training.py        # 🚀 Smart training launcher (start here!)
├── train_gpt_knights_knaves.py    # Main training script
├── data/
│   ├── knights_knaves/    # Custom tokenizer
│   └── n_2.jsonl         # Dataset (100M puzzles)
├── mingpt/               # Model architecture
│   ├── model.py         # GPT implementation
│   ├── trainer.py       # Training loop (distributed-aware)
│   ├── dataset.py       # Data loading
│   └── utils.py         # Utilities
└── scripts/             # Helper scripts
```

## 🧩 Dataset Format

The dataset is in JSONL format with each line containing:
```json
{
  "puzzle": "says 0 (iff (isKnight 0) (isKnave 0)), says 1 (tt), says 0 (ff)",
  "solution": "NK"
}
```

Where:
- `0`, `1` are islander identifiers
- `K` = Knight (truth-teller), `N` = Knave (liar)
- Logical operators: `and`, `or`, `not`, `iff`, `imp`, `tt` (true), `ff` (false)

## 📈 Monitoring

Training automatically logs to [Weights & Biases](https://wandb.ai):
- Loss curves and learning rate
- Puzzle solving accuracy
- Per-solution-type accuracy (KK, KN, NK, NN)
- Example predictions
- GPU utilization (if available)

View your runs at: https://wandb.ai/your-username/knkgpt

## 💾 Checkpoints

Models are saved to `./ckpts/knkgpt/`:
- `checkpoint_0.pt`: Initial model
- `checkpoint_1000.pt`, `checkpoint_2000.pt`, ...: Regular saves
- `best_model.pt`: Best validation loss
- `final_model.pt`: End of training

Resume training:
```bash
python run_training.py --resume ./ckpts/knkgpt/checkpoint_5000.pt
```

## ⚡ Performance Tips

1. **Use Pre-tokenization**: 5-10x faster data loading
   ```bash
   python pretokenize_dataset.py --input data/n_2.jsonl --output data/tokenized
   python run_training.py --pretokenized
   ```

2. **Multi-GPU Training**: Automatically enabled when GPUs available
   ```bash
   python run_training.py  # Uses all GPUs automatically
   ```

3. **Mixed Precision**: Enable for faster training (coming soon)

4. **Optimal Batch Size**: 
   - Single GPU: 64-128
   - Multi-GPU: 32-64 per GPU

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_gpt_knights_knaves.py --batch_size 32

# Or use gradient accumulation (coming soon)
```

### Wandb Errors in Distributed Training
The code automatically handles wandb initialization only on the main process. If you still see errors, ensure you're using the latest version:
```bash
pip install --upgrade wandb
```

### Slow Data Loading
Use pre-tokenized datasets:
```bash
python pretokenize_dataset.py --input data/n_2.jsonl --output data/tokenized
python run_training.py --pretokenized
```

## 📄 License

This project is based on minGPT by Andrej Karpathy and the othello_world repository architecture.