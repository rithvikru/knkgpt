# KnKGPT - Knights and Knaves GPT

A 25M parameter GPT model trained on Knights and Knaves logic puzzles, based on the architecture from [Othello-GPT](https://github.com/likenneth/othello_world).

## Overview

This project trains a transformer model to solve Knights and Knaves puzzles, where:
- Knights always tell the truth
- Knaves always lie
- The model learns to predict whether each islander is a Knight (K) or Knave (N)

## Installation

```bash
uv venv
uv pip install -e .
```

## Training

```bash
python train_gpt_knights_knaves.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook train_gpt_knights_knaves.ipynb
```

## Architecture

- Based on minGPT with modifications for KnK puzzles
- Custom tokenizer for logical expressions
- 25M parameters
- Trained on 100M puzzles with n=2 islanders
