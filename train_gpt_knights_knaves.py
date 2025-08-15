#!/usr/bin/env python3
"""
Train GPT on Knights and Knaves puzzles.
Designed for 8xB200 GPUs with 100M puzzles dataset.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import get_knights_knaves
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample, set_seed


def setup_distributed():
    """Setup for distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(gpu)

        return rank, world_size, gpu
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def validate_model(model, dataset, device, num_samples=1000):
    """
    Validate the model on Knights and Knaves puzzles.
    Check if the model correctly predicts the solution (K/N for each agent).
    """
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset.val))), desc="Validating"):
            # Get a validation puzzle
            puzzle_chars = dataset.val[i]

            # Find where the solution starts (after "=>")
            solution_start = None
            for j in range(len(puzzle_chars) - 3):
                if (
                    puzzle_chars[j] == "="
                    and puzzle_chars[j + 1] == ">"
                    and puzzle_chars[j + 2] == " "
                ):
                    solution_start = j + 3
                    break

            if solution_start is None:
                continue

            # Get context (everything before the solution)
            context = puzzle_chars[:solution_start]

            # Encode context
            train_dataset = CharDataset(dataset)
            x = torch.tensor(
                [train_dataset.stoi[s] for s in context], dtype=torch.long
            )[None, ...].to(device)

            # Generate prediction
            y = sample(model, x, len(puzzle_chars) - solution_start, temperature=0.1)[0]
            prediction = [train_dataset.itos[int(i)] for i in y if i != -1]

            # Extract actual solution
            actual_solution = "".join(
                [c for c in puzzle_chars[solution_start:] if c != -100]
            )
            predicted_solution = "".join(prediction[: len(actual_solution)])

            if predicted_solution == actual_solution:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Validation accuracy: {correct}/{total} = {accuracy:.2%}")

    model.train()
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train GPT on Knights and Knaves puzzles"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/n_2.jsonl",
        help="Path to the Knights and Knaves dataset",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to load (None for all 100M)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size per GPU"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=6e-4, help="Learning rate"
    )
    parser.add_argument(
        "--n-layer", type=int, default=8, help="Number of transformer layers"
    )
    parser.add_argument(
        "--n-head", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--n-embd", type=int, default=512, help="Embedding dimension")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./ckpts",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--validate-every", type=int, default=1000, help="Validate every N steps"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="knights-knaves-gpt", help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="Wandb run name"
    )
    parser.add_argument(
        "--wandb-disabled", action="store_true", help="Disable wandb logging"
    )

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, gpu = setup_distributed()

    # Set random seed
    set_seed(args.seed)

    # Create checkpoint directory
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize wandb on rank 0 only
    if rank == 0 and not args.wandb_disabled:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"gpt_knights_knaves_{time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": {
                    "n_layer": args.n_layer,
                    "n_head": args.n_head,
                    "n_embd": args.n_embd,
                    "vocab_size": None,  # Will be set later
                    "block_size": None,  # Will be set later
                },
                "training": {
                    "batch_size_per_gpu": args.batch_size,
                    "total_batch_size": args.batch_size * world_size,
                    "max_epochs": args.max_epochs,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                    "world_size": world_size,
                },
                "data": {
                    "path": args.data_path,
                    "max_games": args.max_games,
                },
            }
        )

    # Load dataset
    print(f"Loading Knights and Knaves dataset from {args.data_path}...")
    knights_knaves = get_knights_knaves(
        data_path=args.data_path,
        max_games=args.max_games,
        val_split=0.01,  # Small validation set for 100M dataset
        seed=args.seed,
    )

    # Create character dataset wrapper
    train_dataset = CharDataset(knights_knaves)

    # Model configuration
    mconf = GPTConfig(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    model = GPT(mconf)

    if rank == 0:
        print(f"Model configuration:")
        print(f"  Vocab size: {mconf.vocab_size}")
        print(f"  Block size: {mconf.block_size}")
        print(f"  Layers: {mconf.n_layer}")
        print(f"  Heads: {mconf.n_head}")
        print(f"  Embedding dim: {mconf.n_embd}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        # Update wandb config with actual model parameters
        if not args.wandb_disabled:
            wandb.config.update({
                "model.vocab_size": mconf.vocab_size,
                "model.block_size": mconf.block_size,
                "model.total_parameters": total_params,
            })
            
            # Log model architecture
            wandb.watch(model, log="all", log_freq=100)  # Log gradients and parameters every 100 steps
            
            # Create a summary of model architecture
            model_summary = []
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    num_params = sum(p.numel() for p in module.parameters())
                    if num_params > 0:
                        model_summary.append({
                            "layer": name,
                            "type": module.__class__.__name__,
                            "parameters": num_params
                        })
            
            # Log model architecture as a table
            wandb.log({"model_architecture": wandb.Table(
                columns=["layer", "type", "parameters"],
                data=[[item["layer"], item["type"], item["parameters"]] for item in model_summary]
            )})

    # Move model to GPU
    model = model.cuda(gpu)

    # Wrap model for distributed training if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[gpu])

    # Training configuration
    # With 8 GPUs and batch size 512 per GPU, total batch size is 4096
    total_batch_size = args.batch_size * world_size

    t_start = time.strftime("_%Y%m%d_%H%M%S")
    tconf = TrainerConfig(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=True,
        warmup_tokens=len(train_dataset) * train_dataset.block_size * 0.1,  # 10% warmup
        final_tokens=len(train_dataset) * train_dataset.block_size * args.max_epochs,
        num_workers=4,
        ckpt_path=(
            f"{args.checkpoint_dir}/gpt_knights_knaves{t_start}.ckpt"
            if rank == 0
            else None
        ),
        validate_every=args.validate_every,
    )

    if rank == 0:
        print(f"Training configuration:")
        print(f"  Total batch size: {total_batch_size}")
        print(f"  Steps per epoch: {len(train_dataset) // total_batch_size}")
        print(
            f"  Total steps: {len(train_dataset) * args.max_epochs // total_batch_size}"
        )

    # Create trainer
    trainer = Trainer(model, train_dataset, None, tconf)
    
    # Pass wandb logging flag to trainer
    if rank == 0:
        trainer.use_wandb = not args.wandb_disabled
    else:
        trainer.use_wandb = False

    # Add validation callback
    if rank == 0:

        def validation_callback(trainer, epoch, step):
            if step % args.validate_every == 0:
                accuracy = validate_model(
                    (
                        trainer.model.module
                        if hasattr(trainer.model, "module")
                        else trainer.model
                    ),
                    knights_knaves,
                    trainer.device,
                    num_samples=1000,
                )
                print(f"Step {step}: Validation accuracy = {accuracy:.2%}")
                
                # Log to wandb
                if not args.wandb_disabled:
                    wandb.log({
                        "validation/accuracy": accuracy,
                        "epoch": epoch,
                        "step": step,
                    })
                    
                    # Log attention pattern samples periodically
                    if step % (args.validate_every * 5) == 0:  # Every 5 validation steps
                        try:
                            # Get a sample prediction to visualize attention
                            model_to_eval = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                            model_to_eval.eval()
                            
                            # Get one sample
                            sample_idx = 0
                            puzzle_chars = knights_knaves.val[sample_idx]
                            solution_start = None
                            for j in range(len(puzzle_chars) - 3):
                                if puzzle_chars[j] == "=" and puzzle_chars[j + 1] == ">" and puzzle_chars[j + 2] == " ":
                                    solution_start = j + 3
                                    break
                            
                            if solution_start:
                                context = puzzle_chars[:solution_start]
                                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(trainer.device)
                                
                                # Get embeddings
                                with torch.no_grad():
                                    # Get token embeddings
                                    tok_emb = model_to_eval.tok_emb(x)  # (batch, seq_len, n_embd)
                                    
                                    # Log embedding statistics
                                    wandb.log({
                                        "embeddings/mean": tok_emb.mean().item(),
                                        "embeddings/std": tok_emb.std().item(),
                                        "embeddings/histogram": wandb.Histogram(tok_emb.cpu().numpy().flatten()),
                                    })
                            
                            model_to_eval.train()
                        except Exception as e:
                            print(f"Warning: Could not log attention/embedding visualizations: {e}")

        trainer.validation_callback = validation_callback

    # Train the model
    print(f"Starting training on rank {rank}...")
    trainer.train()

    # Final validation
    if rank == 0:
        print("Final validation...")
        final_accuracy = validate_model(
            model.module if hasattr(model, "module") else model,
            knights_knaves,
            trainer.device,
            num_samples=5000,
        )
        print(f"Final validation accuracy: {final_accuracy:.2%}")
        
        # Log final results to wandb
        if not args.wandb_disabled:
            wandb.summary["final_validation_accuracy"] = final_accuracy
            wandb.summary["total_training_steps"] = trainer.iteration if hasattr(trainer, 'iteration') else 0

    # Cleanup
    if rank == 0 and not args.wandb_disabled:
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()

