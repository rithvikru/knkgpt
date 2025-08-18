"""
Training script for Knights and Knaves GPT.
"""

import argparse
import os
from pathlib import Path

import torch
import wandb

from mingpt.model import create_model, GPTConfig
from mingpt.dataset import KnightsKnavesDataModule
from mingpt.trainer import Trainer, TrainerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train KnKGPT on Knights and Knaves puzzles")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="data/n_2.jsonl",
                       help="Path to training data")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use (None for all)")
    parser.add_argument("--train_split", type=float, default=0.95,
                       help="Fraction of data to use for training")
    
    # Model arguments
    parser.add_argument("--n_embd", type=int, default=512,
                       help="Embedding dimension")
    parser.add_argument("--n_layer", type=int, default=8,
                       help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout probability")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=10,
                       help="Maximum number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--grad_norm_clip", type=float, default=1.0,
                       help="Gradient norm clipping")
    
    # Checkpointing arguments
    parser.add_argument("--ckpt_path", type=str, default="ckpts/knkgpt",
                       help="Path to save checkpoints")
    parser.add_argument("--save_every", type=int, default=5000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=1000,
                       help="Evaluate every N steps")
    parser.add_argument("--log_every", type=int, default=100,
                       help="Log metrics every N steps")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="knkgpt",
                       help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                       help="Wandb run name")
    parser.add_argument("--wandb_notes", type=str, default=None,
                       help="Wandb run notes")
    parser.add_argument("--wandb_offline", action="store_true",
                       help="Run wandb in offline mode")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/mps/cpu/auto)")
    parser.add_argument("--use_amp", action="store_true",
                       help="Use automatic mixed precision")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
        
    print(f"Using device: {device}")
    
    # Set wandb mode
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        
    # Create data module
    print("Loading data...")
    data_module = KnightsKnavesDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_length=512,
        num_workers=args.num_workers,
        train_split=args.train_split,
        seed=args.seed,
        max_samples=args.max_samples
    )
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Create model
    print("Creating model...")
    model_config = GPTConfig(
        vocab_size=data_module.vocab_size,
        n_positions=512,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=4 * args.n_embd,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
    )
    model = create_model(vocab_size=data_module.vocab_size)
    
    # Create trainer config
    trainer_config = TrainerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=1e-8,
        adam_betas=(0.9, 0.95),
        grad_norm_clip=args.grad_norm_clip,
        warmup_steps=args.warmup_steps,
        lr_decay=True,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ckpt_path=args.ckpt_path,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name or f"knkgpt_bs{args.batch_size}_lr{args.learning_rate}",
        wandb_notes=args.wandb_notes,
        device=device,
        use_amp=args.use_amp and device == "cuda",
    )
    
    # Create trainer
    trainer = Trainer(model, trainer_config)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        
    # Train
    print("Starting training...")
    trainer.train(train_loader, val_loader)
    
    print("Training completed!")
    
    # Generate some samples at the end
    print("\nGenerating sample predictions:")
    samples = trainer.generate_samples(data_module.tokenizer, num_samples=5)
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}: {sample}")


if __name__ == "__main__":
    main()
