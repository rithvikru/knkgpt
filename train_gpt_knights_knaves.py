#!/usr/bin/env python3
"""
Train GPT on Knights and Knaves puzzles.
"""
import os
import argparse
import torch
from datetime import datetime

from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.dataset import KnightsKnavesDataModule
from mingpt.pretokenized_dataset import PreTokenizedDataModule
from mingpt.utils import set_seed
from mingpt.wandb_utils import init_wandb, finish_run


def main():
    parser = argparse.ArgumentParser(description='Train KnightKnaves GPT')
    
    # Model arguments
    parser.add_argument('--n_layer', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--block_size', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/n_2.jsonl', help='Path to data file')
    parser.add_argument('--pretokenized_dir', type=str, default=None, help='Path to pre-tokenized data directory')
    parser.add_argument('--n_puzzles', type=int, default=None, help='Number of puzzles to use (None for all)')
    parser.add_argument('--train_ratio', type=float, default=0.98, help='Train/val split ratio')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--grad_norm_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--warmup_tokens', type=float, default=375e6, help='Warmup tokens')
    parser.add_argument('--final_tokens', type=float, default=100e9, help='Final tokens for LR decay')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging arguments
    parser.add_argument('--wandb_project', type=str, default='knkgpt', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/knkgpt', help='Checkpoint directory')
    parser.add_argument('--log_every', type=int, default=10, help='Log every N steps')
    parser.add_argument('--eval_every', type=int, default=500, help='Evaluate every N steps')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--eval_samples', type=int, default=1000, help='Number of samples for evaluation')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create data module
    print("Loading data...")
    
    if args.pretokenized_dir:
        print(f"Using pre-tokenized data from: {args.pretokenized_dir}")
        data_module = PreTokenizedDataModule(
            data_dir=args.pretokenized_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
    else:
        print(f"Loading raw data from: {args.data_path}")
        data_module = KnightsKnavesDataModule(
            data_path=args.data_path,
            batch_size=args.batch_size,
            max_length=args.block_size,
            n_puzzles=args.n_puzzles,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    
    # Create model config
    model_config = GPTConfig(
        vocab_size=data_module.get_vocab_size(),
        block_size=data_module.get_block_size(),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        embd_pdrop=args.dropout,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
    )
    
    # Create model
    print("Creating model...")
    model = GPT(model_config)
    
    # Create trainer config
    train_config = TrainerConfig(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        grad_norm_clip=args.grad_norm_clip,
        weight_decay=args.weight_decay,
        lr_decay=True,
        warmup_tokens=args.warmup_tokens,
        final_tokens=args.final_tokens,
        ckpt_dir=args.ckpt_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        num_workers=args.num_workers,
        device=args.device,
    )
    
    # Initialize wandb
    run_name = args.wandb_name or f"knkgpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb_config = {
        'model': model_config.__dict__,
        'training': train_config.__dict__,
        'data': {
            'data_path': args.data_path,
            'n_puzzles': args.n_puzzles,
            'train_ratio': args.train_ratio,
            'train_size': len(data_module.train_dataset),
            'val_size': len(data_module.val_dataset),
        },
        'seed': args.seed,
    }
    
    run = init_wandb(
        project=args.wandb_project,
        name=run_name,
        config=wandb_config,
        tags=['knights-knaves', 'gpt', f'n_layer_{args.n_layer}', f'n_embd_{args.n_embd}'],
        notes=f"Training GPT on Knights and Knaves puzzles with {len(data_module.train_dataset)} training examples"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=data_module.train_dataset,
        val_dataset=data_module.val_dataset,
        config=train_config
    )
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.global_step = checkpoint['step']
        trainer.current_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {trainer.current_epoch}, step {trainer.global_step}")
    
    # Train
    print("Starting training...")
    train_losses, val_losses = trainer.train()
    
    # Finish wandb run
    finish_run()
    
    print("Training complete!")


if __name__ == '__main__':
    main()
