#!/usr/bin/env python3
"""
Smart training launcher that automatically detects and uses the appropriate training setup.
"""
import os
import sys
import subprocess
import torch
import argparse


def get_gpu_count():
    """Get the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def main():
    parser = argparse.ArgumentParser(description='Launch KnightKnaves GPT training')
    parser.add_argument('--config', type=str, default='default', 
                       choices=['small', 'large', 'debug', 'default'],
                       help='Configuration preset')
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all available)')
    parser.add_argument('--pretokenized', action='store_true',
                       help='Use pre-tokenized data')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args, remaining_args = parser.parse_known_args()
    
    # Detect available GPUs
    available_gpus = get_gpu_count()
    requested_gpus = args.gpus if args.gpus is not None else available_gpus
    
    if requested_gpus > available_gpus:
        print(f"Warning: Requested {requested_gpus} GPUs but only {available_gpus} available")
        requested_gpus = available_gpus
    
    # Build the command
    cmd = []
    
    if requested_gpus > 1:
        # Use torchrun for multi-GPU training
        print(f"🚀 Launching distributed training on {requested_gpus} GPUs...")
        cmd = [
            'torchrun',
            '--nproc_per_node', str(requested_gpus),
            'train_gpt_knights_knaves.py'
        ]
    else:
        # Single GPU or CPU training
        device = "GPU" if requested_gpus == 1 else "CPU"
        print(f"🚀 Launching training on {device}...")
        cmd = [sys.executable, 'train_gpt_knights_knaves.py']
    
    # Add configuration presets
    if args.config == 'small':
        print("📦 Using small model configuration")
        cmd.extend(['--n_layer', '4', '--n_head', '4', '--n_embd', '256', '--batch_size', '128'])
    elif args.config == 'large':
        print("📦 Using large model configuration")
        cmd.extend(['--n_layer', '12', '--n_head', '12', '--n_embd', '768', '--batch_size', '32'])
    elif args.config == 'debug':
        print("🐛 Debug mode - using subset of data")
        cmd.extend(['--n_puzzles', '10000', '--max_epochs', '2'])
    
    # Add pretokenized flag
    if args.pretokenized:
        print("💾 Using pre-tokenized data")
        cmd.extend(['--pretokenized_dir', './data/tokenized'])
    
    # Add resume flag
    if args.resume:
        print(f"♻️  Resuming from checkpoint: {args.resume}")
        cmd.extend(['--resume', args.resume])
    
    # Add any remaining arguments
    cmd.extend(remaining_args)
    
    # Print the full command
    print(f"\n🔧 Running command: {' '.join(cmd)}\n")
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    
    print("\n✅ Training completed successfully!")


if __name__ == '__main__':
    main()
