#!/usr/bin/env python3
"""
Pre-tokenize Knights and Knaves dataset for efficient training.
Saves tokenized data as memory-mapped numpy arrays.
"""
import os
import argparse
import numpy as np
import jsonlines
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple

from data.knights_knaves.tokenizer import KnightsKnavesTokenizer


def pretokenize_dataset(
    input_path: str,
    output_dir: str,
    max_length: int = 512,
    chunk_size: int = 100000,
    n_puzzles: int = None,
    train_ratio: float = 0.98,
    seed: int = 42
):
    """
    Pre-tokenize the dataset and save as memory-mapped arrays.
    
    Args:
        input_path: Path to input JSONL file
        output_dir: Directory to save tokenized data
        max_length: Maximum sequence length
        chunk_size: Number of examples to process at once
        n_puzzles: Number of puzzles to use (None for all)
        train_ratio: Train/val split ratio
        seed: Random seed for train/val split
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = KnightsKnavesTokenizer()
    
    # First pass: count total examples and determine splits
    print("Counting examples...")
    total_count = 0
    with jsonlines.open(input_path) as reader:
        for i, _ in enumerate(tqdm(reader)):
            if n_puzzles and i >= n_puzzles:
                break
            total_count += 1
    
    print(f"Total examples: {total_count:,}")
    
    # Determine train/val split
    np.random.seed(seed)
    indices = np.arange(total_count)
    np.random.shuffle(indices)
    
    n_train = int(total_count * train_ratio)
    train_indices = set(indices[:n_train])
    
    print(f"Train examples: {n_train:,}")
    print(f"Val examples: {total_count - n_train:,}")
    
    # Create memory-mapped arrays
    train_data = {
        'input_ids': np.memmap(
            os.path.join(output_dir, 'train_input_ids.npy'),
            dtype=np.int32,
            mode='w+',
            shape=(n_train, max_length - 1)
        ),
        'target_ids': np.memmap(
            os.path.join(output_dir, 'train_target_ids.npy'),
            dtype=np.int32,
            mode='w+',
            shape=(n_train, max_length - 1)
        ),
        'attention_mask': np.memmap(
            os.path.join(output_dir, 'train_attention_mask.npy'),
            dtype=np.bool_,
            mode='w+',
            shape=(n_train, max_length - 1)
        ),
    }
    
    val_data = {
        'input_ids': np.memmap(
            os.path.join(output_dir, 'val_input_ids.npy'),
            dtype=np.int32,
            mode='w+',
            shape=(total_count - n_train, max_length - 1)
        ),
        'target_ids': np.memmap(
            os.path.join(output_dir, 'val_target_ids.npy'),
            dtype=np.int32,
            mode='w+',
            shape=(total_count - n_train, max_length - 1)
        ),
        'attention_mask': np.memmap(
            os.path.join(output_dir, 'val_attention_mask.npy'),
            dtype=np.bool_,
            mode='w+',
            shape=(total_count - n_train, max_length - 1)
        ),
    }
    
    # Second pass: tokenize and save
    print("\nTokenizing and saving...")
    train_idx = 0
    val_idx = 0
    
    # Process in chunks for efficiency
    chunk_buffer = []
    
    with jsonlines.open(input_path) as reader:
        for global_idx, item in enumerate(tqdm(reader, total=total_count)):
            if n_puzzles and global_idx >= n_puzzles:
                break
                
            chunk_buffer.append((global_idx, item))
            
            # Process chunk when buffer is full
            if len(chunk_buffer) >= chunk_size or global_idx == total_count - 1:
                for idx, data in chunk_buffer:
                    # Tokenize
                    puzzle = data['puzzle']
                    solution = data['solution']
                    
                    # Encode the example
                    encoded = tokenizer.encode_example(puzzle, solution)
                    
                    # Pad sequence
                    encoded = tokenizer.pad_sequence(encoded, max_length)
                    
                    # Create input and target
                    input_seq = encoded[:-1]
                    target_seq = encoded[1:]
                    
                    # Create attention mask (1 for real tokens, 0 for padding)
                    attention_mask = [1 if tok != tokenizer.pad_idx else 0 for tok in input_seq]
                    
                    # Save to appropriate split
                    if idx in train_indices:
                        train_data['input_ids'][train_idx] = input_seq
                        train_data['target_ids'][train_idx] = target_seq
                        train_data['attention_mask'][train_idx] = attention_mask
                        train_idx += 1
                    else:
                        val_data['input_ids'][val_idx] = input_seq
                        val_data['target_ids'][val_idx] = target_seq
                        val_data['attention_mask'][val_idx] = attention_mask
                        val_idx += 1
                
                # Clear buffer
                chunk_buffer = []
    
    # Flush memory-mapped arrays
    for data in [train_data, val_data]:
        for arr in data.values():
            del arr
    
    # Save metadata
    metadata = {
        'vocab_size': tokenizer.vocab_size,
        'max_length': max_length,
        'n_train': n_train,
        'n_val': total_count - n_train,
        'train_ratio': train_ratio,
        'seed': seed,
        'pad_token_id': tokenizer.pad_idx,
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save tokenizer
    with open(os.path.join(output_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"\nPre-tokenization complete!")
    print(f"Data saved to: {output_dir}")
    print(f"Train examples: {train_idx:,}")
    print(f"Val examples: {val_idx:,}")
    
    # Calculate size on disk
    total_size = 0
    for filename in os.listdir(output_dir):
        if filename.endswith('.npy'):
            filepath = os.path.join(output_dir, filename)
            total_size += os.path.getsize(filepath)
    
    print(f"Total size on disk: {total_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='Pre-tokenize Knights and Knaves dataset')
    parser.add_argument('--input', type=str, default='./data/n_2.jsonl', help='Input JSONL file')
    parser.add_argument('--output', type=str, default='./data/tokenized', help='Output directory')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--chunk_size', type=int, default=100000, help='Processing chunk size')
    parser.add_argument('--n_puzzles', type=int, default=None, help='Number of puzzles (None for all)')
    parser.add_argument('--train_ratio', type=float, default=0.98, help='Train/val split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    pretokenize_dataset(
        input_path=args.input,
        output_dir=args.output,
        max_length=args.max_length,
        chunk_size=args.chunk_size,
        n_puzzles=args.n_puzzles,
        train_ratio=args.train_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
