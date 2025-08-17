#!/usr/bin/env python3
"""
Parallel pre-tokenization of Knights and Knaves dataset for efficient training.
Uses multiprocessing to maximize CPU core utilization.
Saves tokenized data as memory-mapped numpy arrays.
"""
import os
import argparse
import numpy as np
import jsonlines
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
from functools import partial
import time
from collections import defaultdict
import logging

from data.knights_knaves.tokenizer import KnightsKnavesTokenizer


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def tokenize_batch(batch_data: List[Tuple[int, dict, int]], 
                   tokenizer: KnightsKnavesTokenizer,
                   max_length: int,
                   train_indices: set) -> Dict[str, List]:
    """
    Tokenize a batch of examples.
    
    Args:
        batch_data: List of (global_idx, item, original_order) tuples
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        train_indices: Set of indices that belong to train split
        
    Returns:
        Dictionary with train and val data
    """
    train_results = {
        'input_ids': [],
        'target_ids': [],
        'attention_mask': [],
        'indices': []
    }
    
    val_results = {
        'input_ids': [],
        'target_ids': [],
        'attention_mask': [],
        'indices': []
    }
    
    for global_idx, item, _ in batch_data:
        # Tokenize
        puzzle = item['puzzle']
        solution = item['solution']
        
        # Encode the example
        encoded = tokenizer.encode_example(puzzle, solution)
        
        # Pad sequence
        encoded = tokenizer.pad_sequence(encoded, max_length)
        
        # Create input and target
        input_seq = encoded[:-1]
        target_seq = encoded[1:]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if tok != tokenizer.pad_idx else 0 for tok in input_seq]
        
        # Add to appropriate split
        if global_idx in train_indices:
            train_results['input_ids'].append(input_seq)
            train_results['target_ids'].append(target_seq)
            train_results['attention_mask'].append(attention_mask)
            train_results['indices'].append(global_idx)
        else:
            val_results['input_ids'].append(input_seq)
            val_results['target_ids'].append(target_seq)
            val_results['attention_mask'].append(attention_mask)
            val_results['indices'].append(global_idx)
    
    return {'train': train_results, 'val': val_results}


def process_chunk(chunk_data: Tuple[int, List[Tuple[int, dict, int]]], 
                  tokenizer_pkl: bytes,
                  max_length: int,
                  train_indices: set) -> Tuple[int, Dict[str, List]]:
    """
    Process a chunk of data in a worker process.
    
    Args:
        chunk_data: Tuple of (chunk_id, batch_data)
        tokenizer_pkl: Pickled tokenizer
        max_length: Maximum sequence length
        train_indices: Set of indices that belong to train split
        
    Returns:
        Tuple of (chunk_id, results)
    """
    chunk_id, batch_data = chunk_data
    
    # Unpickle tokenizer in worker process
    import pickle
    tokenizer = pickle.loads(tokenizer_pkl)
    
    # Process the batch
    results = tokenize_batch(batch_data, tokenizer, max_length, train_indices)
    
    return chunk_id, results


def pretokenize_dataset_parallel(
    input_path: str,
    output_dir: str,
    max_length: int = 512,
    batch_size: int = 1000,
    n_puzzles: int = None,
    train_ratio: float = 0.98,
    seed: int = 42,
    num_workers: int = None
):
    """
    Pre-tokenize the dataset using parallel processing.
    
    Args:
        input_path: Path to input JSONL file
        output_dir: Directory to save tokenized data
        max_length: Maximum sequence length
        batch_size: Number of examples per batch for parallel processing
        n_puzzles: Number of puzzles to use (None for all)
        train_ratio: Train/val split ratio
        seed: Random seed for train/val split
        num_workers: Number of worker processes (None for auto-detect)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = KnightsKnavesTokenizer()
    
    # Pickle tokenizer for workers
    tokenizer_pkl = pickle.dumps(tokenizer)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    logging.info(f"Using {num_workers} worker processes")
    
    # First pass: count total examples and determine splits
    logging.info("Counting examples...")
    total_count = 0
    with jsonlines.open(input_path) as reader:
        for i, _ in enumerate(tqdm(reader)):
            if n_puzzles and i >= n_puzzles:
                break
            total_count += 1
    
    logging.info(f"Total examples: {total_count:,}")
    
    # Determine train/val split
    np.random.seed(seed)
    indices = np.arange(total_count)
    np.random.shuffle(indices)
    
    n_train = int(total_count * train_ratio)
    train_indices = set(indices[:n_train])
    
    logging.info(f"Train examples: {n_train:,}")
    logging.info(f"Val examples: {total_count - n_train:,}")
    
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
    
    # Second pass: tokenize and save with parallel processing
    logging.info("\nTokenizing and saving (parallel)...")
    
    # Create index mappings for writing results
    train_idx_mapping = {}
    val_idx_mapping = {}
    train_write_idx = 0
    val_write_idx = 0
    
    for idx in range(total_count):
        if idx in train_indices:
            train_idx_mapping[idx] = train_write_idx
            train_write_idx += 1
        else:
            val_idx_mapping[idx] = val_write_idx
            val_write_idx += 1
    
    # Process in batches
    batch_buffer = []
    chunk_id = 0
    chunks_to_process = []
    
    # Read data and prepare chunks
    with jsonlines.open(input_path) as reader:
        for global_idx, item in enumerate(reader):
            if n_puzzles and global_idx >= n_puzzles:
                break
            
            batch_buffer.append((global_idx, item, len(batch_buffer)))
            
            # When batch is full, add to chunks
            if len(batch_buffer) >= batch_size:
                chunks_to_process.append((chunk_id, batch_buffer.copy()))
                chunk_id += 1
                batch_buffer = []
        
        # Add remaining batch
        if batch_buffer:
            chunks_to_process.append((chunk_id, batch_buffer))
    
    # Set up multiprocessing pool
    process_func = partial(
        process_chunk,
        tokenizer_pkl=tokenizer_pkl,
        max_length=max_length,
        train_indices=train_indices
    )
    
    # Process chunks in parallel
    results_dict = {}
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance
        with tqdm(total=len(chunks_to_process), desc="Processing chunks") as pbar:
            for chunk_id, results in pool.imap_unordered(process_func, chunks_to_process):
                results_dict[chunk_id] = results
                pbar.update(1)
    
    # Write results to memory-mapped arrays in order
    logging.info("Writing results to disk...")
    
    train_written = 0
    val_written = 0
    
    for chunk_id in tqdm(range(len(chunks_to_process)), desc="Writing to disk"):
        chunk_results = results_dict[chunk_id]
        
        # Write train results
        train_res = chunk_results['train']
        for i, global_idx in enumerate(train_res['indices']):
            write_idx = train_idx_mapping[global_idx]
            train_data['input_ids'][write_idx] = train_res['input_ids'][i]
            train_data['target_ids'][write_idx] = train_res['target_ids'][i]
            train_data['attention_mask'][write_idx] = train_res['attention_mask'][i]
            train_written += 1
        
        # Write val results
        val_res = chunk_results['val']
        for i, global_idx in enumerate(val_res['indices']):
            write_idx = val_idx_mapping[global_idx]
            val_data['input_ids'][write_idx] = val_res['input_ids'][i]
            val_data['target_ids'][write_idx] = val_res['target_ids'][i]
            val_data['attention_mask'][write_idx] = val_res['attention_mask'][i]
            val_written += 1
    
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
    
    logging.info(f"\nPre-tokenization complete!")
    logging.info(f"Data saved to: {output_dir}")
    logging.info(f"Train examples: {train_written:,}")
    logging.info(f"Val examples: {val_written:,}")
    
    # Calculate size on disk
    total_size = 0
    for filename in os.listdir(output_dir):
        if filename.endswith('.npy'):
            filepath = os.path.join(output_dir, filename)
            total_size += os.path.getsize(filepath)
    
    logging.info(f"Total size on disk: {total_size / 1e9:.2f} GB")


def benchmark_performance(input_path: str, n_samples: int = 10000):
    """
    Benchmark tokenization performance with different numbers of workers.
    
    Args:
        input_path: Path to input JSONL file
        n_samples: Number of samples to benchmark with
    """
    import tempfile
    import shutil
    
    max_workers = mp.cpu_count()
    worker_counts = [1, 2, 4, 8, max_workers] if max_workers >= 8 else [1, 2, max_workers]
    worker_counts = [w for w in worker_counts if w <= max_workers]
    
    results = []
    
    for num_workers in worker_counts:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            start_time = time.time()
            
            pretokenize_dataset_parallel(
                input_path=input_path,
                output_dir=temp_dir,
                n_puzzles=n_samples,
                num_workers=num_workers,
                batch_size=1000
            )
            
            elapsed_time = time.time() - start_time
            samples_per_second = n_samples / elapsed_time
            
            results.append({
                'workers': num_workers,
                'time': elapsed_time,
                'samples_per_second': samples_per_second
            })
            
            logging.info(f"Workers: {num_workers}, Time: {elapsed_time:.2f}s, "
                        f"Samples/s: {samples_per_second:.0f}")
    
    # Print summary
    print("\n" + "="*50)
    print("Benchmark Results:")
    print("="*50)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Samples/s':<15} {'Speedup':<10}")
    print("-"*50)
    
    baseline_time = results[0]['time']
    for result in results:
        speedup = baseline_time / result['time']
        print(f"{result['workers']:<10} {result['time']:<12.2f} "
              f"{result['samples_per_second']:<15.0f} {speedup:<10.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Parallel pre-tokenization of Knights and Knaves dataset')
    parser.add_argument('--input', type=str, default='./data/n_2.jsonl', help='Input JSONL file')
    parser.add_argument('--output', type=str, default='./data/tokenized_parallel', help='Output directory')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for parallel processing')
    parser.add_argument('--n_puzzles', type=int, default=None, help='Number of puzzles (None for all)')
    parser.add_argument('--train_ratio', type=float, default=0.98, help='Train/val split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=None, 
                       help='Number of worker processes (None for auto-detect)')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_performance(args.input, n_samples=10000)
    else:
        pretokenize_dataset_parallel(
            input_path=args.input,
            output_dir=args.output,
            max_length=args.max_length,
            batch_size=args.batch_size,
            n_puzzles=args.n_puzzles,
            train_ratio=args.train_ratio,
            seed=args.seed,
            num_workers=args.num_workers
        )


if __name__ == '__main__':
    main()