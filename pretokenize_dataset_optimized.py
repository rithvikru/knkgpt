#!/usr/bin/env python3
"""
Highly optimized parallel pre-tokenization using shared memory and advanced techniques.
Maximizes CPU core utilization with minimal overhead.
"""
import os
import argparse
import numpy as np
import jsonlines
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager, RawArray, shared_memory
from functools import partial
import time
import logging
import ctypes
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue as ThreadQueue
import psutil

from data.knights_knaves.tokenizer import KnightsKnavesTokenizer


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Global variables for shared memory access in worker processes
shared_train_input = None
shared_train_target = None
shared_train_mask = None
shared_val_input = None
shared_val_target = None
shared_val_mask = None
shared_shape_info = None


def init_worker(train_input_raw, train_target_raw, train_mask_raw,
                val_input_raw, val_target_raw, val_mask_raw,
                shape_dict):
    """Initialize worker process with shared memory arrays."""
    global shared_train_input, shared_train_target, shared_train_mask
    global shared_val_input, shared_val_target, shared_val_mask, shared_shape_info
    
    shared_train_input = train_input_raw
    shared_train_target = train_target_raw
    shared_train_mask = train_mask_raw
    shared_val_input = val_input_raw
    shared_val_target = val_target_raw
    shared_val_mask = val_mask_raw
    shared_shape_info = shape_dict


def tokenize_and_write_batch(batch_info: Tuple[int, List[Tuple[int, dict, bool, int]]],
                            tokenizer_pkl: bytes,
                            max_length: int) -> int:
    """
    Tokenize a batch and write directly to shared memory.
    
    Args:
        batch_info: Tuple of (batch_id, [(global_idx, item, is_train, write_idx), ...])
        tokenizer_pkl: Pickled tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Number of items processed
    """
    batch_id, batch_data = batch_info
    
    # Unpickle tokenizer
    tokenizer = pickle.loads(tokenizer_pkl)
    
    # Get shared memory as numpy arrays
    n_train = shared_shape_info['n_train']
    n_val = shared_shape_info['n_val']
    seq_len = shared_shape_info['seq_len']
    
    # Create numpy views of shared memory
    train_input = np.frombuffer(shared_train_input, dtype=np.int32).reshape(n_train, seq_len)
    train_target = np.frombuffer(shared_train_target, dtype=np.int32).reshape(n_train, seq_len)
    train_mask = np.frombuffer(shared_train_mask, dtype=np.uint8).reshape(n_train, seq_len)
    
    val_input = np.frombuffer(shared_val_input, dtype=np.int32).reshape(n_val, seq_len)
    val_target = np.frombuffer(shared_val_target, dtype=np.int32).reshape(n_val, seq_len)
    val_mask = np.frombuffer(shared_val_mask, dtype=np.uint8).reshape(n_val, seq_len)
    
    processed = 0
    
    for global_idx, item, is_train, write_idx in batch_data:
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
        attention_mask = np.array([1 if tok != tokenizer.pad_idx else 0 for tok in input_seq], dtype=np.uint8)
        
        # Write directly to shared memory
        if is_train:
            train_input[write_idx] = input_seq
            train_target[write_idx] = target_seq
            train_mask[write_idx] = attention_mask
        else:
            val_input[write_idx] = input_seq
            val_target[write_idx] = target_seq
            val_mask[write_idx] = attention_mask
        
        processed += 1
    
    return processed


def create_shared_arrays(n_train: int, n_val: int, seq_len: int) -> Tuple[RawArray, ...]:
    """Create shared memory arrays for train and validation data."""
    # Create shared memory arrays
    train_input_raw = RawArray(ctypes.c_int32, n_train * seq_len)
    train_target_raw = RawArray(ctypes.c_int32, n_train * seq_len)
    train_mask_raw = RawArray(ctypes.c_uint8, n_train * seq_len)
    
    val_input_raw = RawArray(ctypes.c_int32, n_val * seq_len)
    val_target_raw = RawArray(ctypes.c_int32, n_val * seq_len)
    val_mask_raw = RawArray(ctypes.c_uint8, n_val * seq_len)
    
    return (train_input_raw, train_target_raw, train_mask_raw,
            val_input_raw, val_target_raw, val_mask_raw)


def pretokenize_dataset_optimized(
    input_path: str,
    output_dir: str,
    max_length: int = 512,
    batch_size: int = 1000,
    n_puzzles: int = None,
    train_ratio: float = 0.98,
    seed: int = 42,
    num_workers: int = None,
    prefetch_batches: int = 2
):
    """
    Highly optimized pre-tokenization using shared memory and prefetching.
    
    Args:
        input_path: Path to input JSONL file
        output_dir: Directory to save tokenized data
        max_length: Maximum sequence length
        batch_size: Number of examples per batch for parallel processing
        n_puzzles: Number of puzzles to use (None for all)
        train_ratio: Train/val split ratio
        seed: Random seed for train/val split
        num_workers: Number of worker processes (None for auto-detect)
        prefetch_batches: Number of batches to prefetch
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = KnightsKnavesTokenizer()
    tokenizer_pkl = pickle.dumps(tokenizer)
    
    # Determine number of workers
    if num_workers is None:
        # Use physical cores, not logical cores (hyperthreading)
        num_workers = psutil.cpu_count(logical=False)
        if num_workers is None:
            num_workers = mp.cpu_count()
    
    logging.info(f"Using {num_workers} worker processes")
    logging.info(f"System has {psutil.cpu_count(logical=True)} logical cores, "
                f"{psutil.cpu_count(logical=False)} physical cores")
    
    # First pass: count and split
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
    n_val = total_count - n_train
    train_indices = set(indices[:n_train])
    
    logging.info(f"Train examples: {n_train:,}")
    logging.info(f"Val examples: {n_val:,}")
    
    # Create shared memory arrays
    seq_len = max_length - 1
    shared_arrays = create_shared_arrays(n_train, n_val, seq_len)
    train_input_raw, train_target_raw, train_mask_raw, val_input_raw, val_target_raw, val_mask_raw = shared_arrays
    
    shape_info = {
        'n_train': n_train,
        'n_val': n_val,
        'seq_len': seq_len
    }
    
    # Create index mappings
    train_write_indices = {}
    val_write_indices = {}
    train_idx = 0
    val_idx = 0
    
    for idx in range(total_count):
        if idx in train_indices:
            train_write_indices[idx] = train_idx
            train_idx += 1
        else:
            val_write_indices[idx] = val_idx
            val_idx += 1
    
    # Prepare batches with prefetching
    logging.info("Preparing batches...")
    
    def batch_generator():
        """Generate batches with write indices."""
        batch_buffer = []
        batch_id = 0
        
        with jsonlines.open(input_path) as reader:
            for global_idx, item in enumerate(reader):
                if n_puzzles and global_idx >= n_puzzles:
                    break
                
                is_train = global_idx in train_indices
                write_idx = train_write_indices[global_idx] if is_train else val_write_indices[global_idx]
                
                batch_buffer.append((global_idx, item, is_train, write_idx))
                
                if len(batch_buffer) >= batch_size:
                    yield (batch_id, batch_buffer)
                    batch_buffer = []
                    batch_id += 1
            
            if batch_buffer:
                yield (batch_id, batch_buffer)
    
    # Count total batches
    total_batches = (total_count + batch_size - 1) // batch_size
    
    # Process with pool
    logging.info(f"Processing {total_batches} batches in parallel...")
    
    process_func = partial(
        tokenize_and_write_batch,
        tokenizer_pkl=tokenizer_pkl,
        max_length=max_length
    )
    
    # Use pool with initializer for shared memory
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(*shared_arrays, shape_info)
    ) as pool:
        
        # Process batches with progress bar
        total_processed = 0
        batch_gen = batch_generator()
        
        # Use imap for streaming processing
        with tqdm(total=total_count, desc="Tokenizing") as pbar:
            for processed in pool.imap(process_func, batch_gen, chunksize=1):
                total_processed += processed
                pbar.update(processed)
    
    logging.info(f"Processed {total_processed:,} examples")
    
    # Convert shared memory to memory-mapped files
    logging.info("Saving to disk...")
    
    # Create numpy views of shared memory
    train_input = np.frombuffer(train_input_raw, dtype=np.int32).reshape(n_train, seq_len)
    train_target = np.frombuffer(train_target_raw, dtype=np.int32).reshape(n_train, seq_len)
    train_mask = np.frombuffer(train_mask_raw, dtype=np.uint8).reshape(n_train, seq_len)
    
    val_input = np.frombuffer(val_input_raw, dtype=np.int32).reshape(n_val, seq_len)
    val_target = np.frombuffer(val_target_raw, dtype=np.int32).reshape(n_val, seq_len)
    val_mask = np.frombuffer(val_mask_raw, dtype=np.uint8).reshape(n_val, seq_len)
    
    # Save as memory-mapped files
    train_input_mmap = np.memmap(
        os.path.join(output_dir, 'train_input_ids.npy'),
        dtype=np.int32, mode='w+', shape=(n_train, seq_len)
    )
    train_input_mmap[:] = train_input
    del train_input_mmap
    
    train_target_mmap = np.memmap(
        os.path.join(output_dir, 'train_target_ids.npy'),
        dtype=np.int32, mode='w+', shape=(n_train, seq_len)
    )
    train_target_mmap[:] = train_target
    del train_target_mmap
    
    train_mask_mmap = np.memmap(
        os.path.join(output_dir, 'train_attention_mask.npy'),
        dtype=np.bool_, mode='w+', shape=(n_train, seq_len)
    )
    train_mask_mmap[:] = train_mask.astype(np.bool_)
    del train_mask_mmap
    
    val_input_mmap = np.memmap(
        os.path.join(output_dir, 'val_input_ids.npy'),
        dtype=np.int32, mode='w+', shape=(n_val, seq_len)
    )
    val_input_mmap[:] = val_input
    del val_input_mmap
    
    val_target_mmap = np.memmap(
        os.path.join(output_dir, 'val_target_ids.npy'),
        dtype=np.int32, mode='w+', shape=(n_val, seq_len)
    )
    val_target_mmap[:] = val_target
    del val_target_mmap
    
    val_mask_mmap = np.memmap(
        os.path.join(output_dir, 'val_attention_mask.npy'),
        dtype=np.bool_, mode='w+', shape=(n_val, seq_len)
    )
    val_mask_mmap[:] = val_mask.astype(np.bool_)
    del val_mask_mmap
    
    # Save metadata
    metadata = {
        'vocab_size': tokenizer.vocab_size,
        'max_length': max_length,
        'n_train': n_train,
        'n_val': n_val,
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
    
    # Calculate size on disk
    total_size = 0
    for filename in os.listdir(output_dir):
        if filename.endswith('.npy'):
            filepath = os.path.join(output_dir, filename)
            total_size += os.path.getsize(filepath)
    
    logging.info(f"Total size on disk: {total_size / 1e9:.2f} GB")


def monitor_system_resources():
    """Monitor CPU and memory usage during execution."""
    process = psutil.Process()
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    
    print(f"\nSystem Resources:")
    print(f"  Total CPU cores: {cpu_count} (logical), {cpu_count_physical} (physical)")
    print(f"  CPU usage: {psutil.cpu_percent(interval=1)}%")
    print(f"  Per-core usage: {psutil.cpu_percent(interval=1, percpu=True)}")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"  Total memory: {memory.total / 1e9:.2f} GB")
    print(f"  Available memory: {memory.available / 1e9:.2f} GB")
    print(f"  Memory usage: {memory.percent}%")
    
    # Process info
    print(f"  Process CPU usage: {process.cpu_percent(interval=1)}%")
    print(f"  Process memory: {process.memory_info().rss / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='Optimized parallel pre-tokenization')
    parser.add_argument('--input', type=str, default='./data/n_2.jsonl', help='Input JSONL file')
    parser.add_argument('--output', type=str, default='./data/tokenized_optimized', help='Output directory')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for parallel processing')
    parser.add_argument('--n_puzzles', type=int, default=None, help='Number of puzzles (None for all)')
    parser.add_argument('--train_ratio', type=float, default=0.98, help='Train/val split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=None, 
                       help='Number of worker processes (None for auto-detect physical cores)')
    parser.add_argument('--monitor', action='store_true', help='Monitor system resources')
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_system_resources()
        return
    
    pretokenize_dataset_optimized(
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