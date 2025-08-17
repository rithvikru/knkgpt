#!/usr/bin/env python3
"""
Benchmark data loading performance: original vs pre-tokenized dataset.
"""
import time
import torch
import argparse
import os
from tqdm import tqdm

from mingpt.dataset import KnightsKnavesDataModule
from mingpt.pretokenized_dataset import PreTokenizedDataModule


def benchmark_dataloader(dataloader, name, n_batches=100, device='cuda'):
    """Benchmark a dataloader's performance."""
    print(f"\nBenchmarking {name}...")
    
    # Warmup
    print("Warming up...")
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        if device == 'cuda' and torch.cuda.is_available():
            # Move to GPU to simulate real training
            batch['input'].to(device, non_blocking=True)
            batch['target'].to(device, non_blocking=True)
            torch.cuda.synchronize()
    
    # Actual benchmark
    print(f"Running benchmark ({n_batches} batches)...")
    start_time = time.time()
    
    for i, batch in enumerate(tqdm(dataloader, total=n_batches)):
        if i >= n_batches:
            break
        
        if device == 'cuda' and torch.cuda.is_available():
            # Move to GPU
            batch['input'].to(device, non_blocking=True)
            batch['target'].to(device, non_blocking=True)
            torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    batch_size = batch['input'].shape[0]
    total_samples = n_batches * batch_size
    samples_per_sec = total_samples / elapsed
    ms_per_batch = (elapsed / n_batches) * 1000
    
    print(f"\nResults for {name}:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Samples/sec: {samples_per_sec:.0f}")
    print(f"  ms/batch: {ms_per_batch:.1f}")
    print(f"  Throughput improvement factor: {samples_per_sec / 1000:.1f}x (vs 1k baseline)")
    
    return {
        'name': name,
        'elapsed': elapsed,
        'samples_per_sec': samples_per_sec,
        'ms_per_batch': ms_per_batch,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark data loading performance')
    parser.add_argument('--data_path', type=str, default='./data/n_2.jsonl', help='Path to raw data')
    parser.add_argument('--tokenized_dir', type=str, default='./data/tokenized', help='Path to pre-tokenized data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--n_batches', type=int, default=100, help='Number of batches to benchmark')
    parser.add_argument('--n_puzzles', type=int, default=10000, help='Number of puzzles for original dataset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print(f"Benchmarking configuration:")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Batches to test: {args.n_batches}")
    
    results = []
    
    # Benchmark original dataset (if we limit puzzles to make it reasonable)
    if args.n_puzzles and args.n_puzzles <= 100000:
        print(f"\n{'='*60}")
        print(f"Creating original dataset with {args.n_puzzles} puzzles...")
        
        original_module = KnightsKnavesDataModule(
            data_path=args.data_path,
            batch_size=args.batch_size,
            max_length=512,
            n_puzzles=args.n_puzzles,
            num_workers=args.num_workers,
            train_ratio=0.98,
            seed=42,
        )
        
        original_loader = original_module.train_dataloader()
        result = benchmark_dataloader(
            original_loader, 
            f"Original Dataset ({args.n_puzzles} puzzles)",
            n_batches=min(args.n_batches, len(original_loader)),
            device=args.device
        )
        results.append(result)
    
    # Benchmark pre-tokenized dataset
    if os.path.exists(args.tokenized_dir):
        print(f"\n{'='*60}")
        print(f"Loading pre-tokenized dataset from {args.tokenized_dir}...")
        
        pretokenized_module = PreTokenizedDataModule(
            data_dir=args.tokenized_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
        
        pretokenized_loader = pretokenized_module.train_dataloader()
        result = benchmark_dataloader(
            pretokenized_loader,
            "Pre-tokenized Dataset",
            n_batches=args.n_batches,
            device=args.device
        )
        results.append(result)
    else:
        print(f"\nPre-tokenized data not found at {args.tokenized_dir}")
        print("Run pretokenize_dataset.py first!")
    
    # Compare results
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Performance Comparison:")
        print(f"{'Dataset':<30} {'Samples/sec':>12} {'ms/batch':>10} {'Speedup':>8}")
        print("-" * 60)
        
        baseline_speed = results[0]['samples_per_sec']
        for result in results:
            speedup = result['samples_per_sec'] / baseline_speed
            print(f"{result['name']:<30} {result['samples_per_sec']:>12.0f} "
                  f"{result['ms_per_batch']:>10.1f} {speedup:>8.1f}x")
    
    # Memory usage estimate
    if os.path.exists(args.tokenized_dir):
        print(f"\n{'='*60}")
        print("Memory Usage:")
        
        # Check file sizes
        total_size = 0
        for filename in os.listdir(args.tokenized_dir):
            if filename.endswith('.npy'):
                filepath = os.path.join(args.tokenized_dir, filename)
                size = os.path.getsize(filepath)
                total_size += size
                print(f"  {filename}: {size / 1e9:.2f} GB")
        
        print(f"  Total: {total_size / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
