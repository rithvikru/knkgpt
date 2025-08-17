# Parallel Pretokenization Guide

This guide explains how to maximize CPU core utilization when pretokenizing datasets.

## Overview

We've created three versions of the pretokenization script, each with increasing levels of optimization:

1. **Original** (`pretokenize_dataset.py`) - Sequential processing
2. **Parallel** (`pretokenize_dataset_parallel.py`) - Basic multiprocessing
3. **Optimized** (`pretokenize_dataset_optimized.py`) - Shared memory + advanced optimizations

## Key Optimizations for Maximum CPU Utilization

### 1. Multiprocessing with Proper Worker Count
```python
# Detect physical cores (not logical/hyperthreaded cores)
import psutil
num_workers = psutil.cpu_count(logical=False)
```

### 2. Shared Memory Arrays
- Eliminates data copying between processes
- Direct writing to shared memory from workers
- Significant reduction in memory overhead

### 3. Batch Processing
- Optimal batch sizes (typically 1000-5000 items)
- Reduces overhead of task distribution
- Better cache utilization

### 4. Memory-Mapped Files
- Efficient disk I/O
- Allows processing datasets larger than RAM
- Direct numpy array access

### 5. Process Pool with Initializer
- Reuses worker processes
- Shares tokenizer state efficiently
- Reduces pickle/unpickle overhead

## Usage Examples

### Basic Usage
```bash
# Using the optimized version (recommended)
python pretokenize_dataset_optimized.py --input data/n_2.jsonl --output data/tokenized

# Specify number of workers
python pretokenize_dataset_optimized.py --num_workers 16

# Process subset of data
python pretokenize_dataset_optimized.py --n_puzzles 100000
```

### Monitor System Resources
```bash
# Check CPU and memory usage
python pretokenize_dataset_optimized.py --monitor
```

### Benchmark Performance
```bash
# Compare different worker counts
python pretokenize_dataset_parallel.py --benchmark
```

## Performance Tips

1. **Use Physical Cores**: Hyperthreading often doesn't help for CPU-intensive tasks
   ```python
   num_workers = psutil.cpu_count(logical=False)
   ```

2. **Optimal Batch Size**: Start with 1000, adjust based on your data
   ```bash
   python pretokenize_dataset_optimized.py --batch_size 2000
   ```

3. **Monitor Performance**: Use system monitoring tools
   ```bash
   # In another terminal
   htop  # or top
   ```

4. **SSD vs HDD**: Use SSD for output directory for better write performance

5. **Memory Considerations**: 
   - Ensure sufficient RAM for shared memory arrays
   - Formula: `(n_train + n_val) * max_length * 4 bytes * 3` (for input, target, mask)

## Comparison of Approaches

| Feature | Original | Parallel | Optimized |
|---------|----------|----------|-----------|
| CPU Utilization | ~12% (1 core) | ~80% (all cores) | ~95% (all cores) |
| Memory Efficiency | Good | Moderate | Excellent |
| Scalability | Poor | Good | Excellent |
| Complexity | Low | Medium | High |
| Best For | Small datasets | Medium datasets | Large datasets |

## Expected Performance Gains

Based on typical hardware:
- **8-core CPU**: 6-7x speedup
- **16-core CPU**: 12-14x speedup
- **32-core CPU**: 20-25x speedup

Note: Actual speedup depends on:
- I/O speed (SSD vs HDD)
- Memory bandwidth
- Dataset characteristics
- Tokenization complexity

## Troubleshooting

### High Memory Usage
- Reduce batch size: `--batch_size 500`
- Process data in chunks: `--n_puzzles 1000000`

### Low CPU Utilization
- Check if I/O bound (slow disk)
- Increase batch size: `--batch_size 5000`
- Ensure using physical cores, not logical

### Process Hangs
- Check available memory
- Reduce number of workers
- Check for corrupted input data

## Advanced Configuration

### Custom Worker Affinity
```python
# Pin workers to specific CPU cores
import os
os.sched_setaffinity(0, {0, 1, 2, 3})  # Use cores 0-3
```

### NUMA Awareness
For multi-socket systems:
```bash
# Run on specific NUMA node
numactl --cpunodebind=0 python pretokenize_dataset_optimized.py
```

### Profile Performance
```bash
# Profile CPU usage
python -m cProfile -o profile.stats pretokenize_dataset_optimized.py
```

## Best Practices

1. **Always benchmark** with your specific data
2. **Start with defaults**, then optimize
3. **Monitor resource usage** during processing
4. **Use the optimized version** for production
5. **Keep batch sizes reasonable** (1000-5000)
6. **Ensure sufficient disk space** for output