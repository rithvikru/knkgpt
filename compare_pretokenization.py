#!/usr/bin/env python3
"""
Compare performance of different pretokenization implementations.
"""
import os
import time
import argparse
import tempfile
import shutil
import subprocess
import psutil
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import json


def run_pretokenization(script_name: str, input_path: str, output_dir: str, 
                       n_puzzles: int, num_workers: int = None) -> Dict:
    """Run a pretokenization script and measure performance."""
    
    # Build command
    cmd = [
        'python', script_name,
        '--input', input_path,
        '--output', output_dir,
        '--n_puzzles', str(n_puzzles)
    ]
    
    if num_workers is not None and script_name != 'pretokenize_dataset.py':
        cmd.extend(['--num_workers', str(num_workers)])
    
    # Monitor CPU usage
    cpu_percentages = []
    memory_usage = []
    
    # Start process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc_psutil = psutil.Process(process.pid)
    
    start_time = time.time()
    
    # Monitor while running
    while process.poll() is None:
        try:
            # Get CPU usage for this process and its children
            cpu_percent = proc_psutil.cpu_percent(interval=0.1)
            with proc_psutil.oneshot():
                for child in proc_psutil.children(recursive=True):
                    cpu_percent += child.cpu_percent(interval=0.1)
            
            cpu_percentages.append(cpu_percent)
            
            # Get memory usage
            mem_info = proc_psutil.memory_info()
            memory_usage.append(mem_info.rss / 1e9)  # GB
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        time.sleep(0.1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Get output
    stdout, stderr = process.communicate()
    
    # Calculate statistics
    avg_cpu = np.mean(cpu_percentages) if cpu_percentages else 0
    max_cpu = np.max(cpu_percentages) if cpu_percentages else 0
    avg_memory = np.mean(memory_usage) if memory_usage else 0
    max_memory = np.max(memory_usage) if memory_usage else 0
    
    return {
        'elapsed_time': elapsed_time,
        'samples_per_second': n_puzzles / elapsed_time,
        'avg_cpu_percent': avg_cpu,
        'max_cpu_percent': max_cpu,
        'avg_memory_gb': avg_memory,
        'max_memory_gb': max_memory,
        'cpu_efficiency': avg_cpu / (psutil.cpu_count() * 100),  # Fraction of total CPU used
        'stdout': stdout.decode('utf-8'),
        'stderr': stderr.decode('utf-8')
    }


def compare_implementations(input_path: str, n_puzzles: int = 10000):
    """Compare all three implementations."""
    
    implementations = [
        ('Original (Sequential)', 'pretokenize_dataset.py', None),
        ('Parallel (Basic)', 'pretokenize_dataset_parallel.py', None),
        ('Optimized (Shared Memory)', 'pretokenize_dataset_optimized.py', None),
    ]
    
    # Also test with different worker counts for parallel versions
    cpu_count = psutil.cpu_count(logical=False)
    worker_counts = [1, 2, 4, cpu_count // 2, cpu_count]
    worker_counts = [w for w in worker_counts if w <= cpu_count and w > 0]
    worker_counts = list(set(worker_counts))  # Remove duplicates
    worker_counts.sort()
    
    results = {}
    
    print(f"Comparing implementations with {n_puzzles} samples...")
    print(f"System has {psutil.cpu_count(logical=True)} logical cores, "
          f"{cpu_count} physical cores\n")
    
    # Run each implementation
    for name, script, workers in implementations:
        if not os.path.exists(script):
            print(f"Skipping {name} - script not found: {script}")
            continue
            
        print(f"Running {name}...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                result = run_pretokenization(script, input_path, temp_dir, n_puzzles, workers)
                results[name] = result
                
                print(f"  Time: {result['elapsed_time']:.2f}s")
                print(f"  Samples/s: {result['samples_per_second']:.0f}")
                print(f"  CPU usage: {result['avg_cpu_percent']:.1f}% avg, "
                      f"{result['max_cpu_percent']:.1f}% max")
                print(f"  Memory: {result['avg_memory_gb']:.2f}GB avg, "
                      f"{result['max_memory_gb']:.2f}GB max")
                print(f"  CPU efficiency: {result['cpu_efficiency']*100:.1f}%\n")
                
            except Exception as e:
                print(f"  Error: {e}\n")
    
    # Test different worker counts for parallel versions
    print("\nTesting different worker counts...")
    
    for script_name in ['pretokenize_dataset_parallel.py', 'pretokenize_dataset_optimized.py']:
        if not os.path.exists(script_name):
            continue
            
        script_type = 'Parallel' if 'parallel' in script_name else 'Optimized'
        worker_results = {}
        
        for workers in worker_counts:
            print(f"  {script_type} with {workers} workers...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    result = run_pretokenization(script_name, input_path, temp_dir, 
                                               n_puzzles // 2, workers)  # Use fewer samples
                    worker_results[workers] = result
                    print(f"    Samples/s: {result['samples_per_second']:.0f}")
                    
                except Exception as e:
                    print(f"    Error: {e}")
        
        results[f'{script_type}_workers'] = worker_results
    
    return results


def plot_results(results: Dict, output_dir: str = '.'):
    """Create visualization of results."""
    
    # Create plots directory
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # 1. Overall comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Filter main implementations
    main_results = {k: v for k, v in results.items() if 'workers' not in k}
    
    if main_results:
        names = list(main_results.keys())
        
        # Samples per second
        ax = axes[0, 0]
        samples_per_sec = [main_results[n]['samples_per_second'] for n in names]
        ax.bar(names, samples_per_sec)
        ax.set_ylabel('Samples/second')
        ax.set_title('Processing Speed')
        ax.tick_params(axis='x', rotation=45)
        
        # CPU usage
        ax = axes[0, 1]
        cpu_usage = [main_results[n]['avg_cpu_percent'] for n in names]
        ax.bar(names, cpu_usage)
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('Average CPU Usage')
        ax.tick_params(axis='x', rotation=45)
        
        # Memory usage
        ax = axes[1, 0]
        memory_usage = [main_results[n]['max_memory_gb'] for n in names]
        ax.bar(names, memory_usage)
        ax.set_ylabel('Memory (GB)')
        ax.set_title('Peak Memory Usage')
        ax.tick_params(axis='x', rotation=45)
        
        # CPU efficiency
        ax = axes[1, 1]
        cpu_efficiency = [main_results[n]['cpu_efficiency'] * 100 for n in names]
        ax.bar(names, cpu_efficiency)
        ax.set_ylabel('CPU Efficiency (%)')
        ax.set_title('CPU Utilization Efficiency')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'comparison.png'))
    plt.close()
    
    # 2. Worker scaling plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (impl_type, ax) in enumerate([('Parallel', axes[0]), ('Optimized', axes[1])]):
        key = f'{impl_type}_workers'
        if key in results and results[key]:
            worker_counts = sorted(results[key].keys())
            samples_per_sec = [results[key][w]['samples_per_second'] for w in worker_counts]
            
            ax.plot(worker_counts, samples_per_sec, 'o-', label='Actual')
            
            # Add ideal scaling line
            if worker_counts and samples_per_sec:
                base_speed = samples_per_sec[0] if worker_counts[0] == 1 else samples_per_sec[0] / worker_counts[0]
                ideal_scaling = [base_speed * w for w in worker_counts]
                ax.plot(worker_counts, ideal_scaling, '--', label='Ideal scaling')
            
            ax.set_xlabel('Number of Workers')
            ax.set_ylabel('Samples/second')
            ax.set_title(f'{impl_type} Implementation Scaling')
            ax.legend()
            ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'worker_scaling.png'))
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/plots/")


def generate_report(results: Dict, output_path: str):
    """Generate a detailed report of the comparison."""
    
    with open(output_path, 'w') as f:
        f.write("# Pretokenization Performance Comparison Report\n\n")
        
        f.write(f"System Information:\n")
        f.write(f"- CPU cores: {psutil.cpu_count(logical=True)} logical, "
                f"{psutil.cpu_count(logical=False)} physical\n")
        f.write(f"- Total memory: {psutil.virtual_memory().total / 1e9:.2f} GB\n")
        f.write(f"- CPU model: {subprocess.check_output(['uname', '-p']).decode().strip()}\n\n")
        
        # Main results
        main_results = {k: v for k, v in results.items() if 'workers' not in k}
        
        if main_results:
            f.write("## Implementation Comparison\n\n")
            
            # Find baseline (original)
            baseline_name = None
            baseline_time = None
            for name in main_results:
                if 'Original' in name:
                    baseline_name = name
                    baseline_time = main_results[name]['elapsed_time']
                    break
            
            f.write("| Implementation | Time (s) | Samples/s | Speedup | CPU Usage | Memory (GB) | CPU Efficiency |\n")
            f.write("|----------------|----------|-----------|---------|-----------|-------------|----------------|\n")
            
            for name, result in main_results.items():
                speedup = baseline_time / result['elapsed_time'] if baseline_time else 1.0
                f.write(f"| {name} | {result['elapsed_time']:.2f} | "
                       f"{result['samples_per_second']:.0f} | {speedup:.2f}x | "
                       f"{result['avg_cpu_percent']:.1f}% | {result['max_memory_gb']:.2f} | "
                       f"{result['cpu_efficiency']*100:.1f}% |\n")
        
        # Worker scaling results
        f.write("\n## Worker Scaling Analysis\n\n")
        
        for impl_type in ['Parallel', 'Optimized']:
            key = f'{impl_type}_workers'
            if key in results and results[key]:
                f.write(f"\n### {impl_type} Implementation\n\n")
                f.write("| Workers | Samples/s | Speedup | Efficiency |\n")
                f.write("|---------|-----------|---------|------------|\n")
                
                worker_results = results[key]
                worker_counts = sorted(worker_results.keys())
                
                if worker_counts:
                    base_speed = worker_results[1]['samples_per_second'] if 1 in worker_results else None
                    if base_speed is None and worker_counts:
                        base_speed = worker_results[worker_counts[0]]['samples_per_second'] / worker_counts[0]
                    
                    for workers in worker_counts:
                        speed = worker_results[workers]['samples_per_second']
                        speedup = speed / base_speed if base_speed else 1.0
                        efficiency = speedup / workers if workers > 0 else 0
                        
                        f.write(f"| {workers} | {speed:.0f} | {speedup:.2f}x | {efficiency*100:.1f}% |\n")
        
        f.write("\n## Recommendations\n\n")
        
        # Find best implementation
        if main_results:
            best_impl = max(main_results.items(), key=lambda x: x[1]['samples_per_second'])
            f.write(f"- **Best implementation**: {best_impl[0]} "
                   f"({best_impl[1]['samples_per_second']:.0f} samples/s)\n")
            
            # CPU efficiency recommendation
            max_efficiency = max(r['cpu_efficiency'] for r in main_results.values())
            if max_efficiency < 0.8:
                f.write("- Consider investigating I/O bottlenecks - CPU efficiency is below 80%\n")
            
            # Memory usage
            max_memory = max(r['max_memory_gb'] for r in main_results.values())
            available_memory = psutil.virtual_memory().available / 1e9
            if max_memory > available_memory * 0.8:
                f.write(f"- High memory usage detected ({max_memory:.2f}GB). "
                       f"Consider reducing batch size.\n")
    
    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare pretokenization implementations')
    parser.add_argument('--input', type=str, default='./data/n_2.jsonl', help='Input JSONL file')
    parser.add_argument('--n_puzzles', type=int, default=10000, help='Number of puzzles to test with')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Run comparison
    results = compare_implementations(args.input, args.n_puzzles)
    
    # Save raw results
    results_path = os.path.join(args.output_dir, 'pretokenization_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                json_results[k] = {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating)) else v2 
                                  for k2, v2 in v.items() if k2 not in ['stdout', 'stderr']}
            else:
                json_results[k] = v
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Generate plots
    plot_results(results, args.output_dir)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'pretokenization_report.md')
    generate_report(results, report_path)


if __name__ == '__main__':
    main()