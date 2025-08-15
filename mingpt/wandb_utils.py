"""
Utilities for Weights & Biases logging and monitoring
"""

import torch
import psutil
import GPUtil
import wandb


def get_system_metrics():
    """Get comprehensive system metrics for monitoring"""
    metrics = {}
    
    # CPU metrics
    metrics["system/cpu_percent"] = psutil.cpu_percent(interval=0.1)
    metrics["system/cpu_freq_mhz"] = psutil.cpu_freq().current if psutil.cpu_freq() else 0
    
    # Memory metrics
    memory = psutil.virtual_memory()
    metrics["system/memory_used_gb"] = memory.used / 1024**3
    metrics["system/memory_percent"] = memory.percent
    metrics["system/memory_available_gb"] = memory.available / 1024**3
    
    # GPU metrics (if available)
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        
        # PyTorch CUDA metrics
        metrics["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(gpu_id) / 1024**3
        metrics["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(gpu_id) / 1024**3
        
        # Try to get more detailed GPU metrics using GPUtil
        try:
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                gpu = gpus[gpu_id]
                metrics["system/gpu_utilization_percent"] = gpu.load * 100
                metrics["system/gpu_memory_utilization_percent"] = gpu.memoryUtil * 100
                metrics["system/gpu_temperature_c"] = gpu.temperature
                metrics["system/gpu_power_draw_w"] = gpu.powerDraw if gpu.powerDraw else 0
        except:
            pass
    
    # Disk I/O metrics
    try:
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics["system/disk_read_mb_s"] = disk_io.read_bytes / 1024**2
            metrics["system/disk_write_mb_s"] = disk_io.write_bytes / 1024**2
    except:
        pass
    
    return metrics


def log_model_gradients(model, step):
    """Log gradient statistics to wandb"""
    gradients = []
    gradient_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            gradients.append(grad.flatten())
            gradient_norms.append(grad.norm().item())
    
    if gradients:
        all_gradients = torch.cat(gradients)
        
        wandb.log({
            "gradients/mean": all_gradients.mean().item(),
            "gradients/std": all_gradients.std().item(),
            "gradients/max": all_gradients.max().item(),
            "gradients/min": all_gradients.min().item(),
            "gradients/norm_mean": torch.tensor(gradient_norms).mean().item(),
            "gradients/norm_max": max(gradient_norms),
            "gradients/norm_min": min(gradient_norms),
        }, step=step)


def log_model_weights(model, step):
    """Log weight statistics to wandb"""
    weights = []
    weight_norms = []
    
    for name, param in model.named_parameters():
        if param.data is not None:
            weight = param.data
            weights.append(weight.flatten())
            weight_norms.append(weight.norm().item())
    
    if weights:
        all_weights = torch.cat(weights)
        
        wandb.log({
            "weights/mean": all_weights.mean().item(),
            "weights/std": all_weights.std().item(),
            "weights/max": all_weights.max().item(),
            "weights/min": all_weights.min().item(),
            "weights/norm_mean": torch.tensor(weight_norms).mean().item(),
            "weights/norm_max": max(weight_norms),
            "weights/norm_min": min(weight_norms),
        }, step=step)