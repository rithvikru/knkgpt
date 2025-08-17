"""
Utilities for training and evaluation.
"""
import os
import torch
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import json


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    val_loss: float,
    checkpoint_path: str,
    config: Optional[Dict] = None
):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss,
    }
    
    if config is not None:
        checkpoint['config'] = config
        
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    return checkpoint


def plot_losses(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Val Loss', alpha=0.8)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss plot to {save_path}")
    else:
        plt.show()
        
    plt.close()


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_puzzle_accuracy(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tokenizer,
    device: str = 'cpu',
    max_samples: int = 100
) -> Dict[str, float]:
    """
    Evaluate the model's accuracy on solving puzzles.
    """
    model.eval()
    
    correct_solutions = 0
    total_puzzles = 0
    
    # Track accuracy by solution type
    solution_stats = {}
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if total_puzzles >= max_samples:
                break
                
            inputs = batch['input'].to(device)
            full_seq = batch['full_sequence']
            
            batch_size = inputs.size(0)
            
            for j in range(batch_size):
                if total_puzzles >= max_samples:
                    break
                    
                # Get the puzzle part (everything before solution)
                puzzle_part, solution_part = tokenizer.get_puzzle_solution_split(
                    full_seq[j].tolist()
                )
                
                # Generate solution
                puzzle_tensor = torch.tensor(puzzle_part, device=device).unsqueeze(0)
                generated = model.generate(
                    puzzle_tensor,
                    max_new_tokens=10,  # Solutions are short
                    temperature=0.1,
                    do_sample=False
                )
                
                # Extract generated solution
                generated_tokens = generated[0, len(puzzle_part):].tolist()
                
                # Decode solutions
                true_solution = tokenizer.decode(solution_part).replace('<EOS>', '').strip()
                true_solution = ''.join([c for c in true_solution if c in ['K', 'N']])
                
                pred_solution = tokenizer.decode(generated_tokens)
                
                # Extract just the K/N characters
                pred_solution_chars = ''.join([c for c in pred_solution if c in ['K', 'N']])[:len(true_solution)]
                
                # Check if correct (only if we have a true solution)
                if true_solution:
                    is_correct = pred_solution_chars == true_solution
                    if is_correct:
                        correct_solutions += 1
                else:
                    # Skip if no solution found
                    total_puzzles -= 1
                    continue
                    
                # Track by solution type
                if true_solution not in solution_stats:
                    solution_stats[true_solution] = {'correct': 0, 'total': 0}
                solution_stats[true_solution]['total'] += 1
                if is_correct:
                    solution_stats[true_solution]['correct'] += 1
                    
                total_puzzles += 1
    
    # Calculate accuracies
    overall_accuracy = correct_solutions / total_puzzles if total_puzzles > 0 else 0
    
    results = {
        'overall_accuracy': overall_accuracy,
        'correct_solutions': correct_solutions,
        'total_puzzles': total_puzzles,
        'solution_type_accuracy': {}
    }
    
    for sol_type, stats in solution_stats.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        results['solution_type_accuracy'][sol_type] = {
            'accuracy': acc,
            'correct': stats['correct'],
            'total': stats['total']
        }
    
    return results


def print_puzzle_examples(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tokenizer,
    device: str = 'cpu',
    n_examples: int = 5
):
    """Print example puzzle solutions."""
    model.eval()
    
    examples_shown = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if examples_shown >= n_examples:
                break
                
            inputs = batch['input'].to(device)
            full_seq = batch['full_sequence']
            
            batch_size = inputs.size(0)
            
            for j in range(min(batch_size, n_examples - examples_shown)):
                # Get the puzzle part
                puzzle_part, solution_part = tokenizer.get_puzzle_solution_split(
                    full_seq[j].tolist()
                )
                
                # Generate solution
                puzzle_tensor = torch.tensor(puzzle_part, device=device).unsqueeze(0)
                generated = model.generate(
                    puzzle_tensor,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False
                )
                
                # Extract generated solution
                generated_tokens = generated[0, len(puzzle_part):].tolist()
                
                # Decode everything
                puzzle_text = tokenizer.decode(puzzle_part).replace('<SOS>', '').strip()
                true_solution_raw = tokenizer.decode(solution_part)
                true_solution = ''.join([c for c in true_solution_raw if c in ['K', 'N']])
                pred_solution = tokenizer.decode(generated_tokens)
                pred_solution_chars = ''.join([c for c in pred_solution if c in ['K', 'N']])[:len(true_solution)]
                
                print(f"\n{'='*80}")
                print(f"Example {examples_shown + 1}:")
                print(f"Puzzle: {puzzle_text[:200]}..." if len(puzzle_text) > 200 else f"Puzzle: {puzzle_text}")
                print(f"True Solution: {true_solution}")
                print(f"Predicted Solution: {pred_solution_chars}")
                print(f"Correct: {'✓' if pred_solution_chars == true_solution else '✗'}")
                
                examples_shown += 1
