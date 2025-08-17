"""
Trainer for Knights and Knaves GPT.
"""
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, Any, Callable
import time

from .utils import save_checkpoint, evaluate_puzzle_accuracy, print_puzzle_examples
from .wandb_utils import log_metrics, log_puzzle_examples as log_puzzle_examples_wandb


class TrainerConfig:
    """Configuration for training."""
    # Optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    # Learning rate schedule
    lr_decay = True
    warmup_tokens = 375e6  # 375 million tokens
    final_tokens = 100e9   # 100 billion tokens
    # Checkpointing
    ckpt_dir = './ckpts/knkgpt'
    save_every = 1000
    # Logging
    log_every = 10
    eval_every = 500
    eval_samples = 1000
    # System
    num_workers = 4
    device = 'cuda'
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    """Trainer for GPT model."""
    
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Create checkpoint directory
        os.makedirs(config.ckpt_dir, exist_ok=True)
        
        # Take over whatever gpus are on the system
        self.device = config.device
        if self.device == 'cuda':
            self.model = self.model.cuda()
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs")
                self.model = DataParallel(self.model)
                
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        
        # Set up optimizer
        self.optimizer = self.raw_model.configure_optimizers(config)
        
        # Training state
        self.tokens = 0  # Counter for processed tokens
        self.global_step = 0
        self.current_epoch = 0
        
    def train(self):
        """Main training loop."""
        model, config = self.model, self.config
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        
        model.train()
        best_val_loss = float('inf')
        
        # Training metrics
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        
        for epoch in range(config.max_epochs):
            self.current_epoch = epoch
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.max_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # Get data
                x = batch['input'].to(self.device)
                y = batch['target'].to(self.device)
                
                # Forward pass
                logits, loss = model(x, y)
                epoch_losses.append(loss.item())
                
                # Backward pass
                model.zero_grad()
                loss.backward()
                if config.grad_norm_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                
                # Decay learning rate if configured
                if config.lr_decay:
                    self.tokens += (y >= 0).sum()  # Count non-padding tokens
                    if self.tokens < config.warmup_tokens:
                        # Linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # Cosine decay
                        progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
                    
                    lr = config.learning_rate * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{lr:.2e}",
                    'tokens': f"{self.tokens/1e6:.1f}M"
                })
                
                # Log metrics
                if self.global_step % config.log_every == 0:
                    log_metrics({
                        'train/loss': loss.item(),
                        'train/lr': lr,
                        'train/tokens': self.tokens,
                        'train/epoch': epoch + batch_idx / len(train_loader),
                    }, step=self.global_step)
                
                # Evaluation
                if self.global_step % config.eval_every == 0:
                    val_loss = self.evaluate(val_loader)
                    val_losses.append(val_loss)
                    
                    # Evaluate puzzle accuracy
                    eval_results = evaluate_puzzle_accuracy(
                        self.raw_model,
                        val_loader,
                        self.train_dataset.tokenizer,
                        device=self.device,
                        max_samples=config.eval_samples
                    )
                    
                    log_metrics({
                        'val/loss': val_loss,
                        'val/accuracy': eval_results['overall_accuracy'],
                        'val/correct': eval_results['correct_solutions'],
                        'val/total': eval_results['total_puzzles'],
                    }, step=self.global_step)
                    
                    # Log accuracy by solution type
                    for sol_type, stats in eval_results['solution_type_accuracy'].items():
                        log_metrics({
                            f'val/accuracy_{sol_type}': stats['accuracy'],
                        }, step=self.global_step)
                    
                    print(f"\nVal Loss: {val_loss:.4f}, Accuracy: {eval_results['overall_accuracy']:.2%}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            self.raw_model,
                            self.optimizer,
                            epoch,
                            self.global_step,
                            val_loss,
                            os.path.join(config.ckpt_dir, 'best_model.pt'),
                            config=config.__dict__
                        )
                    
                    model.train()
                
                # Save checkpoint
                if self.global_step % config.save_every == 0:
                    save_checkpoint(
                        self.raw_model,
                        self.optimizer,
                        epoch,
                        self.global_step,
                        epoch_losses[-1] if epoch_losses else 0,
                        os.path.join(config.ckpt_dir, f'checkpoint_{self.global_step}.pt'),
                        config=config.__dict__
                    )
                
                self.global_step += 1
            
            # End of epoch
            train_losses.extend(epoch_losses)
            
            # Print examples
            print(f"\n{'='*80}")
            print(f"End of Epoch {epoch + 1} - Example Predictions:")
            print_puzzle_examples(
                self.raw_model,
                val_loader,
                self.train_dataset.tokenizer,
                device=self.device,
                n_examples=3
            )
            
        # Training complete
        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed/3600:.2f} hours")
        
        # Save final model
        save_checkpoint(
            self.raw_model,
            self.optimizer,
            self.current_epoch,
            self.global_step,
            val_losses[-1] if val_losses else 0,
            os.path.join(config.ckpt_dir, 'final_model.pt'),
            config=config.__dict__
        )
        
        return train_losses, val_losses
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate model on validation set."""
        self.model.eval()
        losses = []
        
        for batch in dataloader:
            x = batch['input'].to(self.device)
            y = batch['target'].to(self.device)
            
            logits, loss = self.model(x, y)
            losses.append(loss.item())
            
        return np.mean(losses)
