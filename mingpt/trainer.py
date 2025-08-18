"""
Trainer for GPT model with wandb integration.
"""

import os
import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm

from .model import GPT, GPTConfig


@dataclass
class TrainerConfig:
    """Configuration for training."""
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_epsilon: float = 1e-8
    adam_betas: tuple = (0.9, 0.95)
    grad_norm_clip: float = 1.0
    
    # Learning rate schedule
    warmup_steps: int = 1000
    lr_decay: bool = True
    
    # Training
    max_epochs: int = 10
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Checkpointing
    ckpt_path: str = "ckpts/knkgpt"
    save_every: int = 1000  # save checkpoint every N steps
    
    # Logging
    log_every: int = 10  # log metrics every N steps
    eval_every: int = 500  # run validation every N steps
    
    # Wandb
    wandb_project: str = "knkgpt"
    wandb_name: Optional[str] = None
    wandb_notes: Optional[str] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Mixed precision
    use_amp: bool = True
    
    
class Trainer:
    """Trainer for GPT model."""
    
    def __init__(self, model: GPT, config: TrainerConfig):
        self.model = model
        self.config = config
        
        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and config.device == "cuda" else None
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup checkpoint directory
        os.makedirs(config.ckpt_path, exist_ok=True)
        
        # Initialize tracking variables
        self.global_step = 0
        self.start_epoch = 0
        
    def _create_optimizer(self):
        """Create AdamW optimizer with weight decay fix."""
        # Separate parameters that should and shouldn't have weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                    
        # Validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} are in both decay/no_decay sets"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Parameters {param_dict.keys() - union_params} were not separated into decay/no_decay"
            
        # Create optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        optimizer = AdamW(optim_groups, lr=self.config.learning_rate, 
                         betas=self.config.adam_betas, eps=self.config.adam_epsilon)
        return optimizer
        
    def _get_lr(self):
        """Get learning rate with warmup and cosine decay."""
        if not self.config.lr_decay:
            return self.config.learning_rate
            
        # Warmup
        if self.global_step < self.config.warmup_steps:
            lr_mult = self.global_step / max(1, self.config.warmup_steps)
        else:
            # Cosine decay
            progress = (self.global_step - self.config.warmup_steps) / \
                      max(1, self.total_steps - self.config.warmup_steps)
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        return self.config.learning_rate * lr_mult
        
    def _set_lr(self, lr):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'config': self.config,
            'model_config': self.model.config,
            'val_loss': val_loss,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        path = os.path.join(self.config.ckpt_path, f'checkpoint_step_{self.global_step}.pt')
        torch.save(checkpoint, path)
        
        # Also save as latest
        latest_path = os.path.join(self.config.ckpt_path, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"Loaded checkpoint from {path} (epoch {self.start_epoch}, step {self.global_step})")
        
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        # Initialize wandb
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            notes=self.config.wandb_notes,
            config={
                'model_config': self.model.config.__dict__,
                'trainer_config': self.config.__dict__,
                'n_params': sum(p.numel() for p in self.model.parameters()),
            }
        )
        
        # Calculate total steps
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        self.total_steps = steps_per_epoch * self.config.max_epochs
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.start_epoch, self.config.max_epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast() if self.scaler else torch.no_grad():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs['loss'] / self.config.gradient_accumulation_steps
                    
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.config.grad_norm_clip > 0:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                        
                    # Update weights
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad()
                    
                    # Update learning rate
                    lr = self._get_lr()
                    self._set_lr(lr)
                    
                    # Update step counter
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.log_every == 0:
                        metrics = {
                            'train/loss': loss.item() * self.config.gradient_accumulation_steps,
                            'train/lr': lr,
                            'train/epoch': epoch,
                            'train/step': self.global_step,
                        }
                        wandb.log(metrics, step=self.global_step)
                        
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                        'lr': f"{lr:.2e}"
                    })
                    
                    # Validation
                    if val_loader is not None and self.global_step % self.config.eval_every == 0:
                        val_loss = self.evaluate(val_loader)
                        wandb.log({'val/loss': val_loss}, step=self.global_step)
                        self.model.train()
                        
                    # Save checkpoint
                    if self.global_step % self.config.save_every == 0:
                        self.save_checkpoint(epoch)
                        
                epoch_loss += loss.item()
                
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / len(train_loader)
            
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, average loss: {avg_loss:.4f}")
            
            # Run validation at end of epoch
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f"Validation loss: {val_loss:.4f}")
                wandb.log({'val/loss_epoch': val_loss, 'epoch': epoch+1}, step=self.global_step)
                
            # Save checkpoint at end of epoch
            self.save_checkpoint(epoch, val_loss if val_loader else None)
            
        wandb.finish()
        
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                total_loss += outputs['loss'].item()
                
        avg_loss = total_loss / len(val_loader)
        return avg_loss
        
    def generate_samples(self, tokenizer, num_samples: int = 5, puzzle_prefix: Optional[str] = None):
        """Generate sample predictions for logging."""
        self.model.eval()
        
        samples = []
        
        with torch.no_grad():
            for i in range(num_samples):
                if puzzle_prefix:
                    # Use provided prefix
                    input_ids = torch.tensor([tokenizer.encode(puzzle_prefix)]).to(self.device)
                else:
                    # Start with just the SOS token
                    input_ids = torch.tensor([[tokenizer.sos_id]]).to(self.device)
                    
                # Generate
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.8,
                    top_k=10
                )
                
                # Decode
                generated_text = tokenizer.decode(generated[0].tolist())
                samples.append(generated_text)
                
        return samples
