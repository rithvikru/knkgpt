"""
Optimized dataset for pre-tokenized Knights and Knaves puzzles.
Uses memory-mapped arrays for efficient loading.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from typing import Dict, Optional


class PreTokenizedKnightsKnavesDataset(Dataset):
    """Dataset that loads pre-tokenized Knights and Knaves puzzles from disk."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        device: str = 'cpu',
        pin_memory: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing pre-tokenized data
            split: 'train' or 'val'
            device: Device to load tensors to ('cpu' or 'cuda')
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.data_dir = data_dir
        self.split = split
        self.device = device
        self.pin_memory = pin_memory and device == 'cuda'
        
        # Load metadata
        with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load tokenizer
        with open(os.path.join(data_dir, 'tokenizer.pkl'), 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Memory-map the arrays
        prefix = 'train' if split == 'train' else 'val'
        n_examples = self.metadata['n_train'] if split == 'train' else self.metadata['n_val']
        max_length = self.metadata['max_length']
        
        self.input_ids = np.memmap(
            os.path.join(data_dir, f'{prefix}_input_ids.npy'),
            dtype=np.int32,
            mode='r',
            shape=(n_examples, max_length - 1)
        )
        
        self.target_ids = np.memmap(
            os.path.join(data_dir, f'{prefix}_target_ids.npy'),
            dtype=np.int32,
            mode='r',
            shape=(n_examples, max_length - 1)
        )
        
        self.attention_mask = np.memmap(
            os.path.join(data_dir, f'{prefix}_attention_mask.npy'),
            dtype=np.bool_,
            mode='r',
            shape=(n_examples, max_length - 1)
        )
        
        self.n_examples = n_examples
        print(f"Loaded {split} dataset with {n_examples:,} examples")
        
        # Pre-allocate pinned memory buffers if using CUDA
        if self.pin_memory:
            self._create_pinned_buffers()
    
    def _create_pinned_buffers(self):
        """Create pinned memory buffers for faster GPU transfer."""
        seq_len = self.input_ids.shape[1]
        
        # Allocate pinned memory
        self.pinned_input = torch.empty(
            (seq_len,), dtype=torch.long, pin_memory=True
        )
        self.pinned_target = torch.empty(
            (seq_len,), dtype=torch.long, pin_memory=True
        )
        self.pinned_mask = torch.empty(
            (seq_len,), dtype=torch.bool, pin_memory=True
        )
    
    def __len__(self) -> int:
        return self.n_examples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load from memory-mapped arrays
        input_ids = self.input_ids[idx]
        target_ids = self.target_ids[idx]
        attention_mask = self.attention_mask[idx]
        
        if self.pin_memory:
            # Copy to pinned memory first
            self.pinned_input.copy_(torch.from_numpy(input_ids))
            self.pinned_target.copy_(torch.from_numpy(target_ids))
            self.pinned_mask.copy_(torch.from_numpy(attention_mask))
            
            # Then transfer to GPU (this is now async)
            input_tensor = self.pinned_input.to(self.device, non_blocking=True)
            target_tensor = self.pinned_target.to(self.device, non_blocking=True)
            mask_tensor = self.pinned_mask.to(self.device, non_blocking=True)
        else:
            # Direct conversion
            input_tensor = torch.from_numpy(input_ids.astype(np.int64))
            target_tensor = torch.from_numpy(target_ids.astype(np.int64))
            mask_tensor = torch.from_numpy(attention_mask)
        
        # For compatibility with existing code, also include full_sequence
        # (reconstruct from input and last target token)
        full_sequence = torch.cat([
            input_tensor,
            target_tensor[-1:] if len(target_tensor) > 0 else torch.tensor([self.tokenizer.pad_idx])
        ])
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'attention_mask': mask_tensor,
            'full_sequence': full_sequence,
        }
    
    def get_vocab_size(self) -> int:
        return self.metadata['vocab_size']
    
    def get_block_size(self) -> int:
        return self.metadata['max_length'] - 1


class PreTokenizedDataModule:
    """Data module for managing pre-tokenized train/val datasets."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = 'cuda',
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ):
        """
        Initialize the data module.
        
        Args:
            data_dir: Directory containing pre-tokenized data
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            device: Device to load data to
            pin_memory: Whether to use pinned memory
            prefetch_factor: Number of batches to prefetch per worker
            persistent_workers: Keep workers alive between epochs
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.pin_memory = pin_memory and device == 'cuda'
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers and num_workers > 0
        
        # Load metadata
        with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load tokenizer
        with open(os.path.join(data_dir, 'tokenizer.pkl'), 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Create datasets
        self.train_dataset = PreTokenizedKnightsKnavesDataset(
            data_dir=data_dir,
            split='train',
            device='cpu',  # DataLoader handles GPU transfer
            pin_memory=False,  # DataLoader handles pinning
        )
        
        self.val_dataset = PreTokenizedKnightsKnavesDataset(
            data_dir=data_dir,
            split='val',
            device='cpu',
            pin_memory=False,
        )
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=True,  # Drop last incomplete batch for stable training
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
    
    def get_vocab_size(self) -> int:
        return self.metadata['vocab_size']
    
    def get_block_size(self) -> int:
        return self.metadata['max_length'] - 1


def benchmark_loading_speed():
    """Benchmark the loading speed of the pre-tokenized dataset."""
    import time
    
    data_dir = './data/tokenized'
    if not os.path.exists(data_dir):
        print(f"Pre-tokenized data not found at {data_dir}")
        print("Run pretokenize_dataset.py first!")
        return
    
    # Create data module
    data_module = PreTokenizedDataModule(
        data_dir=data_dir,
        batch_size=64,
        num_workers=4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # Benchmark loading speed
    train_loader = data_module.train_dataloader()
    
    print("Benchmarking loading speed...")
    start_time = time.time()
    
    n_batches = 100
    for i, batch in enumerate(train_loader):
        if i >= n_batches:
            break
        # Ensure GPU transfer is complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    samples_per_sec = n_batches * data_module.batch_size / elapsed
    
    print(f"Loaded {n_batches} batches in {elapsed:.2f}s")
    print(f"Throughput: {samples_per_sec:.0f} samples/sec")


if __name__ == '__main__':
    benchmark_loading_speed()
