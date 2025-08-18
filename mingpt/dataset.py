"""
Dataset classes for Knights and Knaves puzzles.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from .tokenizer import KnKTokenizer
except ImportError:
    from tokenizer import KnKTokenizer


class KnightsKnavesDataset(Dataset):
    """Dataset for Knights and Knaves puzzles."""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: Optional[KnKTokenizer] = None,
                 max_length: int = 512,
                 train: bool = True,
                 train_split: float = 0.95,
                 seed: int = 42,
                 max_samples: Optional[int] = None):
        """
        Args:
            data_path: Path to JSONL file containing puzzles
            tokenizer: KnKTokenizer instance (creates new if None)
            max_length: Maximum sequence length
            train: Whether this is training or validation set
            train_split: Fraction of data to use for training
            seed: Random seed for train/val split
            max_samples: Maximum number of samples to load (None for all)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer or KnKTokenizer()
        self.max_length = max_length
        self.train = train
        self.train_split = train_split
        self.seed = seed
        
        # Load and prepare data
        self._load_data(max_samples)
        
    def _load_data(self, max_samples: Optional[int] = None):
        """Load puzzles from JSONL file."""
        print(f"Loading data from {self.data_path}...")
        
        puzzles = []
        solutions = []
        
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="Loading puzzles")):
                if max_samples and i >= max_samples:
                    break
                    
                data = json.loads(line.strip())
                puzzles.append(data['puzzle'])
                solutions.append(data['solution'])
        
        # Convert to numpy arrays for easier splitting
        puzzles = np.array(puzzles)
        solutions = np.array(solutions)
        
        # Create train/val split
        np.random.seed(self.seed)
        n_samples = len(puzzles)
        indices = np.random.permutation(n_samples)
        
        split_idx = int(n_samples * self.train_split)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        if self.train:
            self.puzzles = puzzles[train_indices]
            self.solutions = solutions[train_indices]
            print(f"Loaded {len(self.puzzles)} training samples")
        else:
            self.puzzles = puzzles[val_indices]
            self.solutions = solutions[val_indices]
            print(f"Loaded {len(self.puzzles)} validation samples")
            
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get a single puzzle-solution pair."""
        puzzle = self.puzzles[idx]
        solution = self.solutions[idx]
        
        # Encode puzzle and solution
        puzzle_ids, solution_ids = self.tokenizer.encode_puzzle_solution_pair(puzzle, solution)
        
        # Combine puzzle and solution for autoregressive training
        # Format: <sos> puzzle_tokens <eos> solution_tokens
        input_ids = puzzle_ids + solution_ids
        
        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            
        # Create labels (shifted by 1 for next-token prediction)
        # We want to predict: puzzle_tokens[1:] + <eos> + solution_tokens
        labels = input_ids[1:] + [self.tokenizer.eos_id]
        
        # Pad to max_length
        input_len = len(input_ids)
        if input_len < self.max_length:
            pad_len = self.max_length - input_len
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [self.tokenizer.pad_id] * pad_len
        else:
            labels = labels[:self.max_length]
            
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if i < input_len else 0 for i in range(self.max_length)]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'puzzle_len': len(puzzle_ids),  # Store where puzzle ends
        }


class KnightsKnavesDataModule:
    """Data module for managing train/val dataloaders."""
    
    def __init__(self,
                 data_path: str,
                 batch_size: int = 32,
                 max_length: int = 512,
                 num_workers: int = 4,
                 train_split: float = 0.95,
                 seed: int = 42,
                 max_samples: Optional[int] = None):
        """Initialize data module."""
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.train_split = train_split
        self.seed = seed
        self.max_samples = max_samples
        
        # Create shared tokenizer
        self.tokenizer = KnKTokenizer()
        
        # Create datasets
        self.train_dataset = KnightsKnavesDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=max_length,
            train=True,
            train_split=train_split,
            seed=seed,
            max_samples=max_samples
        )
        
        self.val_dataset = KnightsKnavesDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=max_length,
            train=False,
            train_split=train_split,
            seed=seed,
            max_samples=max_samples
        )
        
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from tokenizer."""
        return self.tokenizer.vocab_size


def test_dataset():
    """Test the dataset loading."""
    # Test with a small subset
    data_path = "/Users/rithvikr/projects/trainloop/knkgpt/data/n_2.jsonl"
    
    print("Testing dataset loading...")
    dm = KnightsKnavesDataModule(
        data_path=data_path,
        batch_size=4,
        max_samples=100  # Only load 100 samples for testing
    )
    
    print(f"\nVocabulary size: {dm.vocab_size}")
    print(f"Train dataset size: {len(dm.train_dataset)}")
    print(f"Val dataset size: {len(dm.val_dataset)}")
    
    # Test one batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    
    # Decode first example
    first_input = batch['input_ids'][0].tolist()
    first_puzzle_len = batch['puzzle_len'][0].item()
    
    print(f"\nFirst example:")
    print(f"Full sequence: {dm.tokenizer.decode(first_input)}")
    print(f"Puzzle part: {dm.tokenizer.decode(first_input[:first_puzzle_len])}")
    print(f"Solution part: {dm.tokenizer.decode(first_input[first_puzzle_len:])}")


if __name__ == "__main__":
    test_dataset()
