"""
Dataset classes for Knights and Knaves puzzles.
"""
import torch
from torch.utils.data import Dataset
import jsonlines
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.knights_knaves.tokenizer import KnightsKnavesTokenizer


class KnightsKnavesDataset(Dataset):
    """Dataset for Knights and Knaves puzzles."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[KnightsKnavesTokenizer] = None,
        max_length: int = 512,
        n_puzzles: Optional[int] = None,
        split: str = 'train',
        train_ratio: float = 0.98,
        seed: int = 42,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSONL data file
            tokenizer: KnightsKnavesTokenizer instance (creates new if None)
            max_length: Maximum sequence length
            n_puzzles: Number of puzzles to load (None for all)
            split: 'train' or 'val'
            train_ratio: Ratio of data to use for training
            seed: Random seed for reproducible splits
        """
        self.data_path = data_path
        self.tokenizer = tokenizer or KnightsKnavesTokenizer()
        self.max_length = max_length
        self.split = split
        
        # Load and encode data
        print(f"Loading {split} data from {data_path}...")
        self.data = self._load_data(n_puzzles, train_ratio, seed)
        print(f"Loaded {len(self.data)} {split} examples")
        
    def _load_data(
        self, 
        n_puzzles: Optional[int], 
        train_ratio: float,
        seed: int
    ) -> List[Dict[str, torch.Tensor]]:
        """Load and encode the data."""
        data = []
        
        # Set random seed for reproducible splits
        random.seed(seed)
        
        # Read all puzzles first to determine split
        all_puzzles = []
        with jsonlines.open(self.data_path) as reader:
            for i, line in enumerate(tqdm(reader, desc="Reading puzzles")):
                if n_puzzles and i >= n_puzzles:
                    break
                all_puzzles.append(line)
        
        # Shuffle and split
        random.shuffle(all_puzzles)
        n_train = int(len(all_puzzles) * train_ratio)
        
        if self.split == 'train':
            puzzles_to_use = all_puzzles[:n_train]
        else:  # validation
            puzzles_to_use = all_puzzles[n_train:]
        
        # Encode puzzles
        for item in tqdm(puzzles_to_use, desc=f"Encoding {self.split} puzzles"):
            puzzle = item['puzzle']
            solution = item['solution']
            
            # Encode the example
            encoded = self.tokenizer.encode_example(puzzle, solution)
            
            # Pad sequence
            encoded = self.tokenizer.pad_sequence(encoded, self.max_length)
            
            # Convert to tensor
            encoded_tensor = torch.tensor(encoded, dtype=torch.long)
            
            # Create input and target
            # Input: everything except the last token
            # Target: everything except the first token
            input_seq = encoded_tensor[:-1]
            target_seq = encoded_tensor[1:]
            
            data.append({
                'input': input_seq,
                'target': target_seq,
                'full_sequence': encoded_tensor,
            })
            
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size
    
    def get_block_size(self) -> int:
        return self.max_length - 1  # -1 because we predict next token


class KnightsKnavesDataModule:
    """Data module for managing train/val datasets."""
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        max_length: int = 512,
        n_puzzles: Optional[int] = None,
        num_workers: int = 4,
        train_ratio: float = 0.98,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_puzzles = n_puzzles
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Create shared tokenizer
        self.tokenizer = KnightsKnavesTokenizer()
        
        # Create datasets
        self.train_dataset = KnightsKnavesDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=max_length,
            n_puzzles=n_puzzles,
            split='train',
            train_ratio=train_ratio,
            seed=seed,
        )
        
        self.val_dataset = KnightsKnavesDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=max_length,
            n_puzzles=n_puzzles,
            split='val',
            train_ratio=train_ratio,
            seed=seed,
        )
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size
    
    def get_block_size(self) -> int:
        return self.max_length - 1


def test_dataset():
    """Test the dataset with a small sample."""
    # Use a small number of puzzles for testing
    dataset = KnightsKnavesDataset(
        data_path="../../data/n_2.jsonl",
        max_length=256,
        n_puzzles=100,
        split='train',
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.get_vocab_size()}")
    print(f"Block size: {dataset.get_block_size()}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"Input: {sample['input'].shape}")
    print(f"Target: {sample['target'].shape}")
    print(f"Full sequence: {sample['full_sequence'].shape}")
    
    # Decode the sample
    print(f"\nDecoded sample:")
    decoded = dataset.tokenizer.decode(sample['full_sequence'].tolist())
    print(decoded[:200] + "..." if len(decoded) > 200 else decoded)


if __name__ == "__main__":
    test_dataset()
