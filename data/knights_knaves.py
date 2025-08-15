import json
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import os

class KnightsKnavesDataset:
    """
    Dataset for Knights and Knaves puzzles.
    Each puzzle is tokenized as a sequence for autoregressive training.
    """
    
    def __init__(self, data_path, max_games=None, val_split=0.1, seed=42):
        """
        Args:
            data_path: Path to the JSONL file containing puzzles
            max_games: Maximum number of games to load (None for all)
            val_split: Fraction of data to use for validation
            seed: Random seed for train/val split
        """
        random.seed(seed)
        
        self.puzzles = []
        self.solutions = []
        
        print(f"Loading Knights and Knaves puzzles from {data_path}...")
        
        # Count total lines for progress bar
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                total_lines = sum(1 for _ in f)
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        # Limit if max_games is specified
        if max_games is not None:
            total_lines = min(total_lines, max_games)
        
        # Load puzzles
        with open(data_path, 'r') as f:
            for i, line in enumerate(tqdm(f, total=total_lines, desc="Loading puzzles")):
                if max_games is not None and i >= max_games:
                    break
                    
                data = json.loads(line.strip())
                puzzle = data['puzzle']
                solution = data['solution']
                
                # Create a tokenized sequence: puzzle + separator + solution
                # Format: "puzzle => solution"
                sequence = puzzle + " => " + solution
                # Convert to list of characters and add padding token (-100) for compatibility
                char_sequence = list(sequence) + [-100]
                self.puzzles.append(char_sequence)
                self.solutions.append(solution)
        
        print(f"Loaded {len(self.puzzles)} puzzles")
        
        # Create train/val split
        n_val = int(len(self.puzzles) * val_split)
        indices = list(range(len(self.puzzles)))
        random.shuffle(indices)
        
        val_indices = set(indices[:n_val])
        
        self.train = []
        self.val = []
        
        for i, puzzle in enumerate(self.puzzles):
            if i in val_indices:
                self.val.append(puzzle)
            else:
                self.train.append(puzzle)
        
        print(f"Train: {len(self.train)}, Val: {len(self.val)}")
        
        # For compatibility with Othello dataset interface
        self.ood_perc = 0
    
    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, idx):
        return self.train[idx]


class KnightsKnavesTokenizer:
    """
    Tokenizer for Knights and Knaves puzzles.
    Uses character-level tokenization for simplicity and to handle all possible variations.
    """
    
    def __init__(self, dataset=None):
        # Build vocabulary from dataset if provided, otherwise use a predefined set
        if dataset is not None:
            # Extract unique characters from all puzzles
            chars = set()
            sample_size = min(10000, len(dataset.puzzles))  # Sample for efficiency
            for i in range(sample_size):
                chars.update(list(dataset.puzzles[i]))
            
            # Add special tokens
            self.vocab = ['<pad>'] + sorted(list(chars))
        else:
            # Default character set for Knights and Knaves puzzles
            chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 (),.=>!')
            self.vocab = ['<pad>'] + sorted(list(chars))
        
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.token_to_id['<pad>']
    
    def encode(self, text):
        """Encode text to token IDs using character-level tokenization."""
        return [self.token_to_id.get(char, self.pad_token_id) for char in text]
    
    def decode(self, ids):
        """Decode token IDs back to text."""
        return ''.join([self.id_to_token.get(id, '') for id in ids if id != self.pad_token_id])


def get(data_path="data/n_2.jsonl", max_games=None, val_split=0.1, seed=42):
    """
    Get Knights and Knaves dataset.
    
    Args:
        data_path: Path to the JSONL file
        max_games: Maximum number of games to load (None for all)
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        KnightsKnavesDataset instance
    """
    return KnightsKnavesDataset(data_path, max_games, val_split, seed)