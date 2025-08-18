"""
Custom tokenizer for Knights and Knaves puzzles.
"""
import re
from typing import List, Dict, Tuple, Optional
import json


class KnightsKnavesTokenizer:
    """Tokenizer for Knights and Knaves logical puzzles."""
    
    def __init__(self):
        # Define the vocabulary
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        
        # Logical operators
        self.operators = ['and', 'or', 'not', 'iff', 'imp', 'tt', 'ff']
        
        # Functions
        self.functions = ['isKnight', 'isKnave', 'says']
        
        # Islanders
        self.islanders = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        # Parentheses and comma
        self.delimiters = ['(', ')', ',']
        
        # Solution tokens
        self.solution_tokens = ['K', 'N']
        
        # Build vocabulary
        self.vocab = (self.special_tokens + self.operators + self.functions + 
                     self.islanders + self.delimiters + self.solution_tokens)
        
        # Create token to index mapping
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Special token indices
        self.pad_idx = self.token2idx['<PAD>']
        self.sos_idx = self.token2idx['<SOS>']
        self.eos_idx = self.token2idx['<EOS>']
        self.unk_idx = self.token2idx['<UNK>']
        
        self.vocab_size = len(self.vocab)
        
    def tokenize_puzzle(self, puzzle_str: str) -> List[str]:
        """Tokenize a puzzle string into a list of tokens."""
        # Add spaces around parentheses and commas for easier splitting
        puzzle_str = re.sub(r'([(),])', r' \1 ', puzzle_str)
        
        # Split by whitespace
        tokens = puzzle_str.split()
        
        # Clean up tokens
        clean_tokens = []
        for token in tokens:
            if token:  # Skip empty tokens
                clean_tokens.append(token)
                
        return clean_tokens
    
    def tokenize_solution(self, solution: str) -> List[str]:
        """Tokenize a solution string (e.g., 'KN' -> ['K', 'N'])."""
        return list(solution)
    
    def encode_puzzle(self, puzzle_str: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a puzzle string into token indices."""
        tokens = self.tokenize_puzzle(puzzle_str)
        
        if add_special_tokens:
            tokens = ['<SOS>'] + tokens
            
        # Convert tokens to indices
        indices = []
        for token in tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                indices.append(self.unk_idx)
                
        return indices
    
    def encode_solution(self, solution: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a solution string into token indices."""
        tokens = self.tokenize_solution(solution)
        
        if add_special_tokens:
            tokens = tokens + ['<EOS>']
            
        # Convert tokens to indices
        indices = []
        for token in tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                indices.append(self.unk_idx)
                
        return indices
    
    def encode_example(self, puzzle_str: str, solution_str: str) -> List[int]:
        """Encode a complete example (puzzle + solution) for training."""
        puzzle_indices = self.encode_puzzle(puzzle_str, add_special_tokens=True)
        solution_indices = self.encode_solution(solution_str, add_special_tokens=False)
        
        # Combine puzzle and solution
        return puzzle_indices + solution_indices
    
    def encode(self, text: str) -> List[int]:
        """Generic encode method that tries to detect if input is puzzle or solution."""
        # Simple heuristic: if text contains function names, it's a puzzle
        if any(func in text for func in self.functions):
            return self.encode_puzzle(text)
        else:
            # Assume it's a solution
            return self.encode_solution(text)
    
    def decode(self, indices: List[int]) -> str:
        """Decode a list of token indices back to text."""
        tokens = []
        for idx in indices:
            if idx < len(self.idx2token):
                tokens.append(self.idx2token[idx])
            else:
                tokens.append('<UNK>')
                
        # Join tokens with spaces, but handle parentheses nicely
        text = ' '.join(tokens)
        # Remove spaces around parentheses and commas
        text = re.sub(r'\s*([(),])\s*', r'\1', text)
        
        return text
    
    def get_puzzle_solution_split(self, indices: List[int]) -> Tuple[List[int], List[int]]:
        """Split encoded sequence into puzzle and solution parts."""
        # Find where solution starts (look for first K or N token)
        solution_start = None
        
        for i in range(len(indices)):
            token = self.idx2token.get(indices[i], '<UNK>')
            if token in ['K', 'N']:
                solution_start = i
                break
        
        if solution_start is None:
            # No solution found, everything is puzzle
            return indices, []
            
        puzzle_indices = indices[:solution_start]
        solution_indices = indices[solution_start:]
        
        return puzzle_indices, solution_indices
    
    def pad_sequence(self, indices: List[int], max_length: int) -> List[int]:
        """Pad or truncate sequence to fixed length."""
        if len(indices) >= max_length:
            return indices[:max_length]
        else:
            return indices + [self.pad_idx] * (max_length - len(indices))


def test_tokenizer():
    """Test the tokenizer with a sample puzzle."""
    tokenizer = KnightsKnavesTokenizer()
    
    # Test puzzle from the dataset
    test_puzzle = "says 0 (iff (isKnight 0) (isKnave 0)), says 1 (tt), says 0 (ff)"
    test_solution = "NK"
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {tokenizer.vocab[:20]}...")  # Show first 20 tokens
    
    # Test tokenization
    puzzle_tokens = tokenizer.tokenize_puzzle(test_puzzle)
    print(f"\nPuzzle tokens: {puzzle_tokens}")
    
    solution_tokens = tokenizer.tokenize_solution(test_solution)
    print(f"Solution tokens: {solution_tokens}")
    
    # Test encoding
    encoded = tokenizer.encode_example(test_puzzle, test_solution)
    print(f"\nEncoded sequence: {encoded}")
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Test splitting
    puzzle_part, solution_part = tokenizer.get_puzzle_solution_split(encoded)
    print(f"\nPuzzle part: {tokenizer.decode(puzzle_part)}")
    print(f"Solution part: {tokenizer.decode(solution_part)}")


if __name__ == "__main__":
    test_tokenizer()
