"""
Custom tokenizer for Knights and Knaves puzzles.

This tokenizer handles logical expressions with operators like iff, imp, and, or, not,
as well as functions like isKnight, isKnave, says, and constants tt (true), ff (false).
"""

import re
from typing import List, Dict, Tuple, Optional
import json


class KnKTokenizer:
    """Tokenizer for Knights and Knaves logical expressions."""
    
    def __init__(self):
        # Define special tokens
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"  # start of sequence
        self.eos_token = "<eos>"  # end of sequence
        self.unk_token = "<unk>"  # unknown token
        
        # Define logical operators and keywords
        self.operators = ["iff", "imp", "and", "or", "not"]
        self.functions = ["isKnight", "isKnave", "says"]
        self.constants = ["tt", "ff"]
        self.punctuation = ["(", ")", ","]
        
        # Build vocabulary
        self._build_vocab()
        
        # Compile regex patterns for tokenization
        self._compile_patterns()
        
    def _build_vocab(self):
        """Build vocabulary from all possible tokens."""
        # Start with special tokens
        vocab_list = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        
        # Add operators, functions, constants
        vocab_list.extend(self.operators)
        vocab_list.extend(self.functions)
        vocab_list.extend(self.constants)
        vocab_list.extend(self.punctuation)
        
        # Add numbers 0-9 (for islander IDs and potential larger puzzles)
        vocab_list.extend([str(i) for i in range(10)])
        
        # Add solution tokens
        vocab_list.extend(["N", "K"])  # Knave, Knight
        
        # Create token to ID mapping
        self.token2id = {token: idx for idx, token in enumerate(vocab_list)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        
        self.vocab_size = len(self.token2id)
        self.pad_id = self.token2id[self.pad_token]
        self.sos_id = self.token2id[self.sos_token]
        self.eos_id = self.token2id[self.eos_token]
        self.unk_id = self.token2id[self.unk_token]
        
    def _compile_patterns(self):
        """Compile regex patterns for tokenization."""
        # Pattern for multi-character tokens (operators, functions, constants)
        multi_char_tokens = self.operators + self.functions + self.constants
        # Sort by length (longest first) to ensure proper matching
        multi_char_tokens.sort(key=len, reverse=True)
        
        # Create pattern that matches multi-character tokens or single characters
        pattern_parts = [re.escape(token) for token in multi_char_tokens]
        pattern_parts.extend([r'\d', r'\(', r'\)', r',', r'\s+'])
        
        self.token_pattern = re.compile('|'.join(pattern_parts))
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize a logical expression into a list of tokens."""
        tokens = []
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Find all tokens using regex
        pos = 0
        while pos < len(text):
            # Try to match multi-character tokens first
            match = None
            for token in self.operators + self.functions + self.constants:
                if text[pos:].startswith(token):
                    tokens.append(token)
                    pos += len(token)
                    match = True
                    break
            
            if not match:
                # Single character token
                char = text[pos]
                if char in '(),0123456789':
                    tokens.append(char)
                elif char.isspace():
                    pass  # Skip whitespace
                else:
                    # Unknown character, skip it
                    pass
                pos += 1
                
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text into token IDs."""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.sos_token] + tokens + [self.eos_token]
            
        # Convert tokens to IDs
        ids = []
        for token in tokens:
            if token in self.token2id:
                ids.append(self.token2id[token])
            else:
                ids.append(self.unk_id)
                
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for idx in ids:
            if idx in self.id2token:
                token = self.id2token[idx]
                if skip_special_tokens and token in [self.pad_token, self.sos_token, self.eos_token]:
                    continue
                tokens.append(token)
                
        # Join tokens with appropriate spacing
        text = ""
        for i, token in enumerate(tokens):
            if i > 0 and token not in "(),":
                text += " "
            text += token
            
        return text
    
    def encode_puzzle_solution_pair(self, puzzle: str, solution: str) -> Tuple[List[int], List[int]]:
        """Encode a puzzle and its solution for training."""
        # Encode puzzle
        puzzle_ids = self.encode(puzzle, add_special_tokens=True)
        
        # Encode solution (NK format)
        solution_tokens = list(solution)  # Split "NK" into ["N", "K"]
        solution_ids = [self.token2id[token] for token in solution_tokens if token in self.token2id]
        
        return puzzle_ids, solution_ids
    
    def batch_encode(self, texts: List[str], max_length: Optional[int] = None, 
                     padding: bool = True) -> Dict[str, List[List[int]]]:
        """Batch encode multiple texts with optional padding."""
        encoded = [self.encode(text) for text in texts]
        
        if max_length is None and padding:
            max_length = max(len(seq) for seq in encoded)
            
        if padding and max_length:
            # Pad sequences to max_length
            padded = []
            for seq in encoded:
                if len(seq) < max_length:
                    seq = seq + [self.pad_id] * (max_length - len(seq))
                else:
                    seq = seq[:max_length]
                padded.append(seq)
            encoded = padded
            
        return {"input_ids": encoded}


def test_tokenizer():
    """Test the tokenizer with sample puzzles."""
    tokenizer = KnKTokenizer()
    
    # Test samples
    samples = [
        "says 0 (iff (isKnight 0) (isKnave 0)), says 1 (tt)",
        "says 0 (and (isKnight 0) (ff)), says 1 (or (isKnight 0) (ff))",
        "says 1 (says 1 (isKnave 0))"
    ]
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Token to ID mapping sample: {dict(list(tokenizer.token2id.items())[:20])}")
    
    for sample in samples:
        print(f"\nOriginal: {sample}")
        tokens = tokenizer.tokenize(sample)
        print(f"Tokens: {tokens}")
        encoded = tokenizer.encode(sample)
        print(f"Encoded: {encoded}")
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    test_tokenizer()
