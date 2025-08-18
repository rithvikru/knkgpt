"""
GPT model for Knights and Knaves puzzles.
Adapted from minGPT (https://github.com/karpathy/minGPT) and Othello-GPT.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 29  # KnK tokenizer vocabulary size
    n_positions: int = 512  # maximum sequence length
    n_embd: int = 512  # embedding dimension
    n_layer: int = 8  # number of transformer blocks
    n_head: int = 8  # number of attention heads
    n_inner: Optional[int] = None  # inner dimension of FFN, defaults to 4 * n_embd
    activation_function: str = "gelu"
    resid_pdrop: float = 0.1  # residual dropout
    embd_pdrop: float = 0.1  # embedding dropout
    attn_pdrop: float = 0.1  # attention dropout
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    
    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
            

class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention layer."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Causal mask to ensure attention is only applied to the left
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions))
                                     .view(1, 1, config.n_positions, config.n_positions))
        
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
        
        
class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.activation = nn.GELU() if config.activation_function == "gelu" else nn.ReLU()
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        
        
class Block(nn.Module):
    """Transformer block: attention + feed forward."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        
        
class GPT(nn.Module):
    """GPT Language Model for Knights and Knaves."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Input embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params/1e6:.2f}M")
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, 
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                return_hidden_states: bool = False) -> dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            labels: Target token IDs for language modeling [batch_size, sequence_length]
            return_hidden_states: Whether to return all hidden states
            
        Returns:
            Dictionary with loss (if labels provided), logits, and optionally hidden states
        """
        device = input_ids.device
        B, T = input_ids.size()
        assert T <= self.config.n_positions, f"Sequence length {T} exceeds maximum {self.config.n_positions}"
        
        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        tok_emb = self.wte(input_ids)  # [B, T, n_embd]
        pos_emb = self.wpe(pos)  # [1, T, n_embd]
        x = self.drop(tok_emb + pos_emb)
        
        # Store hidden states if requested
        all_hidden_states = [x] if return_hidden_states else None
        
        # Transformer blocks
        for block in self.h:
            x = block(x)
            if return_hidden_states:
                all_hidden_states.append(x)
                
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            # Replace padding token ids in labels with -100
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:].contiguous()
                shift_labels = shift_labels.masked_fill(shift_attention_mask == 0, -100)
                
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': all_hidden_states
        }
        
    def generate(self, 
                 input_ids: torch.LongTensor,
                 max_new_tokens: int = 10,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.LongTensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Initial sequence [batch_size, sequence_length]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
            
        Returns:
            Generated token IDs [batch_size, sequence_length + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Get predictions for last token
            outputs = self(input_ids)
            logits = outputs['logits']
            logits = logits[:, -1, :] / temperature
            
            # Optionally apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
                
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
        return input_ids
        
        
def create_model(vocab_size: int = 29) -> GPT:
    """Create a 25M parameter GPT model for Knights and Knaves."""
    config = GPTConfig(
        vocab_size=vocab_size,
        n_positions=512,
        n_embd=512,
        n_layer=8,
        n_head=8,
        n_inner=2048,  # 4 * n_embd
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    return GPT(config)


def test_model():
    """Test the model with dummy data."""
    import numpy as np
    
    print("Creating model...")
    model = create_model()
    
    # Create dummy batch
    batch_size = 4
    seq_length = 128
    vocab_size = 29
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Test generation
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=10)
    print(f"\nGenerated shape: {generated.shape}")
    
    
if __name__ == "__main__":
    test_model()
