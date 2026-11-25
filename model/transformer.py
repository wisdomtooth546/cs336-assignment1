from torch import nn
import torch

from model.layers.activations import softmax
from model.layers.attentionhead import CausalMultiHeadSelfAttentionRoPE
from model.layers.embedding import Embedding
from model.layers.ffn import SwiGLUFFN
from model.layers.norm import RMSLayerNorm
from model.layers.linear import Linear


class PostNormTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device = None,
        d_type = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.device = device
        self.max_seq_len = max_seq_len
        self.layer_norm_1 = RMSLayerNorm(d_model, device=device)
        self.layer_norm_2 = RMSLayerNorm(d_model, device=device)
        self.ffn = SwiGLUFFN(d_model, d_ff, device, d_type)
        self.attn = CausalMultiHeadSelfAttentionRoPE(d_model, num_heads, theta, max_seq_len, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.shape[-2], device=self.device)
        x = self.layer_norm_1(x + self.attn(x, token_positions))
        return self.layer_norm_2(x + self.ffn(x))

class PreNormTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device = None,
        d_type = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.device = device
        self.max_seq_len = max_seq_len
        self.layer_norm_1 = RMSLayerNorm(d_model, device=device)
        self.layer_norm_2 = RMSLayerNorm(d_model, device=device)
        self.ffn = SwiGLUFFN(d_model, d_ff, device, d_type)
        self.attn = CausalMultiHeadSelfAttentionRoPE(d_model, num_heads, theta, max_seq_len, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.shape[-2], device=self.device)
        x = x + self.attn(self.layer_norm_1(x), token_positions)
        return x + self.ffn(self.layer_norm_2(x))

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        theta: float,
        num_heads: int,
        device = None
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                theta=theta,
                max_seq_len=context_length,
                device=device
            ) for _ in range(num_layers)
        ])
        self.ln = RMSLayerNorm(
            d_model=d_model,
            device=device
        )
        self.linear = Linear(
            out_features=vocab_size, in_features=d_model, device=device
        )
        self.device = device
    
    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(in_indices)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.linear(self.ln(tokens))

class PostNormTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        theta: float,
        num_heads: int,
        device = None
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList([
            PostNormTransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                theta=theta,
                max_seq_len=context_length,
                device=device
            ) for _ in range(num_layers)
        ])
        self.ln = RMSLayerNorm(
            d_model=d_model,
            device=device
        )
        self.linear = Linear(
            out_features=vocab_size, in_features=d_model, device=device
        )
        self.device = device
    
    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(in_indices)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.linear(self.ln(tokens))
