from einops import rearrange
from torch import nn
import torch

from model.layers.activations import attention
from model.layers.linear import Linear
from model.layers.position import RoPE

class CausalMultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # Dimension of each head 
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model should be divisble by number of heads"
        self.w_q = Linear(in_features=d_model, out_features=d_model, device=device)
        self.w_k = Linear(in_features=d_model, out_features=d_model, device=device)
        self.w_v = Linear(in_features=d_model, out_features=d_model, device=device)
        self.w_o = Linear(in_features=d_model, out_features=d_model, device=device)
        self.rope = RoPE(
            theta=theta,
            max_seq_len=max_seq_len,
            d_k=self.d_k,
            device=device
        )
        self.device = device

    def forward(self, x, token_positions) -> torch.Tensor:
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Rearrange to multihead
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        # Rotate Q and K
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        
        seq_len = V.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))
        attention_scores = rearrange(attention(Q, K, V, mask), "b h s d -> b s (h d)", h=self.num_heads)
        return self.w_o(attention_scores)

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # Dimension of each head 
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model should be divisble by number of heads"
        self.w_q = Linear(in_features=d_model, out_features=d_model)
        self.w_k = Linear(in_features=d_model, out_features=d_model)
        self.w_v = Linear(in_features=d_model, out_features=d_model)
        self.w_o = Linear(in_features=d_model, out_features=d_model)
        

    def forward(self, x) -> torch.Tensor:
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Rearrange to multihead
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)
        
        seq_len = V.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        attention_scores = rearrange(attention(Q, K, V, mask), "b h s d -> b s (h d)", h=self.num_heads)
        return self.w_o(attention_scores)
