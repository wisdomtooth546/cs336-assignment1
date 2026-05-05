import math
from typing import Optional
import torch
import einops
from torch import nn
from jaxtyping import Float, Bool

from cs336_basics.ops.position import RoPE
from cs336_basics.ops.softmax import Softmax

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
    
    def forward(
        self,
        Q: Float[torch.Tensor, "... seq_q d_k"],
        K: Float[torch.Tensor, "... seq_k d_k"],
        V: Float[torch.Tensor, "... seq_k d_v"],
        mask: Optional[Bool[torch.Tensor, "... seq_q seq_k"]] = None,
    ) -> Float[torch.Tensor, "... seq_q d_v"]:
        d_k = Q.shape[-1]
        scores = torch.einsum("... q d, ... k d -> ... q k", Q, K)
        if mask is not None:
            scores = scores.masked_fill(~mask, -torch.inf)
        attn = self.softmax(scores / math.sqrt(d_k), dim=-1)
        return torch.einsum("... q k, ... k d -> ... q d", attn, V)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: str | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError("Invalid number of heads")
        self.d_h = d_model // num_heads # head dimension
        var = 2 / (self.d_h + d_model)
        std = math.sqrt(var)
        self.w_q = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn((num_heads * self.d_h, d_model), device=device),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            ),
            requires_grad=True
        )
        self.w_k = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn((num_heads * self.d_h, d_model), device=device),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            ),
            requires_grad=True
        )
        self.w_v = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn((num_heads * self.d_h, d_model), device=device),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            ),
            requires_grad=True
        )
        # To map output of the multi head attention to d_model
        self.w_o = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn((d_model, num_heads * self.d_h), device=device),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            ),
            requires_grad=True
        )
        self._attention = ScaledDotProductAttention()
        

    def forward(self, x: Float[torch.Tensor, "... seq d_model"]):
        # Project all heads in one matmul: (..., seq, h*d_h)
        Q = x @ self.w_q.T
        K = x @ self.w_k.T
        V = x @ self.w_v.T

        # Split into heads: (..., h, seq, d_h)
        Q = einops.rearrange(Q, "... seq (h dh) -> ... h seq dh", h=self.num_heads)
        K = einops.rearrange(K, "... seq (h dh) -> ... h seq dh", h=self.num_heads)
        V = einops.rearrange(V, "... seq (h dh) -> ... h seq dh", h=self.num_heads)

        seq_len = x.shape[-2]
        # Causal mask: True means "allowed to attend"
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attention = self._attention(Q=Q, K=K, V=V, mask=mask)

        # Merge heads back: (..., seq, h*d_h)
        attention = einops.rearrange(attention, "... h seq dh -> ... seq (h dh)")
        return attention @ self.w_o.T

class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: str | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError("Invalid number of heads")
        self.d_h = d_model // num_heads # head dimension
        var = 2 / (self.d_h + d_model)
        std = math.sqrt(var)
        self.w_q = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn((num_heads * self.d_h, d_model), device=device),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            ),
            requires_grad=True
        )
        self.w_k = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn((num_heads * self.d_h, d_model), device=device),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            ),
            requires_grad=True
        )
        self.w_v = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn((num_heads * self.d_h, d_model), device=device),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            ),
            requires_grad=True
        )
        # To map output of the multi head attention to d_model
        self.w_o = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn((d_model, num_heads * self.d_h), device=device),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            ),
            requires_grad=True
        )
        self._attention = ScaledDotProductAttention()
        self._rope = RoPE(d_k=self.d_h, theta=theta, max_seq_len=max_seq_len, device=device)
        

    def forward(self, x: Float[torch.Tensor, "... seq d_model"], token_positions):
        # Project all heads in one matmul: (..., seq, h*d_h)
        Q = x @ self.w_q.T
        K = x @ self.w_k.T
        V = x @ self.w_v.T

        # Split into heads: (..., h, seq, d_h)
        Q = einops.rearrange(Q, "... seq (h dh) -> ... h seq dh", h=self.num_heads)
        K = einops.rearrange(K, "... seq (h dh) -> ... h seq dh", h=self.num_heads)
        V = einops.rearrange(V, "... seq (h dh) -> ... h seq dh", h=self.num_heads)

        # Rotate with RoPE
        K = self._rope(K, token_positions)
        Q = self._rope(Q, token_positions)

        seq_len = x.shape[-2]
        # Causal mask: True means "allowed to attend"
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attention = self._attention(Q=Q, K=K, V=V, mask=mask)

        # Merge heads back: (..., seq, h*d_h)
        attention = einops.rearrange(attention, "... h seq dh -> ... seq (h dh)")
        return attention @ self.w_o.T

