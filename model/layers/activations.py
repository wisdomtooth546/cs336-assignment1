from einops import einsum
import torch
from torch import nn

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = x.max(dim=dim, keepdim=True).values
    x = x - max_x
    return x.exp() / x.exp().sum(dim=dim, keepdim=True)

def attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    pre_softmax_scores = einsum(Q, K, "... i d, ... j d -> ... i j") / (d_k ** 0.5)
    if mask is not None:
        pre_softmax_scores = pre_softmax_scores.masked_fill(~mask, float("-inf"))
    scores = softmax(pre_softmax_scores, dim=-1) 
    return einsum(scores, V, "... i j, ... j d -> ... i d" )