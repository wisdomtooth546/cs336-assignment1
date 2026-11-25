# Layer norm is applied across the features for each sample
# The input and output should have the same shape
import torch
from torch import nn

class RMSLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        # This will be broadcast across batch and sequence
        # learnable gain parameter
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # To avoid overflow
        x = x.to(torch.float32)
        # Compute RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        norm = x * self.weights/rms
        return norm.to(in_dtype)