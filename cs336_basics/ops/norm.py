import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.weights = nn.Parameter(torch.randn((d_model))) # g_i
        self.eps = eps
        self.device = device
        self.dtype = dtype
    
    def forward(self, x):
        in_dtype = x.dtype
        if in_dtype != torch.float32:
            x = x.to(torch.float32)
        rms = torch.sqrt(
            (torch.sum(torch.square(x), dim=-1, keepdim=True) + self.eps) / self.d_model
        )
        return torch.mul((x/rms), self.weights).to(in_dtype)