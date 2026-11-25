import math
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        var = 2 / (in_features + out_features)
        std = math.sqrt(var)
        self.weights = nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty(out_features, in_features, dtype=dtype, device=device),
            mean=0,
            std=std,
            a=-3 * std,
            b=3 * std
        ),
        requires_grad=True
        )
        # No bias in transformers
        # self.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T