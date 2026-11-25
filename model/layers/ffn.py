from torch import nn
import torch

from model.layers.activations import silu
from model.layers.linear import Linear

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None) -> None:
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device)
        self.w2 = Linear(in_features=d_model, out_features=d_ff, device=device)
        self.w3 = Linear(in_features=d_ff, out_features=d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = self.w1(x)
        gate = self.w2(x)
        return self.w3(silu(projection) * gate)