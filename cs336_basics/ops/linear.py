from torch import nn
import torch
import math
import einops


class Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        # Same layout as torch.nn.Linear.weight: (out_features, in_features).
        init_weights = torch.randn((out_features, in_features), device=device)
        var = 2 / (in_features + out_features)
        std = math.sqrt(var)
        self.weights = nn.Parameter(
            nn.init.trunc_normal_(
                init_weights,
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            ),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weights, "... d_in, out d_in -> ... out")
