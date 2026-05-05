import torch
from torch import nn
import einops

def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)

class SwiGLUPoistionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None, device: str| None = None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = round(8 * d_model/(3 * 64)) * 64 # Make the output 8/3 and a multiple of 64
        self.d_ff = d_ff
        # First Layer
        self.w_1 = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.randn(
                    (self.d_ff, d_model), 
                    device=device
                )
            ),
            requires_grad=True
        )
        # Gating Unit
        self.w_3 = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.randn(
                    ( d_ff, d_model), # Make the output 8/3 and a multiple of 64
                    device=device
                )
            )
        )
        # Last layer
        self.w_2 = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.randn(
                    (d_model, d_ff),
                    device=device
                )
            )
        )

    
    def forward(self, x):
        silud = silu(
            einops.einsum(
                self.w_1, x, "... d_ff d_model, ... d_model -> ... d_ff"
            )
        )
        gated = einops.einsum(
            self.w_3, x, "... d_ff d_model, ... d_model -> ... d_ff"
        )
        return einops.einsum(
            self.w_2,
            torch.mul(
                silud, gated 
            ),
            "... d_model d_ff, ... d_ff -> ... d_model"

        )