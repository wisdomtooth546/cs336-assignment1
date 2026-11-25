from torch import nn
import torch
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        # precompute sine and cosine since it is value independent

        # Dimension indices: 0, 1, ..., d_k/2 - 1
        k = torch.arange(d_k // 2, dtype=torch.float32, device=device)
        i = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        i = rearrange(i, "i -> i 1")

        # Frequencies: 1 / theta^(2k/d_k)
        freqs = 1.0 / (theta ** (2 * k / d_k))
        freqs = rearrange(freqs, "k -> 1 k")
        angle = i @ freqs
        self.register_buffer("cos", angle.cos(), persistent=False)
        self.register_buffer("sin", angle.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        even, odd = rearrange(
            x, '... s (d two) -> two ... s d', two=2
        )
        # positions: [B, S]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # Rotate
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        # Reconstruct
        return rearrange(
            torch.stack([rotated_even, rotated_odd], dim=-1),
            '... s d two -> ... s (d two)'
        )

