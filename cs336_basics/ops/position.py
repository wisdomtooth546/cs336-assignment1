import einops
from einops import rearrange
import torch
from torch import nn

class RoPE(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        d_k -> dimension of q & k vectors
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        if d_k % 2 != 0:
            raise ValueError(f"Expected d_k to be even, received: {d_k}")
        self.max_seq_len = max_seq_len
        self.device = device
        device_ = device if device is not None else None
        positions = torch.arange(max_seq_len, device=device_).float()  # (L,)
        freqs = torch.arange(0, d_k, 2, device=device_).float()  # (d_k/2,)
        inv_freq = 1.0 / (theta ** (freqs / d_k))  # (d_k/2,)
        angles = torch.outer(positions, inv_freq)  # (L, d_k/2)
        self.register_buffer("cos", angles.cos(), persistent=False)  # (L, d_k/2)
        self.register_buffer("sin", angles.sin(), persistent=False)  # (L, d_k/2)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        token_positions : a tensor of shape (..., seq_len)
        x: input tensor of shape (..., seq_len, d_k) 
        """
        seq_len = token_positions.shape[-1]
        if seq_len > self.max_seq_len:
            raise ValueError("INput sequence length greater than max supported")
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected x.shape[-1] == d_k ({self.d_k}), got {x.shape[-1]}")

        # Select cos/sin for the provided positions: (..., seq_len, d_k/2)
        pos = token_positions.to(dtype=torch.long)
        cos = self.cos[pos].to(dtype=x.dtype, device=x.device)
        sin = self.sin[pos].to(dtype=x.dtype, device=x.device)
        cos, sin = self.match_dims(cos, x), self.match_dims(sin, x)

        x1 = x[..., ::2]   # (..., seq_len, d_k/2)
        x2 = x[..., 1::2]  # (..., seq_len, d_k/2)
        x_rot_even = x1 * cos - x2 * sin
        x_rot_odd = x1 * sin + x2 * cos
        x_stacked = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        return x_stacked.flatten(-2)
    
    def match_dims(self, trig_op, x):
        # trig_op: (L, D)
        # x:   (..., L, D)

        while trig_op.ndim < x.ndim:
            trig_op = trig_op.unsqueeze(0)
        return trig_op