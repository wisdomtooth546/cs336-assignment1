from torch import nn
import torch

class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        n_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__() 
        self.vocab_size = vocab_size
        self.n_dim = n_dim
        self.device = device
        self.dtype = dtype
        init_embeddings = nn.init.trunc_normal_(
            torch.randn((vocab_size, n_dim), device=device),
            mean=0,
            std=1,
            a=-3,
            b=3
        )
        self.weights = nn.Parameter(
            init_embeddings,
            requires_grad=True
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]