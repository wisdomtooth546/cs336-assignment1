from torch import nn
import torch

class Softmax(nn.Module):
    def __init__(self) -> None:
        """
        exp(vi) can become inf for large values (then, inf/inf= NaN). We can avoid this by noticing
        that the softmax operation is invariant to adding any constant c to all inputs.
        We can leverage this property for numerical stability—typically,
        we will subtract the largest entry of oi from all elements of oi, making the
        new largest entry 0. 
        """
        super().__init__()
    
    def forward(self, x: torch.Tensor, dim: int):
        # subtract the max value from the dimension for numerical
        # stability
        x = x - x.max(dim=dim, keepdim=True).values
        return x.exp() / x.exp().sum(dim=dim, keepdim=True)



