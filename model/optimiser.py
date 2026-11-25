from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
from typing import Iterable

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float,
        betas: Iterable,
        eps: float,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
    
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data
                t = state.get("t", 1)
                m = b1 * m + (1 - b1) * grad
                m_hat = m / (1 - b1 ** t) # bias correction
                v = b2 * v + (1 - b2) * (grad ** 2)
                v_hat = v/ (1 - b2 ** t) # bias correction
                
                p.data -= lr * weight_decay * p.data
                p.data -= lr * m_hat/(v_hat.sqrt() + eps)
                
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss


