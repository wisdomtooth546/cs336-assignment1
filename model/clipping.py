import torch


def get_global_norm(params, device):
    total = torch.zeros((), device=device)
    for p in params:
        if p.grad is not None:
            total += (p.grad ** 2).sum()
    return total.sqrt()

def clip_gradients(params, max_gradient_norm, eps, device):
    norm = get_global_norm(params, device)
    # scale
    scale = min(1.0, max_gradient_norm / (norm.item() + eps))
    if scale < 1:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)