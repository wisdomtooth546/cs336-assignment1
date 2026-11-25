import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits

    log_sum_exp = shifted_logits.exp().sum(dim=-1).log()

    target_logits = torch.gather(
        shifted_logits,
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    return (log_sum_exp - target_logits).mean()