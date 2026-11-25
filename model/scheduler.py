import math


def cosine_lr(t, lr_min, lr_max, warmup_steps, total_steps):
    if t < warmup_steps:
        return lr_max * t / warmup_steps

    if t >= total_steps:
        return lr_min

    progress = (t - warmup_steps) / (total_steps - warmup_steps)

    return lr_min + 0.5 * (1 + math.cos(math.pi * progress)) * (
        lr_max - lr_min
    )