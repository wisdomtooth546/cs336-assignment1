import torch
from torch import nn
from torch.optim import Optimizer
from typing import BinaryIO, IO
from os import PathLike


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    checkpoint = torch.load(src, weights_only=True)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]