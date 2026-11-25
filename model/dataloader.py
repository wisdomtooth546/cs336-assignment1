import numpy.typing as npt
import numpy as np
import torch

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
):
    starts = np.random.randint(
        0,
        len(dataset) - context_length,
        size=batch_size,
    )
    x = np.stack([
        dataset[i: i+ context_length]
        for i in starts
        ]
    )
    y = np.stack([
        dataset[i + 1: i + context_length + 1]
        for i in starts
    ])
    # Convert to PyTorch tensors and move to device
    inputs = torch.tensor(x, dtype=torch.long, device=device)
    targets = torch.tensor(y, dtype=torch.long, device=device)

    return inputs, targets

