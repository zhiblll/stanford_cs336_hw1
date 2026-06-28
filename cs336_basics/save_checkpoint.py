from __future__ import annotations

import os
from typing import BinaryIO, IO

import torch


CheckpointTarget = str | os.PathLike | BinaryIO | IO[bytes]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointTarget,
) -> None:
    """
    Save a training checkpoint.

    Stores:
    - model parameters/state
    - optimizer state
    - current iteration

    Args:
        model: model whose state should be saved
        optimizer: optimizer whose state should be saved
        iteration: current training step / iteration
        out: file path or binary file-like object
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: CheckpointTarget,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load a training checkpoint and restore model + optimizer state.

    Args:
        src: file path or binary file-like object
        model: model instance to restore into
        optimizer: optimizer instance to restore into

    Returns:
        The saved iteration number.
    """
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    iteration = checkpoint["iteration"]
    return int(iteration)

