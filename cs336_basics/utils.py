import os
import typing

import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a batch of input sequences and corresponding target sequences for language model training.

    Args:
        dataset: numpy array of token IDs, shape (n,)
        batch_size: number of sequences in the batch
        context_length: length of each sequence
        device: PyTorch device string (e.g., 'cpu', 'cuda:0')

    Returns:
        Tuple of (input_sequences, target_sequences), both tensors of shape (batch_size, context_length)
    """
    max_start_idx = len(dataset) - context_length - 1

    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    seq_indices = rearrange(np.arange(context_length), "context_length -> 1 context_length")

    input_indices = rearrange(start_indices, "batch_size -> batch_size 1") + seq_indices
    target_indices = input_indices + 1

    input_sequences = torch.tensor(dataset[input_indices], dtype=torch.long, device=device)
    target_sequences = torch.tensor(dataset[target_indices], dtype=torch.long, device=device)

    return input_sequences, target_sequences


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    os.makedirs(os.path.dirname(out), exist_ok=True)  # type:ignore
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    if not os.path.exists(src):  # type:ignore
        raise FileNotFoundError(f"Checkpoint file not found: {src}")

    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
