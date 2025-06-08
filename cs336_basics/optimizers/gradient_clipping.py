from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    Apply gradient clipping to prevent exploding gradients.

    Computes the global L2 norm across all parameter gradients and scales
    all gradients uniformly if the global norm exceeds max_l2_norm.

    Args:
        parameters: Iterable of model parameters.
        max_l2_norm: Maximum allowed L2 norm for gradients.
    """
    gradients = []
    for p in parameters:
        if p.grad is not None:
            gradients.append(p.grad.view(-1))

    if not gradients:
        return

    global_norm = torch.cat(gradients).norm()

    if global_norm > max_l2_norm:
        clip_factor = max_l2_norm / (global_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_factor)
