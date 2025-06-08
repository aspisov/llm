import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply softmax.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimention to apply softmax over.

    Returns:
        torch.Tensor: Output tensor.
    """

    x_shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp
