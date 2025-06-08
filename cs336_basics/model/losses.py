import torch


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply log softmax.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimention to apply log softmax over.

    Returns:
        torch.Tensor: Output tensor.
    """

    x_shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return x_shifted - torch.log(sum_exp)


def cross_entropy(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss.

    Args:
        y (torch.Tensor): True class indices of shape (...,)
        logits (torch.Tensor): Predicted logits of shape (..., num_classes)

    Returns:
        torch.Tensor: Cross-entorpy loss (scalar)
    """

    log_probs = log_softmax(logits)

    nll_loss = -log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

    return torch.mean(nll_loss)
