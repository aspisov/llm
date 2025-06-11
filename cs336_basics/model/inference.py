import torch

from cs336_basics.model.activations import softmax


def top_p_sampling(logits: torch.Tensor, p: float = 1, temperature: float = 1) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on a logits tensor.

    Args:
        logits (Tensor): Logits tensor of shape (vocab_size,).
        p (float): Cumulative probability threshold.
        temperature (float): Sampling temperature.

    Returns:
        int: Index of the sampled token.
    """
    logits = logits / temperature
    probs = softmax(logits)

    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cumulative_probs <= p
    mask[:, 0] = True
    sorted_probs[~mask] = 0.0

    normalized_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    sampled_indecies = torch.multinomial(normalized_probs, num_samples=1)

    return torch.gather(indices, dim=-1, index=sampled_indecies)
