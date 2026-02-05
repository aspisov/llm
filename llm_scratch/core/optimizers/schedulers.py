import math


def learning_rate_schedule(
    it: float, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    """
    Learning rate schedule with linear warmup followed by cosine annealing.

    Args:
        it: Current iteration (step number).
        max_learning_rate: Peak learning rate after warmup.
        min_learning_rate: Minimum learning rate during and after cosine decay.
        warmup_iters: Number of iterations for linear warmup.
        cosine_cycle_iters: Total iterations for the cosine cycle (including warmup).

    Returns:
        Learning rate for the current iteration.
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        return (
            min_learning_rate
            + (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi))
            * (max_learning_rate - min_learning_rate)
            / 2
        )
    return min_learning_rate
