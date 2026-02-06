import torch

from llm_scratch.core.optimizers import learning_rate_schedule


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if max_lr <= 0:
        raise ValueError("max_lr must be > 0")

    def lr_lambda(step: int) -> float:
        lr = learning_rate_schedule(step, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        return lr / max_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
