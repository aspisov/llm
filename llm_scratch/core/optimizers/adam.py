from collections.abc import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay=0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p

                grad = p.grad.data  # Get the gradient of loss with respect to p

                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad**2

                # Bias correction
                m_hat = m / (1 - betas[0] ** t)
                v_hat = v / (1 - betas[1] ** t)

                p.data -= lr * m_hat / (torch.sqrt(v_hat) + eps)
                p.data -= lr * weight_decay * p.data

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss
