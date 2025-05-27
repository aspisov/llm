import math

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype), requires_grad=True
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "... out_features in_features, ... in_features -> ... out_features")
