import math

import torch
import torch.nn as nn

from cs336_basics.model import Linear, silu


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) feedforward network.

    Implements the gated feedforward network from "GLU Variants Improve Transformer":
    SwiGLU(x) = (SiLU(W1 * x) ⊙ W3 * x) * W2

    This is commonly used in modern transformer architectures like LLaMA.

    Args:
        d_model (int): Input/output dimension.
        d_ff (int | None): Hidden dimension. If None, defaults to 8/3 * d_model rounded to nearest 64.
        device (torch.device | None): Device to place the layer on.
        dtype (torch.dtype | None): Data type for the layer parameters.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = math.ceil(d_model * 8 / 3 / 64) * 64

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))
