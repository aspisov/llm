import math

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    """
    A linear (fully connected) layer that applies an affine transformation.

    Applies Y = XW^T where X is the input and W is the learned weight matrix.
    Uses Xavier/Glorot initialization for stable training.

    Args:
        in_features (int): Size of input features.
        out_features (int): Size of output features.
        device (torch.device | None): Device to place the layer on.
        dtype (torch.dtype | None): Data type for the layer parameters.
    """

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
        """Initialize the weight matrix using truncated normal distribution."""
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return einsum(self.weight, x, "... out_features in_features, ... in_features -> ... out_features")


class Embedding(nn.Module):
    """
    A lookup table that maps token IDs to dense vector representations.

    Args:
        num_embeddings (int): Size of the vocabulary (number of possible tokens).
        embedding_dim (int): Dimensionality of the embedding vectors.
        device (torch.device | None): Device to place the layer on.
        dtype (torch.dtype | None): Data type for the layer parameters.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the embedding matrix using truncated normal distribution."""
        nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for the given token IDs.

        Args:
            token_ids (torch.Tensor): Tensor of token IDs of any shape.

        Returns:
            torch.Tensor: Embedding vectors of shape (*token_ids.shape, embedding_dim).
        """
        return self.embeddings[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes the input using the root mean square, providing
    a simpler alternative to LayerNorm without mean centering.

    Args:
        d_model (int): Dimensionality of the input features.
        eps (float): Small constant for numerical stability. Defaults to 1e-5.
        device (torch.device | None): Device to place the layer on.
        dtype (torch.dtype | None): Data type for the layer parameters.
    """

    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: RMS normalized tensor of the same shape as input.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Reason: Convert to float32 for numerical stability during normalization
        rms = torch.sqrt(einsum(x.pow(2), "... d_model -> ...") / self.d_model + self.eps).unsqueeze(-1)

        result = x / rms * self.gain

        return result.to(in_dtype)
