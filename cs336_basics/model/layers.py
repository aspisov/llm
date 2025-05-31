import math

import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat


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

        self.w = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the weight matrix using truncated normal distribution."""
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.w, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return einsum(self.w, x, "... out_features in_features, ... in_features -> ... out_features")


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

        # Convert to float32 for numerical stability during normalization
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(einsum(x.pow(2), "... d_model -> ...") / self.d_model + self.eps)
        rms = repeat(rms, "... -> ... 1")

        result = x / rms * self.gain

        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


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

        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the weight matrices using truncated normal distribution."""
        std = math.sqrt(2 / (self.d_model + self.d_ff))
        nn.init.trunc_normal_(self.w1, mean=0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.w2, mean=0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.w3, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        h1 = einsum(self.w1, x, "... d_ff d_model, ... d_model -> ... d_ff")
        h2 = einsum(self.w3, x, "... d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.w2, silu(h1) * h2, "... d_model d_ff, ... d_ff -> ... d_model")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even for RoPE"

        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))

        positions = torch.arange(max_seq_len, device=device)
        angles = torch.outer(positions, freqs)

        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_vals = self.cos_cache[token_positions]  # type:ignore
        sin_vals = self.sin_cache[token_positions]  # type:ignore

        x_pairs = rearrange(x, "... (pairs two) -> ... pairs two", two=2)

        rotated_pairs = torch.zeros_like(x_pairs)
        rotated_pairs[..., 0] = x_pairs[..., 0] * cos_vals - x_pairs[..., 1] * sin_vals
        rotated_pairs[..., 1] = x_pairs[..., 0] * sin_vals + x_pairs[..., 1] * cos_vals

        rotated = rearrange(rotated_pairs, " ... pairs two -> ... (pairs two)")
        return rotated
