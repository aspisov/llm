import math

import torch
import torch.nn as nn
from einops import einsum, rearrange

from cs336_basics.model import Linear, softmax


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


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Scaled dot-product attention mechanism.

    Args:
        Q: Query tensor of shape (..., seq_len, d_k)
        K: Key tensor of shape (..., seq_len, d_k)
        V: Value tensor of shape (..., seq_len, d_v)
        mask: Optional boolean mask of shape (seq_len, seq_len).
              True positions have attention weights that sum to 1,
              False positions have attention weights of 0.

    Returns:
        Output tensor of shape (..., seq_len, d_v)
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

    if mask is not None:
        masked_scores = torch.where(mask, scores, float("-inf"))
    else:
        masked_scores = scores

    attn_weights = softmax(masked_scores, dim=-1)

    return einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism with optional Rotary Position Embedding.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        max_seq_len (int | None): Maximum sequence length for RoPE.
        theta (float | None): Base frequency for RoPE.
        device (torch.device | None): Device to place the layer on.
        dtype (torch.dtype | None): Data type for the layer parameters.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = Linear(d_model, self.num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, self.num_heads * self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, self.num_heads * self.d_v, device=device, dtype=dtype)
        self.output_proj = Linear(self.num_heads * self.d_v, d_model, device=device, dtype=dtype)

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).
            token_positions (torch.Tensor | None): Position indices for RoPE.

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        seq_len = x.shape[-2]

        Q = rearrange(
            self.q_proj(x), "... seq_len (num_heads d_k) -> ... seq_len num_heads d_k", num_heads=self.num_heads
        )
        K = rearrange(
            self.k_proj(x), "... seq_len (num_heads d_k) -> ... seq_len num_heads d_k", num_heads=self.num_heads
        )
        V = rearrange(
            self.v_proj(x), "... seq_len (num_heads d_v) -> ... seq_len num_heads d_v", num_heads=self.num_heads
        )

        if hasattr(self, "rope") and token_positions is not None:
            Q = torch.stack([self.rope(Q[..., i, :], token_positions) for i in range(self.num_heads)], dim=-2)
            K = torch.stack([self.rope(K[..., i, :], token_positions) for i in range(self.num_heads)], dim=-2)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        Q_attn = rearrange(Q, "... seq_len num_heads d_k -> ... num_heads seq_len d_k")
        K_attn = rearrange(K, "... seq_len num_heads d_k -> ... num_heads seq_len d_k")
        V_attn = rearrange(V, "... seq_len num_heads d_v -> ... num_heads seq_len d_v")

        attn_output = scaled_dot_product_attention(Q_attn, K_attn, V_attn, mask)

        attn_output = rearrange(attn_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        output = self.output_proj(attn_output)

        return output
