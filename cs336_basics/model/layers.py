import math

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce, repeat


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
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")


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

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the embedding matrix using truncated normal distribution."""
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for the given token IDs.

        Args:
            token_ids (torch.Tensor): Tensor of token IDs of any shape.

        Returns:
            torch.Tensor: Embedding vectors of shape (*token_ids.shape, embedding_dim).
        """
        return self.weight[token_ids]


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

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

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

        rms = torch.sqrt(einsum(x.pow(2), "... d_model -> ...") / self.d_model + self.eps).unsqueeze(-1)

        result = x / rms * self.weight

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


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply softmax.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimention to apply softmax over.

    Returns:
        torch.Tensor: Output tensor.
    """

    x_shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)

        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


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
