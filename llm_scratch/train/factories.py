import torch

from llm_scratch.core.model import Transformer


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_map[dtype]


def build_transformer(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> Transformer:
    return Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        num_heads=num_heads,
        num_layers=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=torch.device(device) if isinstance(device, str) else device,
        dtype=resolve_dtype(dtype),
    )
