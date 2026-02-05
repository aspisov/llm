import torch


def to_torch_dtype(name: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {name}")
    return dtype_map[name]
