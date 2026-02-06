from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig


def run_data_prepare(cfg: DictConfig) -> dict:
    mode = str(cfg.data_prepare.mode)
    if mode != "validate_npz":
        raise ValueError(f"Unsupported data_prepare.mode: {mode}")

    train_path = Path(cfg.data_prepare.train_path)
    val_path = Path(cfg.data_prepare.val_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_path}")

    train = np.load(train_path, mmap_mode="r")["arr_0"]
    val = np.load(val_path, mmap_mode="r")["arr_0"]

    return {
        "data_prepare/mode": mode,
        "data_prepare/train_tokens": float(len(train)),
        "data_prepare/val_tokens": float(len(val)),
    }
