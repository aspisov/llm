import logging
import os
import typing
from dataclasses import dataclass

import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: float
    device: str
    dtype: torch.dtype

    max_lr: float
    min_lr: float
    warmup_iters: int
    cosine_cycle_iters: int
    betas: tuple[float, float]
    weight_decay: float
    max_l2_norm: int

    train_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    val_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    num_iterations: int
    val_frequency: int
    batch_size: int

    initial_checkpoint: str | None
    checkpoints_path: str
    checkpoint_frequency: int | None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        dtype_str = config_dict["hardware"]["dtype"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(dtype_str, torch.float32)

        return cls(
            vocab_size=config_dict["model"]["vocab_size"],
            context_length=config_dict["model"]["context_length"],
            num_layers=config_dict["model"]["num_layers"],
            d_model=config_dict["model"]["d_model"],
            num_heads=config_dict["model"]["num_heads"],
            d_ff=config_dict["model"]["d_ff"],
            rope_theta=config_dict["model"]["rope_theta"],
            device=config_dict["hardware"]["device"],
            dtype=dtype,
            max_lr=config_dict["training"]["max_lr"],
            min_lr=config_dict["training"]["min_lr"],
            warmup_iters=config_dict["training"]["warmup_iters"],
            cosine_cycle_iters=config_dict["training"]["cosine_cycle_iters"],
            betas=tuple(config_dict["training"]["betas"]),
            weight_decay=config_dict["training"]["weight_decay"],
            num_iterations=config_dict["training"]["num_iterations"],
            val_frequency=config_dict["training"]["val_frequency"],
            batch_size=config_dict["training"]["batch_size"],
            max_l2_norm=config_dict["training"]["max_l2_norm"],
            train_path=config_dict["data"]["train_path"],
            val_path=config_dict["data"]["val_path"],
            initial_checkpoint=config_dict["checkpoints"]["initial_checkpoint"],
            checkpoints_path=config_dict["checkpoints"]["checkpoints_path"],
            checkpoint_frequency=config_dict["checkpoints"]["checkpoint_frequency"],
        )

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)

        logger.info("Loaded configuration from %s", config_path)
        return cls.from_dict(config_dict)
