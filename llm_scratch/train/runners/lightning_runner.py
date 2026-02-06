from __future__ import annotations

import torch
from hydra import main as hydra_main
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from llm_scratch.experiments.log_run import log_run


def _map_accelerator(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda"
    if device.startswith("mps"):
        return "mps"
    if device.startswith("cpu"):
        return "cpu"
    raise ValueError(f"Unsupported device string: {device}")


def _map_precision(dtype: torch.dtype) -> str | int:
    if dtype == torch.float16:
        return "16-mixed"
    if dtype == torch.bfloat16:
        return "bf16-mixed"
    if dtype == torch.float32:
        return 32
    raise ValueError(f"Unsupported dtype: {dtype}")


@hydra_main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")

    model = instantiate(cfg.model)
    module = instantiate(cfg.module, model=model)
    datamodule = instantiate(cfg.data)
    callbacks: list[ModelCheckpoint] = instantiate(cfg.callbacks)
    logger = instantiate(cfg.logger)

    log_run(cfg)

    dtype = next(model.parameters()).dtype
    accelerator = _map_accelerator(cfg.hardware.device)
    precision = _map_precision(dtype)

    trainer = instantiate(
        cfg.trainer,
        accelerator=accelerator,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.checkpoints.initial_checkpoint)


if __name__ == "__main__":
    main()
