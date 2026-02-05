from __future__ import annotations

import torch
from hydra import main as hydra_main
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from llm_scratch.data.datamodules.pretrain_datamodule import PretrainDataModule
from llm_scratch.train.config import Config
from llm_scratch.train.lightning.pretrain_module import PretrainLightningModule


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


def _build_callbacks(config: Config) -> list[ModelCheckpoint]:
    callbacks: list[ModelCheckpoint] = []
    if config.checkpoint_frequency is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=config.checkpoints_path,
                every_n_train_steps=config.checkpoint_frequency,
                save_last=True,
                filename="checkpoint_iter_{step}",
            )
        )
    return callbacks


@hydra_main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")

    config = Config.from_dict(OmegaConf.to_container(cfg, resolve=True))
    module = PretrainLightningModule(config)
    datamodule = PretrainDataModule(config)

    trainer = Trainer(
        accelerator=_map_accelerator(config.device),
        devices=cfg.trainer.devices,
        max_steps=config.num_iterations,
        val_check_interval=config.val_frequency,
        limit_val_batches=cfg.trainer.limit_val_batches,
        gradient_clip_val=config.max_l2_norm,
        gradient_clip_algorithm="norm",
        precision=_map_precision(config.dtype),
        callbacks=_build_callbacks(config),
        fast_dev_run=cfg.trainer.fast_dev_run,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=config.initial_checkpoint)


if __name__ == "__main__":
    main()
