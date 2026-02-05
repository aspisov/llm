from __future__ import annotations

import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from omegaconf import DictConfig

from llm_scratch.core.model import cross_entropy


class PretrainLightningModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        training: DictConfig,
    ):
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.training = training

        if self.training.max_lr <= 0:
            raise ValueError("training.max_lr must be > 0")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        logits = self(inputs)
        loss = cross_entropy(y=targets, logits=logits)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        logits = self(inputs)
        loss = cross_entropy(y=targets, logits=logits)
        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
