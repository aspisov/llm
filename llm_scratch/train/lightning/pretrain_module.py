from __future__ import annotations

import torch
from lightning.pytorch import LightningModule

from llm_scratch.core.model import Transformer, cross_entropy
from llm_scratch.core.optimizers import AdamW, learning_rate_schedule
from llm_scratch.train.config import Config


class PretrainLightningModule(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if self.config.max_lr <= 0:
            raise ValueError("max_lr must be > 0")

        self.model = Transformer(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_model=config.d_model,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
            device=None,
            dtype=config.dtype,
        )

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
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.max_lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            lr = learning_rate_schedule(
                step,
                self.config.max_lr,
                self.config.min_lr,
                self.config.warmup_iters,
                self.config.cosine_cycle_iters,
            )
            return lr / self.config.max_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
