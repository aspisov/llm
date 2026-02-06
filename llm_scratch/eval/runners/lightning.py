from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig

from llm_scratch.train.utils import instantiate_loggers


def evaluate_with_lightning(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if not cfg.get("ckpt_path"):
        raise ValueError("ckpt_path must be provided for lightning evaluation")

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    module = hydra.utils.instantiate(cfg.module, model=model)
    loggers = instantiate_loggers(cfg.get("logger"))
    trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)

    trainer.validate(module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "module": module,
        "loggers": loggers,
        "trainer": trainer,
    }
    metric_dict = dict(trainer.callback_metrics)
    return metric_dict, object_dict
