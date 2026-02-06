from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig

from llm_scratch.train.utils import instantiate_callbacks, instantiate_loggers


def run_lightning_stage(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    module = hydra.utils.instantiate(cfg.module, model=model)
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("logger"))
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "module": module,
        "callbacks": callbacks,
        "loggers": loggers,
        "trainer": trainer,
    }
    metric_dict = dict(trainer.callback_metrics)
    return metric_dict, object_dict


def run_torch_stage(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_name = str(cfg.stage.name)
    raise NotImplementedError(
        f"Torch stage runner is not implemented yet for stage '{stage_name}'"
    )
