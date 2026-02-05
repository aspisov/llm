from __future__ import annotations

from typing import Any

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from llm_scratch.experiments.log_run import log_run
from llm_scratch.train.utils import instantiate_callbacks, instantiate_loggers


def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    torch.set_float32_matmul_precision("high")

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    module = hydra.utils.instantiate(cfg.module, model=model)
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("logger"))
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "module": module,
        "callbacks": callbacks,
        "loggers": loggers,
        "trainer": trainer,
    }

    if not HydraConfig.initialized():
        raise RuntimeError("Hydra must be initialized before running train()")
    log_run(cfg)
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = dict(trainer.callback_metrics)
    return metric_dict, object_dict


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
