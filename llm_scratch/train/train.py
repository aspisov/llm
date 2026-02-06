from __future__ import annotations

from typing import Any

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from llm_scratch.experiments.log_run import log_run
from llm_scratch.train.stage_registry import validate_stage_config


def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_stage_config(cfg.stage)

    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    torch.set_float32_matmul_precision("high")

    if not HydraConfig.initialized():
        raise RuntimeError("Hydra must be initialized before running train()")

    log_run(cfg)

    result = hydra.utils.instantiate(cfg.stage.runner, cfg=cfg, _recursive_=False)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("stage runner must return a tuple: (metric_dict, object_dict)")

    metric_dict, object_dict = result
    if not isinstance(metric_dict, dict):
        raise TypeError("stage runner metric_dict must be a dict")
    if not isinstance(object_dict, dict):
        raise TypeError("stage runner object_dict must be a dict")

    return metric_dict, object_dict


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
