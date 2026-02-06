from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from llm_scratch.experiments.log_run import log_run


def infer(cfg: DictConfig) -> dict:
    if not HydraConfig.initialized():
        raise RuntimeError("Hydra must be initialized before running infer()")

    log_run(cfg)
    result = hydra.utils.instantiate(cfg.infer.runner, cfg=cfg, _recursive_=False)
    if not isinstance(result, dict):
        raise TypeError("infer runner must return a dict")
    return result


@hydra.main(config_path="../../configs", config_name="infer", version_base="1.3")
def main(cfg: DictConfig) -> None:
    infer(cfg)


if __name__ == "__main__":
    main()
