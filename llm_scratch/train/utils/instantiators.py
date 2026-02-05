from __future__ import annotations

import hydra
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[Callback]:
    callbacks: list[Callback] = []
    if callbacks_cfg is None:
        return callbacks
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("callbacks config must be a DictConfig")

    for name, callback_cfg in callbacks_cfg.items():
        if not isinstance(callback_cfg, DictConfig):
            raise TypeError(f"callbacks.{name} must be a DictConfig")
        if "_target_" not in callback_cfg:
            raise KeyError(f"callbacks.{name} must define _target_")
        callback = hydra.utils.instantiate(callback_cfg)
        if not isinstance(callback, Callback):
            raise TypeError(f"callbacks.{name} did not instantiate a Lightning Callback")
        callbacks.append(callback)

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Logger]:
    loggers: list[Logger] = []
    if logger_cfg is None:
        return loggers
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("logger config must be a DictConfig")

    for name, logger_item_cfg in logger_cfg.items():
        if not isinstance(logger_item_cfg, DictConfig):
            raise TypeError(f"logger.{name} must be a DictConfig")
        if "_target_" not in logger_item_cfg:
            raise KeyError(f"logger.{name} must define _target_")
        logger = hydra.utils.instantiate(logger_item_cfg)
        if not isinstance(logger, Logger):
            raise TypeError(f"logger.{name} did not instantiate a Lightning Logger")
        loggers.append(logger)

    return loggers
