import pytest
from omegaconf import OmegaConf

from llm_scratch.train.utils import instantiate_callbacks, instantiate_loggers


def test_instantiate_callbacks_requires_target():
    callbacks_cfg = OmegaConf.create({"checkpoint": {"dirpath": "."}})
    with pytest.raises(KeyError, match="must define _target_"):
        instantiate_callbacks(callbacks_cfg)


def test_instantiate_callbacks_requires_dictconfig_items():
    callbacks_cfg = OmegaConf.create({"checkpoint": "not-a-config"})
    with pytest.raises(TypeError, match="must be a DictConfig"):
        instantiate_callbacks(callbacks_cfg)


def test_instantiate_loggers_requires_dictconfig():
    with pytest.raises(TypeError, match="logger config must be a DictConfig"):
        instantiate_loggers(["csv"])


def test_instantiate_loggers_requires_target():
    logger_cfg = OmegaConf.create({"csv": {"save_dir": "."}})
    with pytest.raises(KeyError, match="must define _target_"):
        instantiate_loggers(logger_cfg)
