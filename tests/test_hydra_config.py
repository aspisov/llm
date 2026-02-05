from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import open_dict

from llm_scratch.train.utils import instantiate_callbacks, instantiate_loggers


def test_hydra_train_config_load():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="train",
            return_hydra_config=True,
            overrides=[
                "hardware.device=cpu",
                "hardware.accelerator=cpu",
                "hardware.dtype=float32",
                "hardware.precision=32",
            ],
        )

    HydraConfig().set_config(cfg)
    with open_dict(cfg):
        cfg.paths.output_dir = "/tmp/llm_scratch_test_output"
    model = instantiate(cfg.model)
    module = instantiate(cfg.module, model=model)
    datamodule = instantiate(cfg.data)
    callbacks = instantiate_callbacks(cfg.callbacks)
    loggers = instantiate_loggers(cfg.logger)
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    assert model is not None
    assert module is not None
    assert datamodule is not None
    assert trainer is not None
    assert callbacks
    assert loggers


def test_hydra_eval_config_load():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="eval",
            return_hydra_config=True,
            overrides=[
                "ckpt_path=/tmp/placeholder.ckpt",
                "hardware.device=cpu",
                "hardware.accelerator=cpu",
                "hardware.dtype=float32",
                "hardware.precision=32",
            ],
        )

    HydraConfig().set_config(cfg)
    with open_dict(cfg):
        cfg.paths.output_dir = "/tmp/llm_scratch_test_output"
    model = instantiate(cfg.model)
    module = instantiate(cfg.module, model=model)
    datamodule = instantiate(cfg.data)
    loggers = instantiate_loggers(cfg.logger)
    trainer = instantiate(cfg.trainer, logger=loggers)

    assert model is not None
    assert module is not None
    assert datamodule is not None
    assert trainer is not None
    assert loggers


def test_hydra_train_model_invalid_dtype_raises():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="train",
            return_hydra_config=True,
            overrides=[
                "hardware.device=cpu",
                "hardware.accelerator=cpu",
                "hardware.dtype=float64",
                "hardware.precision=32",
            ],
        )

    with pytest.raises(InstantiationException, match="Unsupported dtype"):
        instantiate(cfg.model)
