from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from hydra.errors import InstantiationException


def test_hydra_config_load():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["hardware.device=cpu", "hardware.dtype=float32"])

    model = instantiate(cfg.model)
    module = instantiate(cfg.module, model=model)
    datamodule = instantiate(cfg.data)

    assert model is not None
    assert module is not None
    assert datamodule is not None


def test_hydra_model_invalid_dtype_raises():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["hardware.device=cpu", "hardware.dtype=float64"])

    with pytest.raises(InstantiationException, match="Unsupported dtype"):
        instantiate(cfg.model)
