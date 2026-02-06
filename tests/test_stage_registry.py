import pytest
from omegaconf import OmegaConf

from llm_scratch.train.stage_registry import validate_stage_config


def test_dpo_requires_lightning_backend():
    stage_cfg = OmegaConf.create(
        {
            "name": "dpo",
            "backend": "torch",
            "runner": {"_target_": "llm_scratch.train.runners.stage_runners.run_torch_stage"},
        }
    )

    with pytest.raises(ValueError, match="must use backend 'lightning'"):
        validate_stage_config(stage_cfg)


def test_rlvr_requires_torch_backend():
    stage_cfg = OmegaConf.create(
        {
            "name": "rlvr",
            "backend": "lightning",
            "runner": {"_target_": "llm_scratch.train.runners.stage_runners.run_lightning_stage"},
        }
    )

    with pytest.raises(ValueError, match="must use backend 'torch'"):
        validate_stage_config(stage_cfg)


def test_unknown_stage_is_rejected():
    stage_cfg = OmegaConf.create(
        {
            "name": "unknown",
            "backend": "lightning",
            "runner": {"_target_": "llm_scratch.train.runners.stage_runners.run_lightning_stage"},
        }
    )

    with pytest.raises(ValueError, match="Unsupported stage"):
        validate_stage_config(stage_cfg)


def test_valid_dpo_stage_config_passes():
    stage_cfg = OmegaConf.create(
        {
            "name": "dpo",
            "backend": "lightning",
            "runner": {"_target_": "llm_scratch.train.runners.stage_runners.run_lightning_stage"},
        }
    )

    validate_stage_config(stage_cfg)
