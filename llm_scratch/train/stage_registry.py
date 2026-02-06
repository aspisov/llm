from __future__ import annotations

from omegaconf import DictConfig


SUPPORTED_STAGE_BACKENDS: dict[str, str] = {
    "pretrain": "lightning",
    "midtrain": "lightning",
    "sft": "lightning",
    "dpo": "lightning",
    "ppo": "torch",
    "rlvr": "torch",
}


def validate_stage_config(stage_cfg: DictConfig) -> None:
    if not isinstance(stage_cfg, DictConfig):
        raise TypeError("stage config must be a DictConfig")

    required_keys = ("name", "backend", "runner")
    for key in required_keys:
        if key not in stage_cfg:
            raise KeyError(f"stage config missing required key: {key}")

    stage_name = str(stage_cfg.name)
    stage_backend = str(stage_cfg.backend)

    if stage_name not in SUPPORTED_STAGE_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_STAGE_BACKENDS))
        raise ValueError(f"Unsupported stage '{stage_name}'. Supported stages: {supported}")

    expected_backend = SUPPORTED_STAGE_BACKENDS[stage_name]
    if stage_backend != expected_backend:
        raise ValueError(
            f"Stage '{stage_name}' must use backend '{expected_backend}', got '{stage_backend}'"
        )
