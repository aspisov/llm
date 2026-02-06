from __future__ import annotations

import hydra
from omegaconf import DictConfig


def run_serve(cfg: DictConfig) -> dict:
    model = hydra.utils.instantiate(cfg.model)
    mode = str(cfg.serve.mode)

    if mode == "dry_run":
        return {
            "serve/mode": mode,
            "serve/num_parameters": float(sum(param.numel() for param in model.parameters())),
        }

    if mode == "api":
        raise NotImplementedError("serve.mode=api is not implemented yet")

    raise ValueError(f"Unsupported serve.mode: {mode}")
