from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig


def run_export(cfg: DictConfig) -> dict:
    mode = str(cfg.export.mode)
    if mode != "metadata_only":
        raise ValueError(f"Unsupported export.mode: {mode}")

    model = hydra.utils.instantiate(cfg.model)
    output_dir = Path(cfg.export.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "export_metadata.json"

    payload = {
        "num_parameters": int(sum(param.numel() for param in model.parameters())),
        "stage": str(cfg.export.stage),
    }
    output_path.write_text(json.dumps(payload, indent=2))

    return {
        "export/mode": mode,
        "export/output_path": str(output_path),
    }
