from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from llm_scratch.core.tokenizer_py import Tokenizer


def run_inference(cfg: DictConfig) -> dict:
    model = hydra.utils.instantiate(cfg.model)
    mode = str(cfg.infer.mode)

    if mode == "dry_run":
        return {
            "infer/mode": mode,
            "infer/num_parameters": float(sum(param.numel() for param in model.parameters())),
        }

    if mode == "generate":
        prompt = str(cfg.infer.prompt)
        if not prompt:
            raise ValueError("infer.prompt must be non-empty when infer.mode=generate")

        vocab_path = Path(cfg.infer.tokenizer_vocab_path)
        merges_path = Path(cfg.infer.tokenizer_merges_path)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Tokenizer vocab file not found: {vocab_path}")
        if not merges_path.exists():
            raise FileNotFoundError(f"Tokenizer merges file not found: {merges_path}")

        tokenizer = Tokenizer.from_files(
            str(vocab_path),
            str(merges_path),
            list(cfg.infer.special_tokens),
        )
        generated = model.generate_text(
            prompt=prompt,
            tokenizer=tokenizer,
            max_tokens=int(cfg.infer.max_tokens),
            top_p=float(cfg.infer.top_p),
            temperature=float(cfg.infer.temperature),
        )
        return {
            "infer/mode": mode,
            "infer/output_text": generated,
        }

    raise ValueError(f"Unsupported infer.mode: {mode}")
