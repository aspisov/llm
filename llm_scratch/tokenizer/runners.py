from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from llm_scratch.core.tokenizer_py import Tokenizer
from llm_scratch.core.tokenizer_py.bpe_trainer import train_and_save_bpe


def run_tokenizer_train(cfg: DictConfig) -> dict:
    mode = str(cfg.tokenizer.train_mode)
    input_path = Path(cfg.tokenizer.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Tokenizer input file not found: {input_path}")

    if mode == "dry_run":
        return {
            "tokenizer/train_mode": mode,
            "tokenizer/input_path": str(input_path),
        }

    if mode == "train_and_save":
        vocab_path = Path(cfg.tokenizer.vocab_path)
        merges_path = Path(cfg.tokenizer.merges_path)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        merges_path.parent.mkdir(parents=True, exist_ok=True)

        train_and_save_bpe(
            input_path=str(input_path),
            vocab_size=int(cfg.tokenizer.vocab_size),
            special_tokens=list(cfg.tokenizer.special_tokens),
            vocab_save_path=str(vocab_path),
            merges_save_path=str(merges_path),
        )

        return {
            "tokenizer/train_mode": mode,
            "tokenizer/vocab_path": str(vocab_path),
            "tokenizer/merges_path": str(merges_path),
        }

    raise ValueError(f"Unsupported tokenizer.train_mode: {mode}")


def run_tokenizer_eval(cfg: DictConfig) -> dict:
    mode = str(cfg.tokenizer.eval_mode)
    if mode != "load_only":
        raise ValueError(f"Unsupported tokenizer.eval_mode: {mode}")

    vocab_path = Path(cfg.tokenizer.vocab_path)
    merges_path = Path(cfg.tokenizer.merges_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Tokenizer vocab file not found: {vocab_path}")
    if not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer merges file not found: {merges_path}")

    tokenizer = Tokenizer.from_files(
        str(vocab_path),
        str(merges_path),
        list(cfg.tokenizer.special_tokens),
    )
    return {
        "tokenizer/eval_mode": mode,
        "tokenizer/vocab_size": float(len(tokenizer.id_to_byte)),
        "tokenizer/num_merges": float(len(tokenizer.merges)),
    }
