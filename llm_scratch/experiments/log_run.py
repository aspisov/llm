from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iter_source_files(root: Path) -> list[Path]:
    skip_dirs = {".git", ".venv", "outputs", "data", "__pycache__"}
    skip_files = {"profile.json", "profile.html"}
    files: list[Path] = []

    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in skip_dirs for part in rel.parts):
            continue
        if path.is_dir():
            continue
        if path.name in skip_files:
            continue
        files.append(path)

    return files


def _git_info(root: Path) -> dict[str, str | bool]:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True)
    sha = result.stdout.strip()
    dirty_result = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True)
    dirty = bool(dirty_result.stdout.strip())
    return {"git_sha": sha, "git_dirty": dirty}


def log_run(cfg, run_dir: Path | None = None, overrides: list[str] | None = None) -> Path:
    root = _repo_root()
    if run_dir is None:
        hydra_config = HydraConfig.get()
        run_dir = Path(hydra_config.runtime.output_dir)
        overrides = list(hydra_config.overrides.task)
    if overrides is None:
        overrides = []
    run_dir.mkdir(parents=True, exist_ok=True)

    unresolved_path = run_dir / "config_not_resolved.yaml"
    resolved_path = run_dir / "config.yaml"

    OmegaConf.save(cfg, unresolved_path)
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(resolved_cfg, resolved_path)

    src_zip_path = run_dir / "src.zip"
    with ZipFile(src_zip_path, "w") as archive:
        for path in _iter_source_files(root):
            archive.write(path, path.relative_to(root))

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": os.sys.version,
        "torch": torch.__version__,
        "overrides": overrides,
        "run_dir": str(run_dir),
        **_git_info(root),
    }
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return run_dir
