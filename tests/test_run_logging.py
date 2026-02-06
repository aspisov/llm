import importlib

from omegaconf import OmegaConf

log_run_module = importlib.import_module("llm_scratch.experiments.log_run")


def test_log_run_writes_artifacts(tmp_path, monkeypatch):
    cfg = OmegaConf.create({})

    monkeypatch.setattr(log_run_module, "_iter_source_files", lambda _root: [])
    monkeypatch.setattr(log_run_module, "_git_info", lambda _root: {"git_sha": "test", "git_dirty": False})

    run_dir = log_run_module.log_run(cfg, run_dir=tmp_path, overrides=[])

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "config_not_resolved.yaml").exists()
    assert (run_dir / "src.zip").exists()
    assert (run_dir / "metadata.json").exists()
