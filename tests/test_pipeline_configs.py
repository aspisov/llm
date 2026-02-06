from pathlib import Path

from hydra import compose, initialize_config_dir


def test_pipeline_entrypoint_configs_compose():
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    config_names = [
        "train",
        "eval",
        "infer",
        "serve",
        "data_prepare",
        "tokenizer_train",
        "tokenizer_eval",
        "export",
    ]

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        for config_name in config_names:
            cfg = compose(config_name=config_name)
            assert cfg is not None
