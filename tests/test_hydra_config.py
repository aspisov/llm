from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from llm_scratch.train.config import Config


def test_hydra_config_load():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config")

    config = Config.from_dict(OmegaConf.to_container(cfg, resolve=True))
    assert config.vocab_size > 0
    assert config.context_length > 0
