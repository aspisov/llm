import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from llm_scratch.data.datamodules.pretrain_datamodule import PretrainDataModule
from llm_scratch.train.lightning.pretrain_module import PretrainLightningModule


def _write_npz(path, data):
    np.savez(path, data)


def test_lightning_pretrain_step(tmp_path):
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    data = np.arange(256, dtype=np.int64)
    _write_npz(train_path, data)
    _write_npz(val_path, data)

    model_cfg = OmegaConf.create(
        {
            "_target_": "llm_scratch.train.factories.build_transformer",
            "vocab_size": 512,
            "context_length": 8,
            "num_layers": 1,
            "d_model": 16,
            "num_heads": 4,
            "d_ff": 32,
            "rope_theta": 10000,
            "device": "cpu",
            "dtype": "float32",
        }
    )
    optimizer_cfg = OmegaConf.create(
        {
            "_target_": "llm_scratch.core.optimizers.AdamW",
            "lr": 1e-3,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
        }
    )
    scheduler_cfg = OmegaConf.create(
        {
            "_target_": "llm_scratch.train.lightning.schedulers.build_cosine_scheduler",
            "max_lr": 1e-3,
            "min_lr": 1e-5,
            "warmup_iters": 2,
            "cosine_cycle_iters": 10,
        }
    )
    training_cfg = OmegaConf.create(
        {
            "max_lr": 1e-3,
            "min_lr": 1e-5,
            "warmup_iters": 2,
            "cosine_cycle_iters": 10,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "max_l2_norm": 1,
        }
    )

    model = instantiate(model_cfg)
    module = PretrainLightningModule(
        model=model,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        training=training_cfg,
    )
    datamodule = PretrainDataModule(
        train_path=str(train_path),
        val_path=str(val_path),
        batch_size=2,
        context_length=8,
    )
    datamodule.setup("fit")

    batch = next(iter(datamodule.train_dataloader()))
    loss = module.training_step(batch, 0)

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
