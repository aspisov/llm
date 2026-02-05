import numpy as np
import torch

from llm_scratch.data.datamodules.pretrain_datamodule import PretrainDataModule
from llm_scratch.train.config import Config
from llm_scratch.train.lightning.pretrain_module import PretrainLightningModule


def _write_npz(path, data):
    np.savez(path, data)


def test_lightning_pretrain_step(tmp_path):
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    data = np.arange(256, dtype=np.int64)
    _write_npz(train_path, data)
    _write_npz(val_path, data)

    config = Config(
        vocab_size=512,
        context_length=8,
        num_layers=1,
        d_model=16,
        num_heads=4,
        d_ff=32,
        rope_theta=10000,
        device="cpu",
        dtype=torch.float32,
        max_lr=1e-3,
        min_lr=1e-5,
        warmup_iters=2,
        cosine_cycle_iters=10,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        max_l2_norm=1,
        train_path=str(train_path),
        val_path=str(val_path),
        num_iterations=5,
        val_frequency=2,
        batch_size=2,
        initial_checkpoint=None,
        checkpoints_path=str(tmp_path / "checkpoints"),
        checkpoint_frequency=None,
    )

    module = PretrainLightningModule(config)
    datamodule = PretrainDataModule(config)
    datamodule.setup("fit")

    batch = next(iter(datamodule.train_dataloader()))
    loss = module.training_step(batch, 0)

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
