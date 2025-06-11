import logging
import os
import typing
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import torch
import yaml

from cs336_basics.model import Transformer, cross_entropy
from cs336_basics.optimizers import AdamW, clip_gradients, learning_rate_schedule
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import get_batch, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@dataclass
class Config:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: float
    device: str
    dtype: torch.dtype

    max_lr: float
    min_lr: float
    warmup_iters: int
    cosine_cycle_iters: int
    betas: tuple[float, float]
    weight_decay: float
    max_l2_norm: int

    train_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    val_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    num_iterations: int
    val_frequency: int
    batch_size: int

    initial_checkpoint: str | None
    checkpoints_path: str
    checkpoint_frequency: int | None

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            Config: Configuration object with values loaded from YAML.
        """
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)

        # Convert dtype string to torch.dtype
        dtype_str = config_dict["hardware"]["dtype"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(dtype_str, torch.float32)

        logger.info(f"Loaded configuration from {config_path}")

        return cls(
            # Model parameters
            vocab_size=config_dict["model"]["vocab_size"],
            context_length=config_dict["model"]["context_length"],
            num_layers=config_dict["model"]["num_layers"],
            d_model=config_dict["model"]["d_model"],
            num_heads=config_dict["model"]["num_heads"],
            d_ff=config_dict["model"]["d_ff"],
            rope_theta=config_dict["model"]["rope_theta"],
            # Hardware
            device=config_dict["hardware"]["device"],
            dtype=dtype,
            # Training parameters
            max_lr=config_dict["training"]["max_lr"],
            min_lr=config_dict["training"]["min_lr"],
            warmup_iters=config_dict["training"]["warmup_iters"],
            cosine_cycle_iters=config_dict["training"]["cosine_cycle_iters"],
            betas=tuple(config_dict["training"]["betas"]),
            weight_decay=config_dict["training"]["weight_decay"],
            num_iterations=config_dict["training"]["num_iterations"],
            val_frequency=config_dict["training"]["val_frequency"],
            batch_size=config_dict["training"]["batch_size"],
            max_l2_norm=config_dict["training"]["max_l2_norm"],
            # Data paths
            train_path=config_dict["data"]["train_path"],
            val_path=config_dict["data"]["val_path"],
            # Checkpoints
            initial_checkpoint=config_dict["checkpoints"]["initial_checkpoint"],
            checkpoints_path=config_dict["checkpoints"]["checkpoints_path"],
            checkpoint_frequency=config_dict["checkpoints"]["checkpoint_frequency"],
        )


# ----------------------------------------------------------------------------------------------------------------------


def evaluate_model(model, val_dataset, config: Config):
    model.eval()
    total_loss = 0
    num_batches = 10

    with torch.inference_mode():
        for _ in range(num_batches):
            inputs, target = get_batch(
                dataset=val_dataset,
                batch_size=config.batch_size,
                context_length=config.context_length,
                device=config.device,
            )
            logits = model(inputs)
            loss = cross_entropy(y=target, logits=logits)
            total_loss += loss.item()

    model.train()
    return total_loss / num_batches


# ----------------------------------------------------------------------------------------------------------------------


@click.command()
@click.option("--config-path", default="config.yaml", help="Path to config file")
def main(config_path: str):
    config = Config.from_yaml(config_path)

    tokenizer = Tokenizer.from_files(
        "outputs/tokenizers/tinystories_bpe_vocab.json",
        "outputs/tokenizers/tinystories_bpe_merges.txt",
        ["<|endoftext|>"],
    )
    model = Transformer(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_model=config.d_model,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=torch.device(config.device),
        dtype=config.dtype,
    )

    optimizer = AdamW(model.parameters(), lr=0, betas=config.betas, weight_decay=config.weight_decay)

    start_iteration = 0
    if config.initial_checkpoint:
        logger.info(f"Loading checkpoint from {config.initial_checkpoint}")
        start_iteration = load_checkpoint(config.initial_checkpoint, model, optimizer)
        logger.info(f"Resumed training from iteration {start_iteration}")

    print(f"Total parameters: {model.count_parameters() / 10**6:.02f}M")

    train_dataset = np.load(config.train_path, mmap_mode="r")["arr_0"]
    val_dataset = np.load(config.val_path, mmap_mode="r")["arr_0"]

    # model = torch.compile(model)

    for it in range(start_iteration, config.num_iterations):
        inputs, targets = get_batch(
            train_dataset, batch_size=config.batch_size, context_length=config.context_length, device=config.device
        )
        logits = model(inputs)
        loss = cross_entropy(y=targets, logits=logits)

        loss.backward()

        lr = learning_rate_schedule(it, config.max_lr, config.min_lr, config.warmup_iters, config.cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        clip_gradients(model.parameters(), config.max_l2_norm)
        optimizer.step()
        optimizer.zero_grad()

        if it % config.val_frequency == 0 and it > 0:
            model.eval()
            val_loss = evaluate_model(model, val_dataset, config)
            model.train()
            logger.info(f"Iteration: {it}, val loss: {val_loss}")
            logger.info(model.generate_text("I'm a language model and", tokenizer, max_tokens=50))

        # Save checkpoint
        if config.checkpoint_frequency and (it + 1) % config.checkpoint_frequency == 0:
            checkpoint_path = Path(config.checkpoints_path) / f"checkpoint_iter_{it + 1}.pt"
            save_checkpoint(model, optimizer, it, str(checkpoint_path))
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        if it % 25 == 0:
            logger.info(f"Iteration: {it}, train loss: {loss.cpu().item()}")

    # Save final checkpoint
    final_checkpoint_path = Path(config.checkpoints_path) / "final_checkpoint.pt"
    save_checkpoint(model, optimizer, config.num_iterations - 1, final_checkpoint_path)
    logger.info(f"Saved final checkpoint to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
