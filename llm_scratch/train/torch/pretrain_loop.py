import logging
from pathlib import Path

import numpy as np
import torch
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
from llm_scratch.core.model import Transformer, cross_entropy
from llm_scratch.core.optimizers import AdamW, clip_gradients, learning_rate_schedule
from llm_scratch.core.tokenizer_py import Tokenizer
from llm_scratch.train.config import Config
from llm_scratch.utils import get_batch, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


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


@hydra_main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")

    config = Config.from_dict(OmegaConf.to_container(cfg, resolve=True))

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

    model = torch.compile(model)
    scaler = torch.amp.GradScaler() if config.dtype == "float16" else None

    for it in range(start_iteration, config.num_iterations):
        inputs, targets = get_batch(
            train_dataset, batch_size=config.batch_size, context_length=config.context_length, device=config.device
        )
        with torch.autocast(device_type=config.device, dtype=config.dtype):
            logits = model(inputs)
            loss = cross_entropy(y=targets, logits=logits)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        lr = learning_rate_schedule(it, config.max_lr, config.min_lr, config.warmup_iters, config.cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        clip_gradients(model.parameters(), config.max_l2_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        if it % config.val_frequency == 0 and it > 0:
            model.eval()
            val_loss = evaluate_model(model, val_dataset, config)
            model.train()
            logger.info(f"Iteration: {it}, val loss: {val_loss}")
            logger.info(model.generate_text("I'm a language model and", tokenizer, max_tokens=256, top_p=0.4))

        # Save checkpoint
        if config.checkpoint_frequency and (it + 1) % config.checkpoint_frequency == 0:
            checkpoint_path = Path(config.checkpoints_path) / f"checkpoint_iter_{it + 1}.pt"
            save_checkpoint(model, optimizer, it, str(checkpoint_path))
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        if it % 100 == 0:
            logger.info(f"Iteration: {it}, train loss: {loss.cpu().item()}")

    # Save final checkpoint
    final_checkpoint_path = Path(config.checkpoints_path) / "final_checkpoint.pt"
    save_checkpoint(model, optimizer, config.num_iterations - 1, final_checkpoint_path)
    logger.info(f"Saved final checkpoint to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
