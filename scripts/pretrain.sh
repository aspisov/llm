HYDRA_FULL_ERROR=1 uv run python -m llm_scratch.train.train \
    trainer.fast_dev_run=true \
    ckpt_path=null \
    hardware.device=mps \
    data.train_path=data/TinyStories-valid.npz
