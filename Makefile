test:
	uv run pytest

run:
	HYDRA_FULL_ERROR=1 uv run python llm_scratch/train/runners/lightning_runner.py trainer.fast_dev_run=true checkpoints.initial_checkpoint=null
