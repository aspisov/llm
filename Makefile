test:
	uv run pytest -v ./tests

run:
	uv run python llm_scratch/train/runners/lightning_runner.py trainer.fast_dev_run=true hardware.device=cpu checkpoints.initial_checkpoint=null
