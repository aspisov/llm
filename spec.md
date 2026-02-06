# LLM Pipeline Spec (v1)

## Goal
Build a clean, Hydra-first repository that supports the full LLM lifecycle:
- data preparation
- tokenizer training/evaluation
- training (pretrain, midtrain, posttrain)
- benchmark evaluation
- inference
- serving (chat UI backend)

Fail-fast is required everywhere: invalid config/state should raise errors, not fallback.

## Canonical entrypoints
- `llm_scratch/train/train.py`
- `llm_scratch/train/eval.py`
- `llm_scratch/infer/infer.py` (new)
- `llm_scratch/serve/serve.py` (new)
- `llm_scratch/data/prepare.py` (new)
- `llm_scratch/tokenizer/train.py` (new)
- `llm_scratch/tokenizer/eval.py` (new)
- `llm_scratch/export/export.py` (new)

## Training stages
Training is routed by `stage`:
- `pretrain`
- `midtrain`
- `sft`
- `dpo`
- `ppo`
- `rlvr`

Rule:
- use Lightning for supervised stages (`pretrain`, `midtrain`, `sft`, `dpo`)
- use pure torch loops for rollout RL stages (`ppo`, `rlvr`)

## Hydra config layout
Keep top-level:
- `configs/train.yaml`
- `configs/eval.yaml`

Add config groups:
- `configs/stage/*.yaml`
- `configs/eval_task/*.yaml`
- `configs/infer/*.yaml`
- `configs/serve/*.yaml`
- `configs/data_prepare/*.yaml`
- `configs/tokenizer/*.yaml`
- `configs/export/*.yaml`

Each `stage` config must define:
- stage runner `_target_`
- objective/loss mode
- required data/task configs
- backend type (`lightning` or `torch`)

## Eval task system
Adopt a task contract similar to nanochat:
- `eval_type`: `categorical` or `generative`
- `num_examples()`
- `get_example(index)`
- `evaluate(conversation, completion)`
- optional `reward(conversation, completion)` for RL-compatible tasks

Conversation object:
- `messages: list[{role, content}]`
- `content` supports plain text and structured parts when needed.

## Inference and serving split
- `infer.py`: offline generation, decoding variants, KV-cache experiments, latency/throughput checks.
- `serve.py`: API/websocket layer for chat UI; uses inference engine internally.

Serving must not duplicate model/decoding logic from inference engine.

## Artifact contract
Every run writes:
- `src.zip`
- `config_not_resolved.yaml`
- `config.yaml`
- `metadata.json`

Stage outputs are explicit:
- checkpoints
- tokenizer files
- eval reports
- exported runtime artifacts

## Validation criteria
- All entrypoints compose and instantiate from Hydra without manual factories.
- Invalid stage/task configs fail with explicit errors.
- At least one categorical and one generative eval task pass smoke tests.
- Inference supports deterministic parity checks with/without KV-cache (same settings).
- Serve endpoint passes single-turn chat smoke test.

## Non-goals for v1
- distributed orchestration redesign
- monthly-scale experiment management tools
- advanced model families (MoE, speculative decoding) beyond placeholders
