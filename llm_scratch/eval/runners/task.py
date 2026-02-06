from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig

from llm_scratch.eval.tasks.base import EvalTask


def _oracle_prediction(conversation: dict) -> str:
    assistant_message = conversation["messages"][-1]
    if assistant_message["role"] != "assistant":
        raise ValueError("conversation must end with an assistant message")
    content = assistant_message["content"]
    if not isinstance(content, str):
        raise TypeError("oracle prediction requires assistant content to be a string")
    return content


def evaluate_with_task_runner(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    task = hydra.utils.instantiate(cfg.eval_task)
    if not isinstance(task, EvalTask):
        raise TypeError("eval_task must instantiate an EvalTask")

    max_examples = int(cfg.eval.max_examples)
    if max_examples <= 0:
        raise ValueError("eval.max_examples must be > 0")

    prediction_mode = str(cfg.eval.prediction_mode)
    if prediction_mode not in {"oracle", "constant"}:
        raise ValueError(f"Unsupported eval.prediction_mode: {prediction_mode}")

    total = min(max_examples, len(task))
    if total == 0:
        raise ValueError("Eval task has zero examples")

    scores: list[float] = []
    for i in range(total):
        conversation = task[i]

        if prediction_mode == "oracle":
            prediction = _oracle_prediction(conversation)
        else:
            if "constant_prediction" not in cfg.eval:
                raise KeyError("eval.constant_prediction must be set when prediction_mode=constant")
            prediction = str(cfg.eval.constant_prediction)

        score = task.evaluate(conversation, prediction)
        scores.append(float(score))

    mean_score = sum(scores) / len(scores)
    metric_dict = {
        "task/score": mean_score,
        "task/num_examples": float(total),
    }
    object_dict = {
        "cfg": cfg,
        "task": task,
    }
    return metric_dict, object_dict
