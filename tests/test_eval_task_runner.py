import pytest
from omegaconf import OmegaConf

from llm_scratch.eval.runners.task import evaluate_with_task_runner


def test_task_runner_oracle_mode_scores_perfect_for_categorical_sanity():
    cfg = OmegaConf.create(
        {
            "eval_task": {"_target_": "llm_scratch.eval.tasks.categorical_sanity.CategoricalSanityTask"},
            "eval": {"max_examples": 2, "prediction_mode": "oracle"},
        }
    )

    metrics, _ = evaluate_with_task_runner(cfg)
    assert metrics["task/score"] == 1.0
    assert metrics["task/num_examples"] == 2.0


def test_task_runner_constant_mode_requires_constant_prediction():
    cfg = OmegaConf.create(
        {
            "eval_task": {"_target_": "llm_scratch.eval.tasks.generative_sanity.GenerativeSanityTask"},
            "eval": {"max_examples": 1, "prediction_mode": "constant"},
        }
    )

    with pytest.raises(KeyError, match="eval.constant_prediction"):
        evaluate_with_task_runner(cfg)


def test_task_runner_rejects_non_positive_max_examples():
    cfg = OmegaConf.create(
        {
            "eval_task": {"_target_": "llm_scratch.eval.tasks.generative_sanity.GenerativeSanityTask"},
            "eval": {"max_examples": 0, "prediction_mode": "oracle"},
        }
    )

    with pytest.raises(ValueError, match="eval.max_examples must be > 0"):
        evaluate_with_task_runner(cfg)
