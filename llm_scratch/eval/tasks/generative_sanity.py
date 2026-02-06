from __future__ import annotations

from llm_scratch.eval.tasks.base import EvalTask


class GenerativeSanityTask(EvalTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rows = [
            {"prompt": "Complete: the sky is", "answer": "blue"},
            {"prompt": "Complete: water is", "answer": "wet"},
        ]

    @property
    def eval_type(self) -> str:
        return "generative"

    def num_examples(self) -> int:
        return len(self._rows)

    def get_example(self, index: int) -> dict:
        row = self._rows[index]
        return {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["answer"]},
            ]
        }

    def evaluate(self, conversation: dict, completion: str) -> bool:
        if not isinstance(completion, str):
            raise TypeError("completion must be a string")
        return completion.strip() == conversation["messages"][-1]["content"]
