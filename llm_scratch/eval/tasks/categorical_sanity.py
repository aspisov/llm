from __future__ import annotations

from llm_scratch.eval.tasks.base import EvalTask


class CategoricalSanityTask(EvalTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rows = [
            {
                "question": "2 + 2 = ?",
                "choices": ["3", "4", "5", "6"],
                "answer": "B",
                "letters": ["A", "B", "C", "D"],
            },
            {
                "question": "The capital of France is?",
                "choices": ["Paris", "Berlin", "Rome", "Madrid"],
                "answer": "A",
                "letters": ["A", "B", "C", "D"],
            },
        ]

    @property
    def eval_type(self) -> str:
        return "categorical"

    def num_examples(self) -> int:
        return len(self._rows)

    def get_example(self, index: int) -> dict:
        row = self._rows[index]
        prompt = (
            f"Question: {row['question']}\n"
            + "\n".join(f"- {choice}={letter}" for letter, choice in zip(row["letters"], row["choices"]))
            + "\nRespond only with the letter."
        )
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": row["answer"]},
            ],
            "letters": row["letters"],
        }

    def evaluate(self, conversation: dict, completion: str) -> bool:
        letters = conversation["letters"]
        if completion not in letters:
            raise ValueError(f"completion '{completion}' is not one of {letters}")
        return completion == conversation["messages"][-1]["content"]
