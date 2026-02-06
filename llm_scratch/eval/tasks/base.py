from __future__ import annotations

from abc import ABC, abstractmethod


class EvalTask(ABC):
    def __init__(self, start: int = 0, stop: int | None = None, step: int = 1):
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if stop is not None and stop < start:
            raise ValueError(f"stop must be >= start, got stop={stop}, start={start}")
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        self.start = start
        self.stop = stop
        self.step = step

    @property
    @abstractmethod
    def eval_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def num_examples(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_example(self, index: int) -> dict:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, conversation: dict, completion: str) -> bool | float:
        raise NotImplementedError

    def __len__(self) -> int:
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        span = stop - start
        if span < 0:
            raise ValueError("invalid slice span; stop is smaller than start")
        return (span + self.step - 1) // self.step

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, int):
            raise TypeError(f"index must be int, got {type(index)}")
        physical_index = self.start + index * self.step
        return self.get_example(physical_index)
