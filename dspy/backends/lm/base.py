import dspy
import os
from pathlib import Path
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel
from joblib import Memory


_cachedir = os.environ.get("DSP_CACHEDIR") or os.path.join(Path.home(), ".joblib_cache")
_cache_memory = Memory(_cachedir, verbose=0)


class BaseLM(BaseModel, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached = _cache_memory.cache(self._call)

    def __call__(self, prompt: str, **kwargs) -> list[str]:
        """Generates `n` predictions for the signature output."""
        if dspy.settings.cache:
            return self._cached(prompt, **kwargs)
        else:
            return self._call(prompt, **kwargs)

    @abstractmethod
    def forward(
        self,
        prompt: str,
        n: int = 1,
        **kwargs,
    ) -> list[str]:
        """Generates `n` predictions for the signature output."""
        ...

    @abstractmethod
    def count_tokens(self, prompt: str) -> int:
        """Counts the number of tokens for a specific prompt."""
        ...
