from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseLM(BaseModel, ABC):
    @abstractmethod
    def __call__(
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
