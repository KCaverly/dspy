import typing as t
from abc import ABC, abstractmethod
from pydantic import BaseModel

from dspy.signatures.signature import Signature

StructuredOutput = t.TypeVar("StructuredOutput", bound=dict[str, t.Any])


class BaseBackend(BaseModel, ABC):
    """A backend takes a signature, its params, and returns a list of structured predictions."""

    @abstractmethod
    def __call__(
        self, signature: Signature, demos: t.List[str], **kwargs
    ) -> list[StructuredOutput]:
        """Generates `n` predictions for the signature output."""
        ...
