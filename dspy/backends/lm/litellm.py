import typing as t

import litellm
from litellm import completion, token_counter
from pydantic import Field


from .base import BaseLM


class LiteLLM(BaseLM):
    STANDARD_PARAMS: dict[str, t.Union[float, int]] = {
        "temperature": 0.0,
        "max_tokens": 150,
        "top_p": 1,
        # Why are we defining arguments for the user?
        # "frequency_penalty": 0,
        # "presence_penalty": 0,
    }

    model: str
    default_params: dict[str, t.Any] = Field(default_factory=dict)

    def __call__(
        self,
        prompt: str,
        n: int = 1,
        **kwargs,
    ) -> list[str]:
        """Generates `n` predictions for the signature output."""
        litellm.drop_params = True

        options = {**self.STANDARD_PARAMS, **self.default_params, **kwargs, "n": n}
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **options,
        )
        choices = [c for c in response["choices"] if c["finish_reason"] != "length"]
        return [c["message"]["content"] for c in choices]

    def count_tokens(self, prompt: str) -> int:
        """Counts the number of tokens for a specific prompt."""
        return token_counter(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
