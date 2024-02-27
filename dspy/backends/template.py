import typing as t

from pydantic import Field

from dspy.signatures.signature import Signature, signature_to_template
from dspy.primitives.example import Example

from .base import BaseBackend, StructuredOutput
from .lm.litellm import LiteLLM


class TemplateBackend(BaseBackend):
    model: LiteLLM

    def __call__(
        self,
        signature: Signature,
        demos: t.List[str] = [],
        **kwargs,
    ) -> list[StructuredOutput]:
        """Wrap the signature and demos into an example, and pass through the Language Model, returning Signature Compliant Output"""

        # Assert that all necessary params are provided
        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            # TODO: We should move this to logging
            print(
                f"WARNING: Not all input fields were provided to module. Present: {present}. Missing: {missing}."
            )

        # Generate Example
        example = Example(demos=demos, **kwargs)

        # Generate Template
        template = signature_to_template(signature)

        # Pass Through Language Model
        for input in signature.input_fields:
            del kwargs[input]

        pred = self.model(template(example), **kwargs)

        # This returns a list of Examples
        extracted = [template.extract(example, prediction) for prediction in pred]

        outputs = []
        for extract in extracted:
            outputs.append({})
            for key in signature.model_fields:
                outputs[-1][key] = extract.get(key)

        return outputs
