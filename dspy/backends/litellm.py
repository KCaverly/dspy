import typing as t

from pydantic import Field

from dspy.signatures.signature import Signature, signature_to_template
from dspy.primitives.example import Example

from .base import BaseBackend, StructuredOutput
from .lm.litellm import LiteLLM


class LiteLLMBackend(BaseBackend):
    model: str
    lm: LiteLLM = Field(default_factory=LiteLLM)

    def __call__(
        self,
        signature: Signature,
        demos: t.List[str] = [],
        **kwargs,
    ) -> list[StructuredOutput]:
        # does this model support tool use? use that
        """
        Create tool from signature output fields and pass as tool to use
        json.loads tool_choice to create the
        """

        # does this model support JSON mode? use that
        """
        Define JSON format and pass as response_format
        json.loads from the messages in the response
        """

        # otherwise, wrap in a Template and pass through to the LM
        """
        See existing code in dspy/predict/predict.py lines 80-87 & 98-108 for
        how it's currently being done.
        This needs to get modified because we want the signature and params to get
        passed directly to ths Backend so we can do structured output using
        tool use / JSON mode
        """

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

        pred = self.lm(template(example), **kwargs)

        # This returns a list of Examples
        extracted = [template.extract(example, prediction) for prediction in pred]

        outputs = []
        for extract in extracted:
            outputs.append({})
            for key in signature.model_fields:
                outputs[-1][key] = extract.get(key)

        return outputs
