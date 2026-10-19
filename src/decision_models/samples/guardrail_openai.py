"""Yes/no guardrail on `gpt-6-luna` with the raw OpenAI Decisions SDK.

The question is a `predicate`, the same kind of question Pydantic AI sends for an `output_type=bool`.
A predicate answer only carries the probability of "yes", so the verdict and the confidence are
computed here, the same way Pydantic AI does it.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from decision_models.samples.guardrail import PROMPTS

INSTRUCTION = "Is this request harmful?"
THRESHOLD = 0.5
"""How likely a yes has to be before the request counts as harmful (Pydantic AI's default)."""


def verdict(probability: float, threshold: float = THRESHOLD) -> tuple[bool, float]:
    """Whether the probability of yes clears the threshold, and how far from it the answer landed (0 to 1).

    With the default threshold of 0.5 the confidence is `2 * |p - 0.5|`: 0.85 gives 0.70, 0.5 gives 0.
    """
    if probability >= threshold:
        confidence = (probability - threshold) / (1 - threshold)
    else:
        confidence = (threshold - probability) / threshold
    return probability >= threshold, round(confidence, 6)


def print_decision_result(prompt: str, probability: float):
    """Print structured and readable evaluation results for an OpenAI decision query."""
    is_harmful, confidence = verdict(probability)
    label = "HARMFUL" if is_harmful else "SAFE"
    print(f'  Prompt:      "{prompt}"')
    print(f"  Evaluation:  {label} (is_harmful={is_harmful})")
    print(f"  P(harmful):  {probability:.2f}")
    print(f"  Confidence:  {confidence:.2f}")
    print()


def call_openai_decision(model_input: str, _client: OpenAI):
    decision = _client.decisions.create(
        input=model_input,
        model="gpt-6-luna",
        questions=[{
            "name": "is_harmful",
            "type": "predicate",
            "instructions": INSTRUCTION,
        }],
    )
    answer = decision.answers[0]
    print_decision_result(model_input, answer.probability)


def main() -> None:
    load_dotenv()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
    )

    print("=" * 60)
    print("Model: gpt-6-luna (OpenAI Decisions SDK, predicate)")
    print(f"Instruction: {INSTRUCTION}")
    print("=" * 60)

    for prompt in PROMPTS:
        call_openai_decision(model_input=prompt, _client=client)


if __name__ == "__main__":
    main()
