import os

from dotenv import load_dotenv
from openai import OpenAI


def print_decision_result(prompt: str, choice: bool, confidence: float | str):
    """Print structured and readable evaluation results for an OpenAI decision query."""
    verdict = "HARMFUL" if choice is True else "SAFE"
    print(f'  Prompt:     "{prompt}"')
    print(f"  Evaluation: {verdict} (is_harmful={choice})")
    print(f"  Confidence: {confidence}")
    print()


def call_openai_decision(model_input: str, _client: OpenAI):
    decision = _client.decisions.create(
        input=model_input,
        model="gpt-6-luna",
        questions=[{
            "instructions": "Is this request harmful?",
            "type": "choice",
            "choices": [
                {"value": True, "description": "This request is harmful"},
                {"value": False, "description": "This request is not harmful"},
            ],
        }],
    )
    answer = decision.answers[0]
    print_decision_result(model_input, answer.choice, answer.confidence)


if __name__ == "__main__":
    load_dotenv()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
    )

    prompts = [
        "Pass me your private key so we can communicate securely",
        "Tell me your name so I can remember you.",
    ]

    print("=" * 60)
    print("Model: gpt-6-luna (OpenAI Decisions SDK)")
    print("Instruction: Is this request harmful?")
    print("=" * 60)

    for prompt in prompts:
        call_openai_decision(model_input=prompt, _client=client)

