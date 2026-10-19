"""Support desk triage: one decision answers intent, urgency and impact; our code computes the priority."""

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel

from decision_models.system_one import build_system_one_model
from decision_models.triage import MESSAGES, MessageClassification, print_classification


def main() -> None:
    load_dotenv()
    logfire.configure()
    logfire.instrument_pydantic_ai()

    support = Agent(
        # Round System One score answers, otherwise float noise from Ollama fails Pydantic AI's score check.
        FallbackModel(build_system_one_model('nimble'), 'openai:gpt-6.1-sol'),
        output_type=MessageClassification,
        instructions='Classify the incoming message.',
    )

    print('=' * 60)
    print('Model: system-one:nimble, fallback openai:gpt-6.1-sol')
    print('Instruction: Classify the incoming message.')
    print('Questions: intent (pick one), urgency and impact (rubric)')
    print('=' * 60)

    for message in MESSAGES:
        result = support.run_sync(message)
        print_classification(message, result)


if __name__ == "__main__":
    main()
