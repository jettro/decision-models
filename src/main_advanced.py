from enum import Enum, IntEnum, StrEnum

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, UseEnumMemberDocstrings
from pydantic_ai.models.fallback import FallbackModel

from decision_models.system_one import build_system_one_model


# Or using Intent semantics:
class MessageIntent(UseEnumMemberDocstrings, StrEnum):
    COMPLIMENT = "compliment"
    """A compliment is a positive remark or expression of praise or admiration."""

    COMPLAINT = "complaint"
    """A complaint is a statement or expression of dissatisfaction or unhappiness."""

    QUESTION = "question"
    """A question is a request for information or clarification."""

class Urgency(UseEnumMemberDocstrings, IntEnum):
    LOW = 0
    """Low urgency means that the message can wait."""
    MEDIUM = 1
    """Medium urgency means that the message should be addressed soon."""
    HIGH = 2
    """High urgency means that the message needs to be addressed immediately."""
    CRITICAL = 3
    """Critical urgency means that the message is a matter of life and death."""

class Impact(UseEnumMemberDocstrings, IntEnum):
    LOW = 0
    """Low impact means that the user will stick around."""
    MEDIUM = 1
    """Medium impact means that the user might leave."""
    HIGH = 2
    """High impact means that we can loose the user."""

class MessageClassification(BaseModel):
    """Classify the incoming message based on its intent, urgency, and impact."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    message_type: MessageIntent
    """Categorization of the incoming message based on its tone and objective"""

    urgency: Urgency
    """Urgency of the message to determine when it needs to be addressed."""

    impact: Impact
    """Impact of the message to determine how it affects the user."""


MESSAGES = [
    'I have been trying to generate an image 10 times, but I keep being logged out.',
    'I need to pay for my dinner, but apple pay is failing and I do not have money.',
    'I noticed an advertorial and I would like to have more information on your offer.',
    'I am done with everything you offer. I want to unsubscribe and you need to delete my account.',
    'I love what you are doing, please continue and allow me to help.',
    'Tell be how I can logout from the app.',
]


def format_distribution(enum_type: type[Enum], probabilities: dict[str, float] | None) -> str:
    """Format the probability per option, using the enum member names, e.g. `LOW 0.57 | MEDIUM 0.29`."""
    if not probabilities:
        return ''
    parts = []
    for member in enum_type:
        probability = probabilities.get(str(member.value))
        if probability is not None:
            parts.append(f'{member.name} {probability:.2f}')
    return ' | '.join(parts)


def print_classification(message: str, result) -> None:
    """Print the classification of a message with the confidence and distribution of every answer."""
    output: MessageClassification = result.output
    response = result.response
    details = response.provider_details or {}
    confidence = details.get('confidence', {})
    probabilities = details.get('probabilities', {})
    scores = details.get('scores', {})

    print(f'  Message:     "{message}"')
    print(f'  Answered by: {response.provider_name}:{response.model_name}')
    for label, field, value in (
        ('Intent', 'message_type', output.message_type),
        ('Urgency', 'urgency', output.urgency),
        ('Impact', 'impact', output.impact),
    ):
        measures = []
        if field in scores:
            measures.append(f'score {scores[field]:.2f}')
        if field in confidence:
            measures.append(f'confidence {confidence[field]:.2f}')
        line = f'  {label + ":":<12} {value.name:<11}'
        if measures:
            line += f' ({", ".join(measures)})'
        print(line)
        distribution = format_distribution(type(value), probabilities.get(field))
        if distribution:
            print(f'  {"":<12}   {distribution}')
    print()


if __name__ == "__main__":
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
