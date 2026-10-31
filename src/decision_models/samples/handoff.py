"""Support desk triage where the fallback language model takes over the steps the decision model hands off."""

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.models.decision import DecisionHandOff, UnfillableRoute, UnsureRoute
from pydantic_ai.models.fallback import FallbackModel

from decision_models.triage import MESSAGES, MessageClassification, print_classification

LANGUAGE_MODEL = 'openai:gpt-6.1-sol'
INSTRUCTIONS = 'Classify the incoming message, or answer it directly if it is a simple question.'


class DraftReply(BaseModel):
    """Write a short answer for a simple question that we can send right away."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    reply: str
    """The answer to send to the user."""


class HandOffRecorder:
    """`fallback_on` handler that only falls back on a hand-off, and remembers the last one for printing."""

    def __init__(self) -> None:
        self.last: DecisionHandOff | None = None

    def __call__(self, error: Exception) -> bool:
        if isinstance(error, DecisionHandOff):
            self.last = error
            return True
        return False


def build_agent(recorder: HandOffRecorder) -> Agent:
    return Agent(
        # Only real hand-offs go to the language model; if Ollama is down, the run fails instead.
        FallbackModel('system-one:nimble', LANGUAGE_MODEL, fallback_on=recorder),
        output_type=[MessageClassification, DraftReply],
        instructions=INSTRUCTIONS,
    )


def describe_hand_off(error: DecisionHandOff) -> str:
    match error:
        case UnsureRoute():
            routes = ' | '.join(f'{label} {p:.2f}' for label, p in error.probabilities.items())
            return f'UnsureRoute: picked {error.route!r} ({error.probability:.2f}) below threshold {error.threshold:.2f}; {routes}'
        case UnfillableRoute():
            return f'UnfillableRoute: picked {error.route!r} ({error.probability:.2f}), but cannot fill it'
        case _:
            return f'{type(error).__name__}: {error.route!r} ({error.probability:.2f})'


def print_route(details: dict) -> None:
    route = details.get('route')
    if not route:
        return
    distribution = ' | '.join(f'{label} {p:.2f}' for label, p in route.get('probabilities', {}).items())
    print(f'  {"Route:":<12} {route.get("choice")}   ({distribution})')


def print_result(message: str, result, hand_off: DecisionHandOff | None) -> None:
    response = result.response
    details = response.provider_details or {}
    if hand_off is not None:
        print(f'  {"Hand-off:":<12} {describe_hand_off(hand_off)}')
    print_route(details)

    match result.output:
        case MessageClassification() if hand_off is None:
            print_classification(message, result)
        case MessageClassification() as output:
            print(f'  Message:     "{message}"')
            print(f'  Answered by: {response.provider_name}:{response.model_name}')
            print(f'  {"Output:":<12} {output!r}')
            print(f'  {"":<12} answered by the fallback, no distribution: no escalation or review check possible')
            print()
        case DraftReply(reply=reply):
            print(f'  Message:     "{message}"')
            print(f'  Answered by: {response.provider_name}:{response.model_name}')
            print(f'  {"Reply:":<12} {reply}')
            print()


def run(title: str, model_settings: dict | None = None) -> None:
    recorder = HandOffRecorder()
    support = build_agent(recorder)

    print('=' * 60)
    print(title)
    print(f'Model: system-one:nimble, fallback {LANGUAGE_MODEL} (only on DecisionHandOff)')
    print(f'Instruction: {INSTRUCTIONS}')
    print('Output types: MessageClassification (decision model) or DraftReply (free text, needs the fallback)')
    if model_settings:
        print(f'Settings: {model_settings}')
    print('=' * 60)

    for message in MESSAGES:
        recorder.last = None
        result = support.run_sync(message, model_settings=model_settings)
        print_result(message, result, recorder.last)


def main() -> None:
    load_dotenv()
    logfire.configure()
    logfire.instrument_pydantic_ai()

    run('Variant 1: hand off a route the decision model cannot fill (UnfillableRoute)')
    run(
        'Variant 2: also hand off a route pick the decision model is unsure of (UnsureRoute)',
        model_settings={'decision_route_threshold': 0.6},
    )


if __name__ == '__main__':
    main()
