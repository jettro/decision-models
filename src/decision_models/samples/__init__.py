"""Runnable samples. Each module has a `main()`; run one with `uv run decision-models <name>`."""

SAMPLES = {
    'guardrail': 'Yes/no guardrail on local nimble and gpt-6-luna, through Pydantic AI (Part 1).',
    'guardrail-openai': 'The same guardrail with the raw OpenAI Decisions SDK (Part 1).',
    'triage': 'Support desk triage: intent, urgency, impact and priority (Part 2).',
    'handoff': 'Support desk triage where the fallback language model takes over hand-offs (Part 2).',
}


def module_name(sample: str) -> str:
    """Return the module that implements a sample, e.g. `guardrail-openai` -> `decision_models.samples.guardrail_openai`."""
    return f'{__name__}.{sample.replace("-", "_")}'
