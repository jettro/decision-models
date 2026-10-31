# Decision Models

A playground project to showcase what decision models can do, built with
[pydantic-ai](https://ai.pydantic.dev/), [OpenAI](https://platform.openai.com/)
and [Ollama](https://ollama.com/).

## Status

A set of runnable samples that compare decision models: a yes/no guardrail
on a local Ollama model and OpenAI `gpt-6-luna`, and a support desk triage
with a local decision model and a cloud language model as fallback.

The triage and handoff samples use Pydantic AI's native `system-one:nimble`
model. Pydantic AI 2.60.0 fixed validation of full-precision scores and
probabilities ([PR #11092](https://github.com/pydantic/pydantic-ai/pull/11092));
this project requires 2.62.0 or later and no longer rounds HTTP responses.

## Running the samples

```bash
uv run decision-models                   # list the samples
uv run decision-models guardrail         # nimble and gpt-6-luna through Pydantic AI
uv run decision-models guardrail-openai  # the raw OpenAI Decisions SDK
uv run decision-models triage            # support desk triage with priority
uv run decision-models handoff           # hand-off to the fallback language model
```

`make run SAMPLE=triage` and `uv run python -m decision_models.samples.triage`
do the same.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- (optional) [Ollama](https://ollama.com/) running locally if using the
  `ollama` provider
- (optional) an OpenAI API key if using the `openai` provider

## Setup

```bash
make install
cp .env.example .env   # then edit values as needed
```

## Configuration

Settings are read from environment variables or a local `.env` file (see
`.env.example`):

| Variable          | Description                                      | Default                        |
|-------------------|---------------------------------------------------|---------------------------------|
| `MODEL_PROVIDER`  | `openai` or `ollama`                              | `ollama`                        |
| `MODEL_NAME`      | Model name for the chosen provider                | `llama3.1`                      |
| `OPENAI_API_KEY`  | API key, required when using `openai`             | -                                |
| `OLLAMA_BASE_URL` | Ollama's OpenAI-compatible endpoint               | `http://localhost:11434/v1`     |

For the `triage` and `handoff` samples, set `SYSTEM_ONE_BASE_URL` to your
System One endpoint, e.g. `http://localhost:11434/v1` for Ollama, and install
the `nimble` model (`ollama pull nimble`). These samples also need
`OPENAI_API_KEY` for their fallback language model. `SYSTEM_ONE_API_KEY` is
optional if your System One server requires bearer authentication.

## Project layout

```
src/decision_models/
├── __init__.py      # the decision-models command
├── settings.py      # environment-based configuration
├── models.py        # pydantic-ai model factory (OpenAI / Ollama)
├── priority.py      # escalation, priority and review on top of a triage
├── triage.py        # shared triage types, messages and printing
└── samples/         # one module with a main() per sample
    ├── guardrail.py
    ├── guardrail_openai.py
    ├── triage.py
    └── handoff.py
```

## Common tasks

See the `Makefile` for available tasks:

```bash
make install   # sync dependencies
make run       # list the samples, or run one with SAMPLE=<name>
make lint      # run ruff checks
make format    # auto-format with ruff
make test      # run the test suite
make clean     # remove caches and build artifacts
```

## Notes

Working notes and observations are kept in `notes.md` for turning into a
future blog post.
