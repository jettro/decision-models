# Decision Models

A playground project to showcase what decision models can do, built with
[pydantic-ai](https://ai.pydantic.dev/), [OpenAI](https://platform.openai.com/)
and [Ollama](https://ollama.com/).

## Status

Early boilerplate stage. The project currently only wires up configuration
and a provider-agnostic model factory (OpenAI or Ollama) via pydantic-ai.

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

## Project layout

```
src/decision_models/
├── __init__.py    # entry point
├── settings.py     # environment-based configuration
└── models.py        # pydantic-ai model factory (OpenAI / Ollama)
```

## Common tasks

See the `Makefile` for available tasks:

```bash
make install   # sync dependencies
make run       # run the project entry point
make lint      # run ruff checks
make format    # auto-format with ruff
make test      # run the test suite
make clean     # remove caches and build artifacts
```

## Notes

Working notes and observations are kept in `notes.md` for turning into a
future blog post.
