# Notes

Working notes for the decision-models project, to be turned into a blog
post later. Keep entries dated and informal.

## 2026-10-16

- Project bootstrapped with `uv init`.
- Dependencies: pydantic-ai, openai, ollama, python-dotenv (+ ruff, pytest for dev).
- Boilerplate: `settings.py` (env-based config) and `models.py` (provider-agnostic
  pydantic-ai model factory for OpenAI / Ollama). No agents yet.
