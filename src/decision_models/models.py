"""Factory helpers for building pydantic-ai models from project settings."""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from decision_models.settings import Settings, get_settings


def build_model(settings: Settings | None = None) -> Model:
    """Build the pydantic-ai model configured by `settings`.

    Supports two providers:
    - "openai": uses the OpenAI API directly (requires `OPENAI_API_KEY`).
    - "ollama": uses a local Ollama instance through its OpenAI-compatible API.
    """
    settings = settings or get_settings()

    if settings.model_provider == "openai":
        provider = OpenAIProvider(api_key=settings.openai_api_key)
    elif settings.model_provider == "ollama":
        provider = OpenAIProvider(base_url=settings.ollama_base_url, api_key="ollama")
    else:  # pragma: no cover - guarded by Settings' Literal type
        raise ValueError(f"Unsupported model provider: {settings.model_provider}")

    return OpenAIChatModel(settings.model_name, provider=provider)
