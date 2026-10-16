"""Application settings loaded from the environment (and an optional .env file)."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the decision-models project.

    Values are read from environment variables or a local `.env` file.
    See `.env.example` for the supported keys.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Which provider to use when building the pydantic-ai model.
    model_provider: Literal["openai", "ollama"] = "ollama"

    # Model name for the chosen provider, e.g. "gpt-4o-mini" or "llama3.1".
    model_name: str = "llama3.1"

    # OpenAI configuration.
    openai_api_key: str | None = None

    # Ollama configuration (OpenAI-compatible endpoint).
    ollama_base_url: str = "http://localhost:11434/v1"


def get_settings() -> Settings:
    """Return a fresh `Settings` instance built from the current environment."""
    return Settings()
