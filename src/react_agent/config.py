"""Shared configuration for the ReAct agent, read from environment variables."""

import os
from dataclasses import dataclass
from typing import Self

from dotenv import load_dotenv

DEFAULT_PROVIDER = "ollama"
SUPPORTED_PROVIDERS = ("ollama", "openai")
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
REQUEST_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class Settings:
    """Provider and model settings for the agent."""

    provider: str
    model: str
    ollama_base_url: str = DEFAULT_OLLAMA_URL

    @classmethod
    def from_env(cls) -> Self:
        """Read settings when called, allowing runtime environment changes."""
        load_dotenv()

        ollama_base_url = (
            os.getenv("OLLAMA_BASE_URL", "").strip()
            or DEFAULT_OLLAMA_URL
        )

        return cls(
            provider=os.getenv(
                "LLM_PROVIDER", DEFAULT_PROVIDER
            ).strip().lower(),
            model=os.getenv("LLM_MODEL", "").strip(),
            ollama_base_url=ollama_base_url,
        )
