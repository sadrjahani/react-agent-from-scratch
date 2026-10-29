"""Thin chat wrapper: one function for Ollama and any OpenAI-compatible API.

``chat()`` is the only function the rest of the project calls. Which provider
and model it uses comes from environment variables (normally a ``.env`` file),
so switching models never requires a code change.

Environment variables:
    LLM_PROVIDER: ``ollama`` (default) or ``openai``. ``openai`` means any
        OpenAI-compatible service, including free hosted ones.
    LLM_MODEL: Model name exactly as the provider spells it (required).
    OLLAMA_BASE_URL: Ollama's OpenAI-compatible address (optional).
    OPENAI_API_KEY / OPENAI_BASE_URL: read by the OpenAI SDK itself when
        ``LLM_PROVIDER=openai``.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self, TypedDict

from dotenv import load_dotenv
from openai import OpenAI

__all__ = ["Message", "Settings", "chat"]


DEFAULT_PROVIDER = "ollama"
SUPPORTED_PROVIDERS = ("ollama", "openai")
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
# A local model can take a while to load on the first request, so this is
# generous. It still stops the agent from hanging forever on a dead server.
REQUEST_TIMEOUT_SECONDS = 300

# Copies .env values into the environment. It does NOT override variables.
load_dotenv()


class Message(TypedDict):
    """One chat message in the OpenAI format."""

    role: str  # "user" or "assistant" (the system prompt is passed separately)
    content: str


@dataclass(frozen=True)
class Settings:
    """Provider and model chosen through environment variables."""

    provider: str
    model: str

    @classmethod
    def from_env(cls) -> Self:
        """Read the settings now.

        Reading at call time (not at import time) means changing an
        environment variable takes effect immediately, and tests can set it
        with ``monkeypatch.setenv``.
        """
        return cls(
            provider=os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower(),
            model=os.getenv("LLM_MODEL", "").strip(),
        )


def chat(
    messages: Sequence[Message],
    system: str = "",
    temperature: float = 0.0,
    stop: Sequence[str] | None = None,
) -> str:
    """Send a conversation to the configured provider and return the reply text.

    Args:
        messages: The conversation so far, oldest first. Must not be empty.
        system: Optional system prompt.
        temperature: 0.0 makes answers as repeatable as possible, which
            matters for an agent that must follow a strict output format.
        stop: Strings that end generation. The agent passes ``["Observation:"]``
            so the model cannot invent a tool result by itself.

    Returns:
        The model's reply as plain text (possibly empty).

    Raises:
        ValueError: ``messages`` is empty, ``LLM_MODEL`` is not set, or
            ``LLM_PROVIDER`` is not a supported provider.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    settings = Settings.from_env()
    if not settings.model:
        raise ValueError("LLM_MODEL is not set. Add it to your .env file.")

    if settings.provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
        # Ollama ignores the key, but the OpenAI SDK refuses an empty one,
        # so any non-empty placeholder works.
        api_key = "ollama"
    elif settings.provider == "openai":
        # None lets the SDK read OPENAI_BASE_URL and OPENAI_API_KEY itself.
        base_url = None
        api_key = None
    else:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.provider!r}. Use one of: {supported}")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
    
    full_messages: list[Message] = (
        [{"role": "system", "content": system}, *messages] if system else list(messages)
    )
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "messages": full_messages,
        "temperature": temperature,
    }
    if stop:
        kwargs["stop"] = list(stop)
    response = client.chat.completions.create(**kwargs)
    # The API can return None content (for example when only a tool call was made).
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Say hello in one short sentence."}]))