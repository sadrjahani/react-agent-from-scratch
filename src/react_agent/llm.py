"""Thin chat wrapper for Ollama and OpenAI-compatible APIs.
The rest of the project calls only chat(). Provider and model settings
are read from environment variables, normally through a .env file.
"""

from collections.abc import Sequence
from typing import Any, TypedDict

from openai import OpenAI

from .config import (
    DEFAULT_OLLAMA_URL,
    REQUEST_TIMEOUT_SECONDS,
    SUPPORTED_PROVIDERS,
    Settings,
)

__all__ = ["Message", "Settings", "chat"]


class Message(TypedDict):
    """One chat message in the OpenAI format."""

    role: str
    content: str


def chat(
    messages: Sequence[Message],
    system: str = "",
    temperature: float = 0.0,
    stop: Sequence[str] | None = None,
) -> str:
    """Send a conversation to the configured provider and return reply text.

    Raises:
        ValueError: If messages are empty, the model is missing,
            or the configured provider is unsupported.
        RuntimeError: If the API returns no choices.
    """
    if not messages:
        raise ValueError("messages must not be empty")

    settings = Settings.from_env()

    if settings.provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unknown LLM_PROVIDER: {settings.provider!r}. "
            f"Use one of: {supported}"
        )

    if not settings.model:
        raise ValueError("LLM_MODEL is not set. Add it to your .env file.")

    if settings.provider == "ollama":
        base_url = settings.ollama_base_url
        # Ollama ignores the key, but the OpenAI SDK requires a non-empty key.
        api_key = "ollama"
    else:
        # Let the OpenAI SDK read OPENAI_API_KEY and OPENAI_BASE_URL.
        base_url = None
        api_key = None

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    full_messages: list[Message] = (
        [{"role": "system", "content": system}, *messages]
        if system
        else list(messages)
    )

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "messages": full_messages,
        "temperature": temperature,
    }

    if stop:
        kwargs["stop"] = list(stop)

    response = client.chat.completions.create(**kwargs)

    if not response.choices:
        raise RuntimeError("The chat API returned no choices.")

    return response.choices[0].message.content or ""


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Say hello in one short sentence."}]))
