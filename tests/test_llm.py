"""Tests for react_agent.llm: configuration errors and request building.

No network is used; the OpenAI client is replaced with a small fake.
"""

from types import SimpleNamespace

import pytest

from react_agent import llm
from react_agent.llm import chat

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without LLM settings (a real .env may have set some)."""
    for name in ("LLM_PROVIDER", "LLM_MODEL", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace the OpenAI client with a fake and return a record of its use."""
    record = SimpleNamespace(init=None, calls=[], reply="hello")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            record.init = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            record.calls.append(kwargs)
            message = SimpleNamespace(content=record.reply)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    return record


def test_missing_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="LLM_MODEL"):
        chat(MESSAGES)


def test_empty_messages_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(ValueError, match="messages"):
        chat([])


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_PROVIDER", "foo")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        chat(MESSAGES)


def test_default_provider_is_ollama(monkeypatch: pytest.MonkeyPatch, fake_openai) -> None:
    monkeypatch.setenv("LLM_MODEL", "phi4-mini")
    assert chat(MESSAGES) == "hello"
    assert fake_openai.init["base_url"] == llm.DEFAULT_OLLAMA_URL
    assert fake_openai.calls[0]["model"] == "phi4-mini"


def test_system_prompt_is_first_message_and_stop_is_passed(
    monkeypatch: pytest.MonkeyPatch, fake_openai
) -> None:
    """loop.py relies on both: the system prompt and stop=["Observation:"]."""
    monkeypatch.setenv("LLM_MODEL", "m")
    chat(MESSAGES, system="be brief", stop=["Observation:"])
    call = fake_openai.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "be brief"}
    assert call["messages"][1:] == MESSAGES
    assert call["stop"] == ["Observation:"]


def test_none_content_becomes_empty_string(monkeypatch: pytest.MonkeyPatch, fake_openai) -> None:
    """Some servers return None content; the agent should get '' instead."""
    monkeypatch.setenv("LLM_MODEL", "m")
    fake_openai.reply = None
    assert chat(MESSAGES) == ""