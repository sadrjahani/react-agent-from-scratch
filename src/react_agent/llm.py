# from react_agent.llm import chat
"""Thin chat wrapper: one function, three providers (Anthropic, OpenAI, Ollama)."""

import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
MODEL = os.getenv("LLM_MODEL", "")


def _chat_anthropic(messages, system, temperature, stop):
    import anthropic

    client = anthropic.Anthropic()
    kwargs = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system
    if stop:
        kwargs["stop_sequences"] = stop
    response = client.messages.create(**kwargs)
    return response.content[0].text


def _chat_openai_compatible(messages, system, temperature, stop, base_url=None, api_key=None):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    full_messages = ([{"role": "system", "content": system}] if system else []) + messages
    kwargs = {"model": MODEL, "messages": full_messages, "temperature": temperature}
    if stop:
        kwargs["stop"] = stop
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def chat(messages, system="", temperature=0.0, stop=None):
    """Send messages to the configured provider and return the reply text."""
    if not MODEL:
        raise ValueError("LLM_MODEL is not set. Add it to your .env file.")
    if PROVIDER == "anthropic":
        return _chat_anthropic(messages, system, temperature, stop)
    if PROVIDER == "openai":
        return _chat_openai_compatible(messages, system, temperature, stop)
    if PROVIDER == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return _chat_openai_compatible(
            messages, system, temperature, stop, base_url=base_url, api_key="ollama"
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Say hello in one short sentence."}]))