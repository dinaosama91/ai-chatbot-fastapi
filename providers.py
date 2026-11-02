"""LLM providers. Each one streams the assistant's reply as text chunks.

Switch providers with the LLM_PROVIDER environment variable ("anthropic" or "openai").
"""
import os
from typing import AsyncIterator, Protocol


class Provider(Protocol):
    def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        """Yield the reply as text chunks. `messages` alternate user/assistant, starting with user."""
        ...


class AnthropicProvider:
    def __init__(self, model: str, max_tokens: int):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.model = model
        self.max_tokens = max_tokens

    async def stream(self, system, messages):
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class OpenAIProvider:
    def __init__(self, model: str, max_tokens: int):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
        self.model = model
        self.max_tokens = max_tokens

    async def stream(self, system, messages):
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            stream=True,
            messages=[{"role": "system", "content": system}, *messages],
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
}


def get_provider() -> Provider:
    name = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    if name not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Use 'anthropic' or 'openai'.")

    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[name]
    max_tokens = int(os.getenv("MAX_TOKENS", "1024"))

    if name == "anthropic":
        return AnthropicProvider(model, max_tokens)
    return OpenAIProvider(model, max_tokens)
