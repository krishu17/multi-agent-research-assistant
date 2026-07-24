"""LLM provider abstraction.

Every node in this project talks to the model through `get_llm()`, so the
graph/crew logic never depends on a specific vendor SDK. Select the backend
with the LLM_PROVIDER environment variable:

    LLM_PROVIDER=openai      -> langchain-openai (needs OPENAI_API_KEY)
    LLM_PROVIDER=anthropic   -> langchain-anthropic (needs ANTHROPIC_API_KEY)
    LLM_PROVIDER=ollama      -> langchain-ollama, local model, no API key
    LLM_PROVIDER=mock        -> deterministic offline stand-in (default)

The mock backend exists so the agent graph, tool-calling loop, and tests
can run end-to-end with zero cost and zero network access while the real
providers are wired in. It is NOT a real model -- see mock_llm.py.
"""
from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "mock").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1"), temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
            temperature=temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=temperature)

    if provider == "mock":
        from .mock_llm import MockChatModel

        return MockChatModel()

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
