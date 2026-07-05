from __future__ import annotations
from abc import ABC, abstractmethod
from core.registry import register
from core.llm_clients import get_client, LLMResponse
from core.retrieval.base import RetrievedItem


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, query: str, contexts: list[RetrievedItem]) -> LLMResponse: ...

    @staticmethod
    def _format_contexts(contexts: list[RetrievedItem]) -> str:
        lines = []
        for c in contexts:
            tag = f"[#{c.id}]" if c.kind == "text" else f"[img#{c.id}]"
            lines.append(f"{tag} {c.text}")
        return "\n".join(lines)


class _ProviderGenerator(BaseGenerator):
    _provider: str = ""
    _default_model: str = ""

    def __init__(self, api_key: str, model: str | None = None):
        self._client = get_client(self._provider, api_key)
        self._model = model or self._default_model

    def generate(self, system_prompt: str, query: str, contexts: list[RetrievedItem]) -> LLMResponse:
        user_msg = f"Context:\n{self._format_contexts(contexts)}\n\nQuestion: {query}"
        return self._client.chat(system_prompt, user_msg, model=self._model)


@register("generation", "claude-sonnet")
class ClaudeGenerator(_ProviderGenerator):
    _provider = "claude"
    _default_model = "claude-sonnet-4-6"


@register("generation", "gemini-2.5-flash")
class GeminiFlashGenerator(_ProviderGenerator):
    _provider = "gemini"
    _default_model = "gemini-2.5-flash"


@register("generation", "gemini-2.5-pro")
class GeminiProGenerator(_ProviderGenerator):
    _provider = "gemini"
    _default_model = "gemini-2.5-pro"


@register("generation", "gpt-4o-mini")
class OpenAIGenerator(_ProviderGenerator):
    _provider = "openai"
    _default_model = "gpt-4o-mini"


@register("generation", "llama-3.1-8b-instant")
class Llama8BGenerator(_ProviderGenerator):
    _provider = "groq"
    _default_model = "llama-3.1-8b-instant"


@register("generation", "llama-4-scout")
class LlamaScoutGenerator(_ProviderGenerator):
    _provider = "groq"
    _default_model = "meta-llama/llama-4-scout-17b-16e-instruct"
