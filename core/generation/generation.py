from __future__ import annotations
from abc import ABC, abstractmethod
from core.registry import register
from core.llm_clients import get_client, LLMResponse
from core.retrieval.base import RetrievedItem

# Vision-capable models -- vision_grounded only actually attaches image bytes
# for these; other models silently fall back to caption-only text (the image
# context item is still cited by id, just described rather than seen).
_VISION_CAPABLE = {"claude", "gemini", "openai"}


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, query: str, contexts: list[RetrievedItem], **kwargs) -> LLMResponse: ...

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

    def generate(self, system_prompt: str, query: str, contexts: list[RetrievedItem],
                 history: list[dict] | None = None, vision_grounded: bool = False,
                 temperature: float | None = None, top_p: float | None = None,
                 top_k: int | None = None, max_tokens: int = 1024) -> LLMResponse:
        # Every context item's caption/text still goes in as before (so
        # citations like [img#id] always work); vision_grounded additionally
        # attaches the actual image bytes for image-kind items, on models
        # that can read them, so the answer can depend on what the figure
        # actually shows rather than just its caption.
        image_paths = None
        if vision_grounded and self._provider in _VISION_CAPABLE:
            image_paths = [c.meta.get("path") for c in contexts if c.kind == "image" and c.meta.get("path")]
            image_paths = image_paths or None

        user_msg = f"Context:\n{self._format_contexts(contexts)}\n\nQuestion: {query}"
        return self._client.chat(system_prompt, user_msg, model=self._model, max_tokens=max_tokens,
                                  history=history, temperature=temperature, top_p=top_p, top_k=top_k,
                                  images=image_paths)


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
