"""
One interface for every place the pipeline needs to call an LLM:
generation, "Rewrite with LLM" (system prompt), the per-stage "Suggest"
advisor, and image captioning. Each provider is a thin adapter so adding a
4th provider later is one small class, not a rewrite of every caller.

API keys are passed in explicitly per call (never read from a hardcoded
default) -- the UI holds them in memory only, per the platform's own
credential-handling rule.
"""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, system: str, user: str, model: str, max_tokens: int = 1024,
              history: list[dict] | None = None, temperature: float | None = None,
              top_p: float | None = None, top_k: int | None = None,
              images: list[Path] | None = None) -> LLMResponse: ...

    @abstractmethod
    def caption_image(self, image_path, model: str) -> LLMResponse: ...


# ---- Pricing table (USD per 1K tokens, input/output) -- used only to show
# an estimated cost in traces; keep it approximate and easy to update. ----
PRICING = {
    "claude-sonnet-4-6": (0.003, 0.015),
    "gemini-2.5-flash": (0.000075, 0.0003),
    "gemini-2.5-pro": (0.00125, 0.005),
    "gpt-4o-mini": (0.00015, 0.0006),
    "llama-3.1-8b-instant": (0.00005, 0.00008),
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.00011, 0.00034),
}


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    rates = PRICING.get(model, (0.0, 0.0))
    return round(in_tok / 1000 * rates[0] + out_tok / 1000 * rates[1], 6)


def _image_block_anthropic(image_path: Path) -> dict:
    import base64
    media_type = "image/png" if str(image_path).endswith("png") else "image/jpeg"
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}


def _image_block_openai(image_path: Path) -> dict:
    import base64
    media_type = "image/png" if str(image_path).endswith("png") else "image/jpeg"
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}


class ClaudeClient(BaseLLMClient):
    def __init__(self, api_key: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(self, system: str, user: str, model: str = "claude-sonnet-4-6", max_tokens: int = 1024,
              history: list[dict] | None = None, temperature: float | None = None,
              top_p: float | None = None, top_k: int | None = None,
              images: list[Path] | None = None) -> LLMResponse:
        user_content = user
        if images:
            user_content = [_image_block_anthropic(p) for p in images] + [{"type": "text", "text": user}]
        messages = [*(history or []), {"role": "user", "content": user_content}]
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        resp = self._client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=messages, **kwargs,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(text, resp.usage.input_tokens, resp.usage.output_tokens,
                            _cost(model, resp.usage.input_tokens, resp.usage.output_tokens))

    def caption_image(self, image_path, model: str = "claude-sonnet-4-6") -> LLMResponse:
        resp = self._client.messages.create(
            model=model, max_tokens=200,
            messages=[{"role": "user", "content": [
                _image_block_anthropic(image_path),
                {"type": "text", "text": "Caption this image in one factual sentence, for retrieval indexing."},
            ]}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(text, resp.usage.input_tokens, resp.usage.output_tokens,
                            _cost(model, resp.usage.input_tokens, resp.usage.output_tokens))


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai

    @staticmethod
    def _to_gemini_history(history: list[dict] | None) -> list[dict]:
        """Gemini's chat history uses role 'model' where everyone else says
        'assistant', and each turn's content must be a list of parts."""
        if not history:
            return []
        return [{"role": "model" if h["role"] == "assistant" else "user", "parts": [h["content"]]}
                for h in history]

    def chat(self, system: str, user: str, model: str = "gemini-2.5-flash", max_tokens: int = 1024,
              history: list[dict] | None = None, temperature: float | None = None,
              top_p: float | None = None, top_k: int | None = None,
              images: list[Path] | None = None) -> LLMResponse:
        m = self._genai.GenerativeModel(model, system_instruction=system)
        gen_config = {"max_output_tokens": max_tokens}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if top_p is not None:
            gen_config["top_p"] = top_p
        if top_k is not None:
            gen_config["top_k"] = top_k

        user_parts = [user]
        if images:
            from PIL import Image
            user_parts = [Image.open(p) for p in images] + [user]

        chat = m.start_chat(history=self._to_gemini_history(history))
        resp = chat.send_message(user_parts, generation_config=gen_config)
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        return LLMResponse(resp.text, in_tok, out_tok, _cost(model, in_tok, out_tok))

    def caption_image(self, image_path, model: str = "gemini-2.5-flash") -> LLMResponse:
        from PIL import Image
        m = self._genai.GenerativeModel(model)
        resp = m.generate_content([
            "Caption this image in one factual sentence, for retrieval indexing.",
            Image.open(image_path),
        ])
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        return LLMResponse(resp.text, in_tok, out_tok, _cost(model, in_tok, out_tok))


class OpenAIClient(BaseLLMClient):
    """Used for text-embedding-3-*, and available as a generation option too."""
    def __init__(self, api_key: str):
        import openai
        self._client = openai.OpenAI(api_key=api_key)

    def chat(self, system: str, user: str, model: str = "gpt-4o-mini", max_tokens: int = 1024,
              history: list[dict] | None = None, temperature: float | None = None,
              top_p: float | None = None, top_k: int | None = None,
              images: list[Path] | None = None) -> LLMResponse:
        # OpenAI's chat-completions API has no top_k knob -- silently ignored,
        # the UI grays this field out for openai/groq so it's not a broken promise.
        user_content = user
        if images:
            user_content = [{"type": "text", "text": user}] + [_image_block_openai(p) for p in images]
        messages = [{"role": "system", "content": system}, *(history or []),
                    {"role": "user", "content": user_content}]
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        resp = self._client.chat.completions.create(model=model, max_tokens=max_tokens, messages=messages, **kwargs)
        u = resp.usage
        return LLMResponse(resp.choices[0].message.content, u.prompt_tokens, u.completion_tokens,
                            _cost(model, u.prompt_tokens, u.completion_tokens))

    def caption_image(self, image_path, model: str = "gpt-4o-mini") -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=model, max_tokens=200,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Caption this image in one factual sentence, for retrieval indexing."},
                _image_block_openai(image_path),
            ]}],
        )
        u = resp.usage
        return LLMResponse(resp.choices[0].message.content, u.prompt_tokens, u.completion_tokens,
                            _cost(model, u.prompt_tokens, u.completion_tokens))


class GroqClient(BaseLLMClient):
    """Groq's API is OpenAI-compatible, so this reuses the `openai` SDK
    pointed at Groq's endpoint instead of pulling in a separate `groq`
    package -- one fewer dependency for the same wire format."""
    def __init__(self, api_key: str):
        import openai
        self._client = openai.OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def chat(self, system: str, user: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 1024,
              history: list[dict] | None = None, temperature: float | None = None,
              top_p: float | None = None, top_k: int | None = None,
              images: list[Path] | None = None) -> LLMResponse:
        # Same as OpenAI: no top_k on this API.
        user_content = user
        if images:
            user_content = [{"type": "text", "text": user}] + [_image_block_openai(p) for p in images]
        messages = [{"role": "system", "content": system}, *(history or []),
                    {"role": "user", "content": user_content}]
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        resp = self._client.chat.completions.create(model=model, max_tokens=max_tokens, messages=messages, **kwargs)
        u = resp.usage
        return LLMResponse(resp.choices[0].message.content, u.prompt_tokens, u.completion_tokens,
                            _cost(model, u.prompt_tokens, u.completion_tokens))

    def caption_image(self, image_path, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=model, max_tokens=200,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Caption this image in one factual sentence, for retrieval indexing."},
                _image_block_openai(image_path),
            ]}],
        )
        u = resp.usage
        return LLMResponse(resp.choices[0].message.content, u.prompt_tokens, u.completion_tokens,
                            _cost(model, u.prompt_tokens, u.completion_tokens))


_PROVIDERS = {"claude": ClaudeClient, "gemini": GeminiClient, "openai": OpenAIClient, "groq": GroqClient}


def get_client(provider: str, api_key: str) -> BaseLLMClient:
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](api_key)


# ---------------------------------------------------------------------------
# Higher-level helpers used directly by the UI: rewrite prompt, suggest option
# ---------------------------------------------------------------------------

REWRITE_GUIDELINES = {
    "claude": "Best for precise, structure-preserving rewrites -- use when the prompt already has a shape you want kept.",
    "gemini": "Fastest and cheapest -- use for quick iteration when you'll throw away most drafts.",
    "openai": "Good middle ground -- use when you want a second opinion different from Claude/Gemini's house style.",
    "groq": "Fastest and cheapest of all -- Groq's inference speed makes it best for rapid iteration or high-volume runs.",
}


def rewrite_system_prompt(current_prompt: str, provider: str, api_key: str, model: str | None = None) -> LLMResponse:
    client = get_client(provider, api_key)
    system = (
        "You improve system prompts for a RAG assistant. Keep the same intent and constraints, "
        "but make it more precise, add explicit citation instructions if missing, and remove ambiguity. "
        "Return ONLY the rewritten prompt, no preamble."
    )
    kwargs = {"model": model} if model else {}
    return client.chat(system, current_prompt, **kwargs)


def score_image_relevance(provider: str, api_key: str, query: str, image_path, model: str | None = None) -> float:
    """The optional 'vision-LLM relevance scoring' reranking toggle: a cheap
    vision call judging one (query, image) pair on what the image actually
    shows, not just its caption. Reply parsing is defensive -- providers
    won't always obey 'number only' -- and falls back to a neutral 0.5
    rather than crashing the rerank pass over one bad response."""
    client = get_client(provider, api_key)
    system = ("Rate how relevant this image is to answering the question, on a 0.0-1.0 scale "
              "(1.0 = directly answers it, 0.0 = completely unrelated). Reply with ONLY the number.")
    kwargs = {"model": model} if model else {}
    resp = client.chat(system, query, max_tokens=8, images=[image_path], **kwargs)
    match = re.search(r"\d*\.?\d+", resp.text)
    return max(0.0, min(1.0, float(match.group()))) if match else 0.5


def suggest_option(stage: str, options: list[str], context: str, provider: str, api_key: str,
                    model: str | None = None) -> LLMResponse:
    """The per-stage '✦ Suggest' advisor: recommends one of the available
    dropdown options given a short description of the actual data/query."""
    client = get_client(provider, api_key)
    system = (
        f"You are an RAG pipeline advisor for the '{stage}' stage. "
        f"Given the context, recommend ONE option from {options} and justify it in 2 sentences max."
    )
    kwargs = {"model": model} if model else {}
    return client.chat(system, context, **kwargs)
