from __future__ import annotations
from core.registry import register
from core.embedding.base import BaseEmbedder


@register("embedding", "gemini-text-embedding")
class GeminiEmbedder(BaseEmbedder):
    """Google's gemini-embedding-001 via the Gemini API (text-embedding-004,
    the previous model, was retired -- embedContent now 404s for it). Costs
    per call (unlike the local options) -- use when demonstrating the
    API-embedding quality/cost trade-off, not as the default for a small
    local corpus."""

    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:  # Gemini's embed_content takes one text at a time
            resp = self._genai.embed_content(model="models/gemini-embedding-001", content=t)
            out.append(resp["embedding"])
        return out

    @property
    def dim(self) -> int:
        return 3072  # gemini-embedding-001's default output dimension
