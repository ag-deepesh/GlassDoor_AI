from __future__ import annotations
from core.registry import register
from core.embedding.base import BaseEmbedder


@register("embedding", "gemini-text-embedding")
class GeminiEmbedder(BaseEmbedder):
    """Google's text-embedding-004 via the Gemini API. Costs per call (unlike
    the local options) -- use when demonstrating the API-embedding
    quality/cost trade-off, not as the default for a small local corpus."""

    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:  # Gemini's embed_content takes one text at a time
            resp = self._genai.embed_content(model="models/text-embedding-004", content=t)
            out.append(resp["embedding"])
        return out

    @property
    def dim(self) -> int:
        return 768  # text-embedding-004's fixed output dimension
