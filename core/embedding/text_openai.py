from __future__ import annotations
from core.registry import register
from core.embedding.base import BaseEmbedder


class _OpenAIEmbedder(BaseEmbedder):
    _model_name: str = ""
    _dim: int = 0

    def __init__(self, api_key: str):
        import openai
        self._client = openai.OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model_name, input=texts)
        return [d.embedding for d in resp.data]

    @property
    def dim(self) -> int:
        return self._dim


@register("embedding", "openai-text-embedding-3-small")
class OpenAISmallEmbedder(_OpenAIEmbedder):
    _model_name = "text-embedding-3-small"  # 1536-dim, cheaper, strong general baseline
    _dim = 1536


@register("embedding", "openai-text-embedding-3-large")
class OpenAILargeEmbedder(_OpenAIEmbedder):
    _model_name = "text-embedding-3-large"  # 3072-dim, higher quality, ~6x the cost of small
    _dim = 3072
