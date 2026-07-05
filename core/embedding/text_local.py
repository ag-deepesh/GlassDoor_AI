from __future__ import annotations
from core.registry import register
from core.embedding.base import BaseEmbedder

_MODEL_NAMES = {
    "minilm-l6": "all-MiniLM-L6-v2",     # 384-dim, ~90MB, default -- fast, plenty accurate for <=20 files
    "bge-small": "BAAI/bge-small-en-v1.5",  # 384-dim, ~130MB, slightly stronger on retrieval benchmarks
}


class _LocalTextEmbedder(BaseEmbedder):
    _model_key: str = ""

    def __init__(self):
        self._model = None  # lazy: don't download/load until actually used

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_MODEL_NAMES[self._model_key])

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()


@register("embedding", "minilm-l6")
class MiniLMEmbedder(_LocalTextEmbedder):
    _model_key = "minilm-l6"


@register("embedding", "bge-small")
class BGESmallEmbedder(_LocalTextEmbedder):
    _model_key = "bge-small"


# BGE-M3 intentionally NOT registered yet (deferred per your "leave it for
# now" call) -- the full checkpoint is ~2.2GB and noticeably slower on a
# CPU-only Air. Adding it later is exactly this pattern: a 15-line class
# with @register("embedding", "bge-m3"), model name "BAAI/bge-m3".
