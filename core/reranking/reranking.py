from __future__ import annotations
from abc import ABC, abstractmethod
from core.registry import register
from core.retrieval.base import RetrievedItem


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, items: list[RetrievedItem], keep_top: int) -> list[RetrievedItem]: ...


@register("reranking", "none")
class NoRerank(BaseReranker):
    """Passthrough -- use when retrieval is already precise, or to show the
    baseline in an A/B comparison against a real reranker."""
    def rerank(self, query: str, items: list[RetrievedItem], keep_top: int) -> list[RetrievedItem]:
        return items[:keep_top]


@register("reranking", "cross-encoder")
class CrossEncoderRerank(BaseReranker):
    """Scores (query, chunk) jointly with a small transformer instead of
    comparing two separately-computed vectors -- more accurate than the
    retrieval stage's similarity score, at the cost of one forward pass per
    candidate. Local model, no API cost -- best precision-per-rupee option.
    Only reranks text items; images pass through unscored and keep their
    retrieval-stage rank (a cross-encoder needs a text/text pair)."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, items: list[RetrievedItem], keep_top: int) -> list[RetrievedItem]:
        self._ensure_loaded()
        text_items = [i for i in items if i.kind == "text"]
        other_items = [i for i in items if i.kind != "text"]
        if not text_items:
            return items[:keep_top]

        pairs = [(query, i.text) for i in text_items]
        scores = self._model.predict(pairs)
        for item, score in zip(text_items, scores):
            item.score = float(score)
        text_items.sort(key=lambda i: -i.score)
        return (text_items + other_items)[:keep_top]
