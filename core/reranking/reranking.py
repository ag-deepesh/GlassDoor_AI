from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Callable
from core.registry import register
from core.retrieval.base import RetrievedItem

# (query, image_path) -> 0.0-1.0 relevance score, from a vision LLM. Optional --
# only supplied when the "vision-LLM relevance scoring" toggle is on.
VisionScorer = Callable[[str, str], float]


def _sigmoid(x: float) -> float:
    """Squashes a cross-encoder's raw logit into a 0..1 relevance probability --
    the standard interpretation for ms-marco-style cross-encoders -- so it's on
    the same scale as a vision-LLM score or a cosine-similarity retrieval score
    and the three can be safely sorted together."""
    return 1 / (1 + math.exp(-x))


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, items: list[RetrievedItem], keep_top: int,
               vision_scorer: VisionScorer | None = None) -> list[RetrievedItem]: ...


@register("reranking", "none")
class NoRerank(BaseReranker):
    """Passthrough -- use when retrieval is already precise, or to show the
    baseline in an A/B comparison against a real reranker."""
    def rerank(self, query: str, items: list[RetrievedItem], keep_top: int,
               vision_scorer: VisionScorer | None = None) -> list[RetrievedItem]:
        return items[:keep_top]


@register("reranking", "cross-encoder")
class CrossEncoderRerank(BaseReranker):
    """Scores (query, chunk) jointly with a small transformer instead of
    comparing two separately-computed vectors -- more accurate than the
    retrieval stage's similarity score, at the cost of one forward pass per
    candidate.

    Images compete for the SAME keep_top budget as text, not an uncapped
    extra allowance: an image with a caption (the default caption-text-embed
    path) is scored by running that caption through the identical
    cross-encoder as a text surrogate, then squashed to 0..1 so it's
    comparable to text scores. An image with no caption (clip-local -- CLIP
    embeds pixels directly, never generates one) keeps its retrieval-stage
    cosine score unless vision_scorer is supplied, in which case that's the
    only way to judge it on what it actually shows rather than an arbitrary
    passthrough score.
    """

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, items: list[RetrievedItem], keep_top: int,
               vision_scorer: VisionScorer | None = None) -> list[RetrievedItem]:
        if not items:
            return items[:keep_top]
        self._ensure_loaded()

        captioned = [i for i in items if i.text.strip()]
        uncaptioned_ids = {id(i) for i in items if not i.text.strip()}
        if captioned:
            pairs = [(query, i.text) for i in captioned]
            raw_scores = self._model.predict(pairs)
            for item, raw in zip(captioned, raw_scores):
                item.score = _sigmoid(float(raw))

        if vision_scorer:
            for img in [i for i in items if i.kind == "image"]:
                path = img.meta.get("path")
                if not path:
                    continue
                vscore = vision_scorer(query, path)
                # Blend with the caption-based score so one bad vision call can't
                # solely promote/bury a candidate that already has signal; an
                # uncaptioned image has no prior score to blend with.
                img.score = vscore if id(img) in uncaptioned_ids else (img.score + vscore) / 2

        items.sort(key=lambda i: -i.score)
        return items[:keep_top]
