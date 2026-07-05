from __future__ import annotations
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Every embedder maps a list of texts to a list of same-length vectors.
    Image embedding reuses this exactly: an image is captioned once (see
    core/llm_clients.py caption_image), and the caption text is embedded
    here -- so images and text always live in the same vector space,
    letting retrieval compare them directly with plain cosine similarity."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...
