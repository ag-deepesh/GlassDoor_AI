from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievedItem:
    id: str
    text: str
    score: float
    kind: str = "text"  # "text" or "image"
    meta: dict = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}


class BaseRetriever(ABC):
    """top_k is manually configurable everywhere, default 10, per the
    platform's requirements -- not hardcoded inside any retriever."""

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievedItem]: ...
