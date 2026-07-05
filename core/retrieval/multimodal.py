from __future__ import annotations
from core.retrieval.base import RetrievedItem
from core.vectorstore.chroma_store import ChromaStore


def query_images(store: ChromaStore, query_embedding: list[float], top_k: int = 4) -> list[RetrievedItem]:
    """Runs the image-collection query for one search and returns plain
    RetrievedItems -- same shape as text results, so downstream code (UI,
    reranking, eval) never needs to special-case images."""
    res = store.query_images(query_embedding, top_k=top_k)
    if not res["ids"][0]:
        return []
    return [RetrievedItem(id=id_, text=doc, score=1 - dist, kind="image", meta=meta)
            for id_, doc, dist, meta in zip(
                res["ids"][0], res["documents"][0], res["distances"][0], res["metadatas"][0])]


def merge_results(text_items: list[RetrievedItem], image_items: list[RetrievedItem],
                   mode: str = "text-only") -> list[RetrievedItem]:
    """The 'Result mode' control. A thin function, not a class -- it has no
    state of its own, just a rule for combining two already-computed lists.

    - "text-only":      ignore images entirely.
    - "joint":           rank everything together by score. Only valid when
                         image vectors share the text embedder's space
                         (caption-text-embed) -- scores are directly
                         comparable. Using this with CLIP image vectors
                         would silently compare two different scales.
    - "separate-merge":  keep each ranking's own internal order, just
                         concatenate text results then image results --
                         always safe, regardless of embedding space.
    """
    if mode == "text-only" or not image_items:
        return text_items
    if mode == "joint":
        return sorted(text_items + image_items, key=lambda i: -i.score)
    if mode == "separate-merge":
        return text_items + image_items
    raise ValueError(f"Unknown result mode '{mode}'. Choose: text-only, joint, separate-merge")
