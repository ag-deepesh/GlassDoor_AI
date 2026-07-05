from __future__ import annotations
from pathlib import Path
from core.schemas import Chunk, ImageAsset


class ChromaStore:
    """Local, persistent, in-process vector store -- no server to run, which
    is what makes this comfortable on a MacBook Air for <=20 files. Text and
    image vectors live in separate collections (their embedding spaces
    aren't always the same -- e.g. CLIP vs. MiniLM) but share `doc_id` in
    metadata so a retrieval result can always be traced back to its source
    document."""

    def __init__(self, persist_dir: Path | str = "out/chroma"):
        import chromadb
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self.text = self._client.get_or_create_collection("text_chunks")
        self.images = self._client.get_or_create_collection("image_assets")

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.text.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "page": c.page or 0, **{k: str(v) for k, v in c.meta.items()}}
                       for c in chunks],
        )

    def add_images(self, images: list[ImageAsset], embeddings: list[list[float]]) -> None:
        if not images:
            return
        self.images.upsert(
            ids=[i.id for i in images],
            embeddings=embeddings,
            documents=[i.caption or "" for i in images],
            metadatas=[{"doc_id": i.id.rsplit("_img", 1)[0], "page": i.page or 0, "path": str(i.path)}
                       for i in images],
        )

    def query_text(self, query_embedding: list[float], top_k: int = 10) -> dict:
        return self.text.query(query_embeddings=[query_embedding], n_results=top_k)

    def query_images(self, query_embedding: list[float], top_k: int = 5) -> dict:
        return self.images.query(query_embeddings=[query_embedding], n_results=top_k)

    def all_text_documents(self) -> tuple[list[str], list[str]]:
        """(ids, texts) for every chunk -- used to build the BM25 index,
        which needs the full corpus rather than a vector query."""
        got = self.text.get()
        return got["ids"], got["documents"]
