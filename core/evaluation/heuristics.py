"""
Per-stage reference-free reports that DON'T call an LLM.

Design decision, stated plainly: full RAGAS metrics (Faithfulness,
AnswerRelevancy, ContextPrecision, ...) need a real answer to judge
against -- they don't cleanly apply to parsing/chunking/embedding/retrieval
in isolation, before generation has happened. Running an LLM judge at every
stage anyway would burn tokens for questionable signal. Instead, each early
stage gets a fast, deterministic, zero-cost heuristic report -- still a
concise report + recommendation, per the requirement, just not RAGAS. RAGAS
proper is reserved for the generation stage in core/evaluation/ragas_eval.py,
where it has an actual answer to evaluate.
"""
from __future__ import annotations
from core.schemas import ParsedDoc, Chunk
from core.retrieval.base import RetrievedItem


def _snippet(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def parsing_report(docs: list[ParsedDoc]) -> dict:
    total_words = sum(len(d.full_text.split()) for d in docs)
    total_images = sum(len(d.images) for d in docs)
    total_ocr = sum(sum(1 for b in d.text_blocks if b.source_ocr) for d in docs)
    empty_docs = [d.doc_id for d in docs if not d.full_text.strip()]

    notes = []
    if empty_docs:
        notes.append(f"{len(empty_docs)} doc(s) produced NO text ({', '.join(empty_docs)}) -- "
                      f"try enabling OCR if these are scanned pages")
    if total_images == 0:
        notes.append("no images were extracted -- confirm 'Extract images' is on if the corpus has diagrams/figures")
    if total_ocr > 0:
        notes.append(f"{total_ocr} text block(s) came from OCR, not a native text layer -- spot-check quality")
    recommendation = " · ".join(notes) if notes else "Parsing looks healthy -- text and images extracted as expected."

    return {"scores": {"n_docs": len(docs), "total_words": total_words,
                        "total_images": total_images, "n_ocr_blocks": total_ocr},
            "recommendation": recommendation}


def chunking_report(chunks: list[Chunk], target_size: int) -> dict:
    if not chunks:
        return {"scores": {}, "recommendation": "No chunks produced -- check that parsing returned text."}
    sizes = [c.n_tokens for c in chunks]
    mean = sum(sizes) / len(sizes)
    variance = sum((s - mean) ** 2 for s in sizes) / len(sizes)
    std = variance ** 0.5
    oversized = sum(1 for s in sizes if s > target_size * 1.3)

    notes = []
    if std > target_size * 0.5:
        notes.append("high size variance -- consider Fixed or Markdown-structure chunking for more uniform chunks")
    if oversized > len(chunks) * 0.1:
        notes.append(f"{oversized} chunk(s) are 30%+ over target size -- Recursive's fallback split may need a smaller chunk_size")
    if mean < target_size * 0.3:
        notes.append("chunks are much smaller than target -- check for excessive paragraph breaks fragmenting the text")
    recommendation = " · ".join(notes) if notes else "Chunk sizes are consistent and close to target -- no changes needed."

    largest = max(chunks, key=lambda c: c.n_tokens)
    smallest = min(chunks, key=lambda c: c.n_tokens)
    return {"scores": {"n_chunks": len(chunks), "mean_tokens": round(mean, 1), "std_tokens": round(std, 1)},
            "recommendation": recommendation,
            "examples": {
                "largest": {"id": largest.chunk_id, "n_tokens": largest.n_tokens, "text": _snippet(largest.text)},
                "smallest": {"id": smallest.chunk_id, "n_tokens": smallest.n_tokens, "text": _snippet(smallest.text)},
            }}


def embedding_report(n_vectors: int, dim: int, n_images: int = 0) -> dict:
    notes = []
    if n_vectors == 0:
        notes.append("no vectors were produced -- check that chunking ran before embedding")
    recommendation = " · ".join(notes) if notes else f"{n_vectors} vectors indexed at dim={dim} -- ready for retrieval."
    return {"scores": {"n_text_vectors": n_vectors, "dim": dim, "n_image_vectors": n_images},
            "recommendation": recommendation}


def retrieval_report(items: list[RetrievedItem]) -> dict:
    if not items:
        return {"scores": {}, "recommendation": "No results retrieved -- check the query and that the index isn't empty."}
    scores = [i.score for i in items]
    top, floor = max(scores), min(scores)
    n_images = sum(1 for i in items if i.kind == "image")

    notes = []
    if top - floor > 0.4:
        notes.append("wide score spread -- the lowest-ranked results may be irrelevant; consider a smaller top_k or reranking")
    if top < 0.5:
        notes.append("even the top result scores low -- the answer may not be in the corpus, or try Hybrid-RRF")
    recommendation = " · ".join(notes) if notes else "Retrieved scores look healthy and tightly clustered."

    ranked = sorted(items, key=lambda i: i.score, reverse=True)
    top_item, bottom_item = ranked[0], ranked[-1]
    return {"scores": {"n_results": len(items), "top_score": round(top, 3), "score_floor": round(floor, 3),
                        "n_images": n_images},
            "recommendation": recommendation,
            "examples": {
                "top": {"id": top_item.id, "kind": top_item.kind, "score": round(top_item.score, 3),
                        "text": _snippet(top_item.text)},
                "bottom": {"id": bottom_item.id, "kind": bottom_item.kind, "score": round(bottom_item.score, 3),
                           "text": _snippet(bottom_item.text)},
            }}
