"""
The orchestrator, split into the platform's two phases:

  - Knowledge Base phase: build_kb() runs parsing -> chunking -> embedding
    once per corpus and persists the result (Chroma dir + metadata.json
    recording which embedding model built it).
  - Query phase: answer_query() runs retrieval -> reranking -> optional
    ReAct refinement -> generation, once per question, against a KB loaded
    with Pipeline.load() (which locks embedding_method to whatever the KB
    was actually built with -- never a dropdown at query time).

Both phases are generators yielding a StageReport per completed stage, or a
StageError if a stage fails -- a failure never erases the reports already
yielded for earlier stages, mirroring the CLI's original "print every
completed stage before a clear failure message" behavior, just formalized
as typed objects an API/UI can stream and render instead of terminal text.

Both phases also support the same interactive/run-all distinction as
before: pass on_stage_reviewed to get called with every yielded item, and
set interactive=True to have its return value gate whether the next stage
runs (the "require acknowledgment before advancing" behavior).
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Callable, Iterator

from core.schemas import StageReport, StageError, TraceEvent
from core.pipeline_config import PipelineConfig
from core.registry import get

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "what", "how", "why", "does", "do", "did", "which", "that",
    "this", "with", "by", "as", "at", "be", "it", "its", "from", "can", "will",
}


def _matched_terms(query: str, text: str, limit: int = 5) -> list[str]:
    """Cheap, zero-LLM-cost rationale for why a chunk scored the way it did --
    the query terms it actually contains. No semantic claim, just surfacing
    the lexical overlap a human can sanity-check against the score."""
    query_terms = [w.strip(".,?!:;\"'()").lower() for w in query.split()]
    query_terms = [w for w in query_terms if len(w) > 2 and w not in _STOPWORDS]
    text_lower = text.lower()
    seen, matched = set(), []
    for term in query_terms:
        if term not in seen and term in text_lower:
            seen.add(term)
            matched.append(term)
        if len(matched) == limit:
            break
    return matched
from core import kb_store
from core.parsing.assorted import EXT_TO_METHOD
from core.chunking.base import ChunkConfig
from core.vectorstore.chroma_store import ChromaStore
from core.retrieval.base import RetrievedItem
from core.retrieval.multimodal import query_images, merge_results
from core.evaluation import heuristics

# Import every registrable module for its registration side-effect -- this
# is the one place that needs to know all of them exist; everything else
# just calls registry.get(stage, method).
from core.parsing import assorted as _assorted  # noqa: F401 (pdf/docx/pptx/md/txt/assorted)
from core.chunking import fixed as _f, recursive as _r, sentence as _s, markdown_structure as _m, semantic as _sem  # noqa: F401,E501
from core.embedding import text_local as _tl, text_gemini as _tg, text_openai as _to  # noqa: F401
from core.reranking import reranking as _rr  # noqa: F401
from core.generation import generation as _gen  # noqa: F401
from core.retrieval import semantic as _rs, keyword as _rk, hybrid_rrf as _rh  # noqa: F401
from core.judge import gemini as _jg, claude as _jc, openai as _jo, groq as _jgr  # noqa: F401


def _hint_for(stage: str, error: Exception) -> str:
    """One-line, actionable hint for a StageError -- deliberately simple,
    plain-language pattern matching on common failure shapes rather than an
    exhaustive taxonomy; the raw exception is still in what_failed."""
    if isinstance(error, KeyError):
        return f"Missing API key for provider {error} -- add it in the API keys panel and try again."
    if isinstance(error, FileNotFoundError):
        return "A required file or persisted KB path wasn't found -- check the KB name and that build_kb() completed."
    if "Connection" in type(error).__name__ or "Timeout" in type(error).__name__:
        return "A network call failed -- check your internet connection and that the relevant API key is valid."
    msg = str(error).lower()
    if "api_key" in msg or "api key" in msg or "auth" in msg or "credentials" in msg or "unauthorized" in msg:
        return "Missing or invalid API key for this stage's provider -- check the API keys panel and try again."
    return f"{stage.capitalize()} stopped -- check the '{stage}' configuration and inputs, then retry."


class Pipeline:
    def __init__(self, config: PipelineConfig, workdir: Path):
        self.config = config
        self.workdir = Path(workdir)
        self.store = ChromaStore(self.workdir / "chroma")

        # Filled in progressively as stages run.
        self.docs = []
        self.chunks = []
        self.text_embedder = None
        self.retrieved: list[RetrievedItem] = []
        self.reranked: list[RetrievedItem] = []
        self.answer = None

    @classmethod
    def load(cls, kb_name: str, api_keys: dict, **config_overrides) -> "Pipeline":
        """Reconstruct a Pipeline for querying an already-built KB.
        embedding_method is locked from the KB's own metadata.json -- never
        taken from a caller-supplied config -- so retrieval always embeds
        the query with the exact model that embedded the corpus. Only the
        query text needs embedding at this point; the corpus's vectors are
        already persisted in the KB's Chroma dir."""
        meta = kb_store.load_metadata(kb_name)
        cfg = PipelineConfig(embedding_method=meta["embedding_method"], api_keys=api_keys, **config_overrides)
        pipeline = cls(cfg, workdir=kb_store.kb_path(kb_name))
        pipeline.text_embedder = pipeline._make_embedder()
        return pipeline

    # -- helpers -------------------------------------------------------
    def _report(self, stage: str, method: str, t0: float, input_summary: str, output_summary: str,
                tokens: int = 0, cost: float = 0.0, ref_free: dict | None = None,
                ref_based: dict | None = None) -> StageReport:
        trace = TraceEvent(stage=stage, method=method, input_summary=input_summary,
                            output_summary=output_summary, latency_ms=round((time.time() - t0) * 1000, 1),
                            tokens=tokens, cost_usd=cost)
        return StageReport(stage=stage, method=method, output_preview=output_summary, trace=trace,
                            eval_reference_free=ref_free, eval_with_reference=ref_based)

    @staticmethod
    def _items_for_trace(items: list[RetrievedItem], query: str | None = None,
                          prev_items: list[RetrievedItem] | None = None) -> list[dict]:
        """Rank + score-sorted item dicts for the UI's per-stage 'data'
        panel -- same shape for retrieval and reranking so the UI can
        render both with one component. When query is given, each item also
        gets a rule-based rationale (matched query terms, and -- when
        prev_items is given, i.e. this is post-rerank -- its rank shift
        versus the pre-rerank order), all zero-cost, no extra LLM call."""
        ranked = sorted(items, key=lambda i: i.score, reverse=True)
        prev_rank = None
        if prev_items is not None:
            prev_ranked = sorted(prev_items, key=lambda i: i.score, reverse=True)
            prev_rank = {it.id: i + 1 for i, it in enumerate(prev_ranked)}
        out = []
        for i, it in enumerate(ranked):
            d = {"rank": i + 1, "id": it.id, "kind": it.kind, "score": round(it.score, 4), "text": it.text}
            if query is not None:
                d["matched_terms"] = _matched_terms(query, it.text)
            if prev_rank is not None:
                old_rank = prev_rank.get(it.id)
                d["rank_change"] = None if old_rank is None else old_rank - (i + 1)
            out.append(d)
        return out

    def _make_embedder(self):
        cfg = self.config
        embedder_cls = get("embedding", cfg.embedding_method)
        needs_key = cfg.embedding_method.startswith(("gemini", "openai"))
        if not needs_key:
            return embedder_cls()
        provider = "gemini" if cfg.embedding_method.startswith("gemini") else "openai"
        api_key = cfg.api_keys.get(provider)
        if not api_key:
            raise ValueError(f"Missing {provider} API key -- add it in the API keys panel and try again.")
        return embedder_cls(api_key)

    def _save_kb_metadata(self, input_dir: Path) -> None:
        n_images = sum(len(d.images) for d in self.docs)
        paths = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in EXT_TO_METHOD]
        metadata = {
            "embedding_method": self.config.embedding_method,
            "embedding_dim": self.text_embedder.dim if self.text_embedder else 0,
            "parsing_method": self.config.parsing_method,
            "chunking_method": self.config.chunking_method,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "image_embedding_method": self.config.image_embedding_method if n_images else None,
            "source_files": [{"filename": p.name, "size": p.stat().st_size, "sha1": kb_store.file_sha1(p)}
                              for p in sorted(paths)],
            "n_docs": len(self.docs), "n_chunks": len(self.chunks), "n_images": n_images,
        }
        kb_store.save_metadata(self.workdir.name, metadata)

    # -- stages ----------------------------------------------------------
    def run_parsing(self, input_dir: Path) -> StageReport:
        t0 = time.time()
        cfg = self.config
        paths = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in EXT_TO_METHOD]
        parser_cls = get("parsing", cfg.parsing_method if cfg.parsing_method != "assorted" else "assorted")
        parser = parser_cls(extract_images=cfg.extract_images, ocr=cfg.ocr, assets_dir=self.workdir / "assets")
        self.docs = [parser.parse(p, doc_id=p.stem) for p in paths]

        report = heuristics.parsing_report(self.docs)
        total_words = sum(len(d.full_text.split()) for d in self.docs)
        out = self._report("parsing", cfg.parsing_method, t0, f"{len(paths)} files",
                            f"{len(self.docs)} docs, {total_words} words", ref_free=report)
        out.trace.extra["items"] = [
            {"doc_id": d.doc_id, "format": d.format, "n_pages": d.n_pages,
             "n_words": len(d.full_text.split()), "text": d.full_text}
            for d in self.docs
        ]
        return out

    def run_chunking(self) -> StageReport:
        t0 = time.time()
        cfg = self.config
        chunker_cls = get("chunking", cfg.chunking_method)
        chunker = chunker_cls(ChunkConfig(chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap))
        self.chunks = [c for doc in self.docs for c in chunker.chunk(doc)]

        report = heuristics.chunking_report(self.chunks, cfg.chunk_size)
        out = self._report("chunking", cfg.chunking_method, t0, f"{len(self.docs)} docs",
                            f"{len(self.chunks)} chunks", ref_free=report)
        out.trace.extra["items"] = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "page": c.page, "n_tokens": c.n_tokens, "text": c.text}
            for c in self.chunks
        ]
        return out

    def run_embedding(self) -> StageReport:
        t0 = time.time()
        cfg = self.config
        self.text_embedder = self._make_embedder()

        texts = [c.text for c in self.chunks]
        vectors = self.text_embedder.embed(texts) if texts else []
        self.store.add_chunks(self.chunks, vectors)

        n_images = sum(len(d.images) for d in self.docs)
        if n_images and cfg.image_embedding_method == "caption-text-embed":
            from core.embedding.image_caption import CaptionThenEmbed
            caption_key = cfg.api_keys.get(cfg.caption_provider)
            if not caption_key:
                raise ValueError(f"Missing {cfg.caption_provider} API key -- add it in the API keys panel and try again.")
            img_embedder = CaptionThenEmbed(self.text_embedder, cfg.caption_provider, caption_key)
            all_images = [img for d in self.docs for img in d.images]
            img_vectors = img_embedder.embed_images(all_images)
            self.store.add_images(all_images, img_vectors)

        report = heuristics.embedding_report(len(vectors), self.text_embedder.dim if vectors else 0, n_images)
        return self._report("embedding", cfg.embedding_method, t0, f"{len(texts)} chunks",
                             f"{len(vectors)} vectors", ref_free=report)

    def run_retrieval(self, query: str) -> StageReport:
        t0 = time.time()
        cfg = self.config
        from core.retrieval.semantic import SemanticRetriever
        from core.retrieval.keyword import KeywordRetriever
        from core.retrieval.hybrid_rrf import HybridRRFRetriever, rrf_fuse

        if cfg.retrieval_method == "semantic":
            retriever = SemanticRetriever(self.store, self.text_embedder, top_k=cfg.top_k)
        elif cfg.retrieval_method == "keyword":
            retriever = KeywordRetriever(self.store, top_k=cfg.top_k)
        elif cfg.retrieval_method == "hybrid-rrf":
            retriever = HybridRRFRetriever(SemanticRetriever(self.store, self.text_embedder, top_k=cfg.top_k),
                                            KeywordRetriever(self.store, top_k=cfg.top_k), top_k=cfg.top_k)
        else:
            raise ValueError(f"Unknown retrieval method '{cfg.retrieval_method}'")

        # KB results first; blended with live web (RRF) only if the toggle is on.
        kb_items = retriever.retrieve(query)
        web_warning, web_cost = None, 0.0
        if cfg.web_enabled:
            from core.web_search import TavilyClient, TavilySearchError, COST_PER_SEARCH_USD
            try:
                web_items = TavilyClient(cfg.api_keys.get("tavily", "")).search(query, max_results=cfg.top_k)
                text_items = rrf_fuse([kb_items, web_items], top_k=cfg.top_k)
                web_cost = COST_PER_SEARCH_USD
            except TavilySearchError as e:
                web_warning = f"Tavily web search skipped ({e}) -- KB-only results used."
                text_items = kb_items
        else:
            text_items = kb_items

        image_items = []
        if cfg.result_mode != "text-only":
            q_emb = self.text_embedder.embed([query])[0]
            image_items = query_images(self.store, q_emb, top_k=4)
        self.retrieved = merge_results(text_items, image_items, mode=cfg.result_mode)

        report = heuristics.retrieval_report(self.retrieved)
        if web_warning:
            report["recommendation"] = f"{web_warning} · {report['recommendation']}"
        out = self._report("retrieval", cfg.retrieval_method, t0, f"query: {query[:60]}",
                            f"{len(self.retrieved)} items ({len(image_items)} images)",
                            cost=web_cost, ref_free=report)
        if web_warning:
            out.trace.extra["web_warning"] = web_warning
        out.trace.extra["items"] = self._items_for_trace(self.retrieved, query=query)
        return out

    def run_reranking(self, query: str) -> StageReport:
        t0 = time.time()
        cfg = self.config
        reranker_cls = get("reranking", cfg.reranking_method)
        reranker = reranker_cls()
        pre_rerank = self.retrieved
        self.reranked = reranker.rerank(query, self.retrieved, keep_top=cfg.rerank_keep_top)

        report = heuristics.retrieval_report(self.reranked)  # same shape of report, post-rerank
        out = self._report("reranking", cfg.reranking_method, t0, f"{len(self.retrieved)} candidates",
                            f"{len(self.reranked)} kept", ref_free=report)
        out.trace.extra["items"] = self._items_for_trace(self.reranked, query=query, prev_items=pre_rerank)
        return out

    def run_generation(self, query: str, reference: str | None = None) -> StageReport:
        t0 = time.time()
        cfg = self.config
        gen_cls = get("generation", cfg.generation_method)
        provider = gen_cls._provider
        generator = gen_cls(cfg.api_keys[provider])
        resp = generator.generate(cfg.system_prompt, query, self.reranked)
        self.answer = resp.text

        ref_free, ref_based = None, None
        judge_provider = cfg.judge_provider or provider  # auto-match generation's provider by default
        judge_key = cfg.api_keys.get(judge_provider)
        if not judge_key:
            ref_free = {"scores": {}, "recommendation":
                        f"RAGAS scoring skipped: no API key configured for judge provider '{judge_provider}'."}
        else:
            try:
                from core.evaluation.ragas_eval import get_judge, reference_free_report, reference_based_report
                judge = get_judge(judge_provider, judge_key)
                contexts = [i.text for i in self.reranked]
                ref_free = reference_free_report(query, contexts, self.answer, judge).as_dict()
                if reference:
                    ref_based = reference_based_report(query, contexts, self.answer, reference, judge).as_dict()
            except Exception as e:  # RAGAS/judge is best-effort -- generation output still stands without it
                ref_free = {"scores": {}, "recommendation": f"RAGAS scoring unavailable this run: {e}"}

        return self._report("generation", cfg.generation_method, t0, f"query: {query[:60]}",
                             self.answer[:200], tokens=resp.input_tokens + resp.output_tokens,
                             cost=resp.cost_usd, ref_free=ref_free, ref_based=ref_based)

    # -- orchestration ----------------------------------------------------
    def _run_stages(self, stage_fns: list[tuple[str, str, Callable[[], StageReport]]], interactive: bool,
                     on_stage_reviewed: Callable[[StageReport | StageError], bool] | None
                     ) -> Iterator[StageReport | StageError]:
        """Runs stage_fns in order, yielding a StageReport or StageError
        after each one. If on_stage_reviewed is given, it's called with
        every yielded item (so it can render both successes and failures);
        in interactive mode, its return value on a StageReport decides
        whether to continue -- the 'require acknowledgment before
        advancing' gate. A StageError always halts the run, but every
        earlier StageReport is still valid and was already yielded."""
        for stage, method, fn in stage_fns:
            try:
                report = fn()
            except Exception as e:
                err = StageError(stage=stage, method=method, what_failed=f"{type(e).__name__}: {e}",
                                  hint=_hint_for(stage, e))
                if on_stage_reviewed is not None:
                    on_stage_reviewed(err)
                yield err
                return
            yield report
            if on_stage_reviewed is not None:
                result = on_stage_reviewed(report)
                if interactive:
                    report.acknowledged = bool(result)
                    if not report.acknowledged:
                        return

    def build_kb(self, input_dir: Path, interactive: bool = False,
                 on_stage_reviewed: Callable[[StageReport | StageError], bool] | None = None
                 ) -> Iterator[StageReport | StageError]:
        cfg = self.config
        stage_fns = [
            ("parsing", cfg.parsing_method, lambda: self.run_parsing(input_dir)),
            ("chunking", cfg.chunking_method, lambda: self.run_chunking()),
            ("embedding", cfg.embedding_method, lambda: self.run_embedding()),
        ]
        had_error = False
        for item in self._run_stages(stage_fns, interactive, on_stage_reviewed):
            had_error = had_error or isinstance(item, StageError)
            yield item
        if not had_error:
            self._save_kb_metadata(input_dir)

    def answer_query(self, query: str, reference: str | None = None, interactive: bool = False,
                      on_stage_reviewed: Callable[[StageReport | StageError], bool] | None = None
                      ) -> Iterator[StageReport | StageError]:
        cfg = self.config
        stage_fns = [
            ("retrieval", cfg.retrieval_method, lambda: self.run_retrieval(query)),
            ("reranking", cfg.reranking_method, lambda: self.run_reranking(query)),
        ]
        if cfg.react_enabled:
            from core.react_loop import run_react_loop
            stage_fns.append(("react", cfg.react_judge_method, lambda: run_react_loop(self, query)))
        stage_fns.append(("generation", cfg.generation_method, lambda: self.run_generation(query, reference)))
        yield from self._run_stages(stage_fns, interactive, on_stage_reviewed)

    def run_all(self, input_dir: Path, query: str, reference: str | None = None,
                interactive: bool = False,
                on_stage_reviewed: Callable[[StageReport | StageError], bool] | None = None
                ) -> list[StageReport | StageError]:
        """Back-compat convenience combining build_kb() + answer_query()
        into the original one-shot call -- new code should call the two
        phases directly (build once per corpus, query many times against
        the saved KB)."""
        reports = list(self.build_kb(input_dir, interactive=interactive, on_stage_reviewed=on_stage_reviewed))
        if reports and isinstance(reports[-1], StageError):
            return reports
        reports += list(self.answer_query(query, reference=reference, interactive=interactive,
                                           on_stage_reviewed=on_stage_reviewed))
        return reports
