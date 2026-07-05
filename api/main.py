"""
FastAPI layer over core/pipeline.py -- the same Pipeline the CLI drives,
now streamed to the browser over SSE instead of printed to a terminal.
Every event on the wire is a StageReport or StageError, JSON-encoded via
dataclasses.asdict() -- the exact same shape core/schemas.py already
defines, so the UI's per-stage panel and the CLI's --step output can never
drift apart from what the API sends.

Threading note: each request's Pipeline (and its ChromaStore/SQLite
connection) is built AND fully driven on one dedicated background thread
via _bridge() below, never touched from the request's own thread. A plain
sync generator handed to StreamingResponse gets iterated one `next()` call
at a time on Starlette's executor thread pool, which can hop across
different worker threads between calls -- and a SQLite connection opened
on one thread then queried from another reliably surfaces as chromadb's
Rust bindings raising "attempt to write a readonly database", not a clean
thread-safety error. Confining the whole pipeline run to one thread avoids
that entirely.

Run locally: `uvicorn api.main:app --reload` (see SETUP.md).
"""
from __future__ import annotations
import asyncio
import dataclasses
import json
import queue
import threading
from typing import AsyncIterator, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core import kb_store, llm_clients
from core.kb_store import UploadedFile, UploadLimitError
from core.pipeline import Pipeline
from core.pipeline_config import PipelineConfig
from core.registry import all_stages
from core.schemas import StageReport, StageError

app = FastAPI(title="GlassBox API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_methods=["*"], allow_headers=["*"],
)

_SENTINEL = object()


async def _bridge(worker) -> AsyncIterator:
    """Runs worker(q) entirely on one dedicated background thread and
    relays whatever it puts on the queue back to the async event loop --
    see the module docstring for why this thread confinement matters."""
    q: queue.Queue = queue.Queue()
    threading.Thread(target=worker, args=(q,), daemon=True).start()
    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _SENTINEL:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def _to_sse(item) -> str:
    if isinstance(item, (StageReport, StageError)):
        kind = "error" if isinstance(item, StageError) else "report"
        payload = {"type": kind, **dataclasses.asdict(item)}
    else:
        payload = item  # already a plain dict -- e.g. the final done/summary marker
    return f"data: {json.dumps(payload)}\n\n"


async def _sse(worker) -> AsyncIterator[str]:
    async for item in _bridge(worker):
        yield _to_sse(item)


@app.get("/registry")
def get_registry():
    """Every registered method per stage -- what each dropdown should show."""
    return all_stages()


@app.get("/kbs")
def list_kbs():
    """Existing KBs + their metadata, for the 'use existing' picker."""
    return kb_store.list_kbs()


@app.delete("/kbs/{name}")
def delete_kb(name: str):
    kb_store.delete_kb(name)
    return {"ok": True}


@app.post("/kbs/build")
async def build_kb(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    parsing_method: str = Form("assorted"),
    extract_images: bool = Form(True),
    ocr: bool = Form(False),
    chunking_method: str = Form("recursive"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(64),
    embedding_method: str = Form("minilm-l6"),
    image_embedding_method: str = Form("caption-text-embed"),
    api_keys: str = Form("{}"),  # JSON-encoded {"claude": "...", "gemini": "...", "openai": "..."}
):
    try:
        keys = json.loads(api_keys)
    except json.JSONDecodeError:
        raise HTTPException(400, "api_keys must be a JSON object")
    if kb_store.kb_exists(name):
        raise HTTPException(409, f"A KB named '{name}' already exists -- pick a different name or delete it first.")

    file_bytes = [(f.filename, await f.read()) for f in files]
    try:
        kb_store.validate_upload([UploadedFile(filename=n, size=len(c)) for n, c in file_bytes])
    except UploadLimitError as e:
        raise HTTPException(413, str(e))

    kb_path = kb_store.create_kb(name)
    uploads_dir = kb_path / "uploads"
    for filename, content in file_bytes:
        (uploads_dir / filename).write_bytes(content)

    cfg = PipelineConfig(
        parsing_method=parsing_method, extract_images=extract_images, ocr=ocr,
        chunking_method=chunking_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        embedding_method=embedding_method, image_embedding_method=image_embedding_method,
        api_keys=keys,
    )

    def worker(q: queue.Queue):
        try:
            pipeline = Pipeline(cfg, workdir=kb_path)  # constructed here -- same thread that will run every stage
            for item in pipeline.build_kb(uploads_dir):
                q.put(item)
            q.put({"type": "done"})
        except Exception as e:
            q.put(e)
        finally:
            q.put(_SENTINEL)

    return StreamingResponse(_sse(worker), media_type="text/event-stream")


class QueryRequest(BaseModel):
    kb_name: str
    query: str
    reference: Optional[str] = None
    api_keys: dict = {}

    retrieval_method: str = "hybrid-rrf"
    top_k: int = 10
    result_mode: str = "text-only"
    web_enabled: bool = False

    reranking_method: str = "cross-encoder"
    rerank_keep_top: int = 4

    react_enabled: bool = False
    react_max_iterations: int = 3
    react_judge_method: str = "gemini-2.5-flash"

    generation_method: str = "claude-sonnet"
    system_prompt: Optional[str] = None
    judge_provider: str = "claude"  # RAGAS judge -- distinct from react_judge_method above


@app.post("/query")
def query(req: QueryRequest):
    if not kb_store.kb_exists(req.kb_name):
        raise HTTPException(404, f"No KB named '{req.kb_name}'.")

    overrides = dict(
        retrieval_method=req.retrieval_method, top_k=req.top_k, result_mode=req.result_mode,
        web_enabled=req.web_enabled, reranking_method=req.reranking_method, rerank_keep_top=req.rerank_keep_top,
        react_enabled=req.react_enabled, react_max_iterations=req.react_max_iterations,
        react_judge_method=req.react_judge_method, generation_method=req.generation_method,
        judge_provider=req.judge_provider,
    )
    if req.system_prompt:
        overrides["system_prompt"] = req.system_prompt

    def worker(q: queue.Queue):
        try:
            pipeline = Pipeline.load(req.kb_name, api_keys=req.api_keys, **overrides)
        except Exception as e:
            q.put(e)
            q.put(_SENTINEL)
            return

        total_cost, total_latency_ms, total_tokens = 0.0, 0.0, 0
        had_error = False
        try:
            for item in pipeline.answer_query(req.query, reference=req.reference):
                if isinstance(item, StageReport):
                    total_cost += item.trace.cost_usd
                    total_latency_ms += item.trace.latency_ms
                    total_tokens += item.trace.tokens
                had_error = had_error or isinstance(item, StageError)
                q.put(item)
            q.put({"type": "done", "answer": None if had_error else pipeline.answer,
                   "total_cost_usd": round(total_cost, 6), "total_latency_ms": round(total_latency_ms, 1),
                   "total_tokens": total_tokens})
        except Exception as e:
            q.put(e)
        finally:
            q.put(_SENTINEL)

    return StreamingResponse(_sse(worker), media_type="text/event-stream")


class RewriteRequest(BaseModel):
    prompt: str
    provider: str
    api_key: str
    model: Optional[str] = None


@app.post("/llm/rewrite-prompt")
def rewrite_prompt(req: RewriteRequest):
    resp = llm_clients.rewrite_system_prompt(req.prompt, req.provider, req.api_key, req.model)
    return {"text": resp.text, "tokens": resp.input_tokens + resp.output_tokens, "cost_usd": resp.cost_usd}


class SuggestRequest(BaseModel):
    stage: str
    options: list[str]
    context: str
    provider: str
    api_key: str
    model: Optional[str] = None


@app.post("/llm/suggest")
def suggest(req: SuggestRequest):
    resp = llm_clients.suggest_option(req.stage, req.options, req.context, req.provider, req.api_key, req.model)
    return {"text": resp.text, "tokens": resp.input_tokens + resp.output_tokens, "cost_usd": resp.cost_usd}
