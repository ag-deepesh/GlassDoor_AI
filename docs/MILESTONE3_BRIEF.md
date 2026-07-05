# GlassBox — Milestone 3 build brief

Paste this whole file to Claude Code as the kickoff prompt (or keep it in the repo as `docs/MILESTONE3_BRIEF.md` and point Claude Code at it). It assumes `glassbox_milestone2.zip` and `rag-lab-mockup.jsx` are already unzipped/placed in the repo.

## Goal

Turn the Milestone 2 Python backend + the React mockup into a real, locally-runnable AI-training platform: a RAG lab (primary) with an optional ReAct-loop refinement stage, built to also host a separate ReAct-loop *app* later on the same shared infra.

## Architecture: two phases, not one straight line

**Knowledge Base phase** (run once per corpus, persisted): Upload files → Parsing → Chunking → Embedding → saved as a named KB (Chroma persist dir + metadata.json recording which embedding model built it — query time must reuse that exact model, not a dropdown).

**Query phase** (run once per question, against a selected KB): Query text + web toggle → Retrieval (KB, blended with live web via hybrid-RRF *only if web toggle is on*) → Reranking → **optional ReAct refinement loop** → Generation.

Both phases keep **stepwise** (acknowledge each stage) and **run-all** modes, matching the existing `Pipeline.run_all(interactive=True/False)` pattern — just split across two entry points instead of one.

## The ReAct loop (new stage, optional, its own toggle)

- Sits after Reranking, before Generation.
- When ON: an LLM judges whether reranked context is sufficient to answer the query. If not, it rewrites/narrows the query and re-runs Retrieval → Reranking. Capped at a configurable max iterations (default 3, range 1-5).
- **Judge model**: dropdown of available models (whatever API keys are supplied), **default Gemini Flash**.
- **Web toggle governs both Retrieval and the ReAct loop identically**: if web search is off, no stage ever calls it. If on, it's used at Retrieval *and* re-queried on every ReAct iteration (this was explicitly chosen over a cheaper "first-pass-only" option — cost per iteration should be shown in the trace so the trade-off is visible, not hidden).
- When OFF, pipeline behaves exactly as Milestone 2 — zero added cost or latency.

## Evaluation — every stage, no exceptions

| Stage | Phase | Eval | Cost |
|---|---|---|---|
| Parsing | KB | heuristic | free |
| Chunking | KB | heuristic | free |
| Embedding | KB | heuristic | free |
| Retrieval (+ web blend if on) | Query | heuristic | free (+ Tavily cost if on) |
| Reranking | Query | heuristic | free |
| ReAct loop (optional) | Query | heuristic — iterations run, sufficiency reached y/n, score delta before/after | LLM judge cost per iteration, itemized |
| Generation | Query | full RAGAS: reference-free always; + reference-based if a gold answer is supplied | LLM cost |

Every single stage returns the same `StageReport` shape already defined in Milestone 2: **output preview + eval scores + plain-language recommendation.** No stage is exempt.

## Error handling

Reuse and extend the existing CLI behavior (prints every completed stage before a clear failure message) into a structured per-stage error card in the UI: **stage, method, what failed, one-line actionable hint** (e.g. "Tavily key missing/rate-limited — web blending skipped, KB-only results used" rather than a raw traceback). Never let a failure hide already-completed stage reports.

## Web search provider

Tavily. Needs its own API key input alongside the Claude/Gemini/OpenAI keys, kept in-memory only (same rule as `pipeline_config.py`'s existing `api_keys` field — never persisted).

## Upload limits (KB creation)

≤20 files, ≤25MB/file, ≤150MB total corpus. Formats: PDF, DOCX, PPTX, MD, TXT (all already parsed in Milestone 2) + scanned PDFs via the existing OCR toggle.

## Extensibility (already solved, don't rebuild)

`registry.py`'s decorator pattern already makes "add/remove a dropdown option" = "add/remove a file." Keep using it for the new ReAct-loop registrations too.

## UI / aesthetic

Keep the mockup's existing design system (paper bg, ink text, teal accent, JetBrains Mono + Space Grotesk) — it already reads as a technical lab notebook, not a generic chat app. Wire it to real data instead of the simulated `INITIAL_STAGES`. Keep the per-stage "Learn" panels, the "✦ Suggest" advisor button, and the system-prompt rewrite block — all already designed in the mockup and backed by `llm_clients.py`.

## Local run target

MacBook Air 16GB/512GB. Chroma runs in-process (no server). MiniLM/BGE-small embeddings run locally. Only generation, non-local embeddings, judge calls, and Tavily need network + keys.

## Cloud deploy path (later, not blocking local dev)

Docker for backend (FastAPI + Chroma persist dir as a mounted volume) and frontend (static build). API keys as environment variables at the host. No architecture change needed.

## Engagement features worth adding (nice-to-have, sequence after core works)

- Compare mode: two configs side by side, diff the eval numbers.
- Exportable run notebook (markdown/PDF of a full run — doubles as an interview portfolio piece).
- Running cost/token tally across the session.
- "Explain this metric" tooltip with the real formula.
- ReAct-loop app as its own tab later, sharing the same LLM-client/tracing/eval infra.

## Build order suggestion for Claude Code

1. FastAPI app wrapping the existing `Pipeline` class, split into `build_kb()` and `answer_query()`, streaming `StageReport`s over SSE.
2. Add the ReAct-loop module (`core/react_loop.py`), registered like every other stage.
3. Add Tavily client + heuristic eval for the blended retrieval stage.
4. Wire `rag-lab-mockup.jsx` to the real endpoints, replacing `INITIAL_STAGES` simulation.
5. KB picker (use existing / create new) + upload UI + limits above.
6. Structured error cards.
7. `SETUP.md` for Mac + Windows, `Dockerfile`s for later cloud deploy.

---

## Status: built (this milestone)

All 7 build-order steps above are implemented. Notable decisions and known gaps, for whoever picks this up next:

- **Two API request per query in "step" mode is not true server-side cost gating.** The UI's step-by-step pacing (buffer-and-reveal one stage at a time) is client-side only — the backend still computes the full `answer_query()` run in one pass regardless of run mode, matching how the original mockup's own step mode worked (a client-paced reveal, not a real pause). True mid-run cost gating (stopping the server before an expensive stage until the user acknowledges) would need a resumable/session-based endpoint design — not built here, flagged as a real limitation if cost control mid-run ever becomes a hard requirement.
- **"Compare mode" and the other engagement nice-to-haves are deliberately deferred**, per the brief's own sequencing note.
- **`ragas` 0.4.x has an upstream bug**: it unconditionally imports a `langchain_community` submodule that's since been dropped (that package is being sunset). `core/evaluation/ragas_eval.py` shims the one missing symbol; `requirements.txt` pins `ragas==0.4.3` to keep this verified combination stable.
- **KB data lives in `data/kbs/`** by default, overridable via `GLASSBOX_DATA_DIR` — see `SETUP.md` for why this matters if the repo lives in a cloud-synced folder (OneDrive/Dropbox/iCloud).
