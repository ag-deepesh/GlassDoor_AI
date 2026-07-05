# GlassBox — Milestone 2: full pipeline, evaluation-driven development

Builds on Milestone 1 (parsing). This milestone adds every remaining
stage — chunking, embedding, vector store, retrieval, reranking,
generation — plus the evaluation layer and the orchestrator that ties it
all together, runnable step-by-step or all at once.

## What's here

```
core/
  schemas.py            # + StageReport: {output, trace, eval_reference_free, eval_with_reference, acknowledged}
  pipeline_config.py     # every dropdown/numeric-input choice, in one place -- also the "export config" shape
  pipeline.py             # orchestrator: run_<stage>() one at a time, or run_all(interactive=True/False)
  tokenizer.py             # ~4-chars/token approximation, no network dependency (tiktoken needs one)

  chunking/
    base.py, fixed.py, recursive.py, sentence.py, markdown_structure.py, semantic.py
    # chunk_size/overlap are manual, default 512/64 (ChunkConfig)
    # markdown_structure groups by heading hierarchy (reuses section_path from md_parser.py)
    # semantic merges sentences while embedding-similarity stays high (embed_fn injectable for testing)

  embedding/
    base.py, text_local.py (MiniLM-L6 default, BGE-small), text_gemini.py, text_openai.py (3-small/large)
    image_caption.py        # caption-text-embed (default) + CLIP (local, advanced) -- registered as "image_embedding"
    # BGE-M3 intentionally NOT registered yet, per your "leave it for now" call

  vectorstore/chroma_store.py   # two collections: text_chunks, image_assets, sharing doc_id

  retrieval/
    base.py, semantic.py, keyword.py (BM25), hybrid_rrf.py (Reciprocal Rank Fusion, k=60)
    multimodal.py            # the "Result mode" control: text-only / joint / separate-merge
    # top_k is manual everywhere, default 10

  reranking/reranking.py    # none (passthrough), cross-encoder (local, ms-marco-MiniLM)
  generation/generation.py   # claude-sonnet, gemini-2.5-flash/pro, gpt-4o-mini -- thin wrappers over llm_clients

  llm_clients.py             # ONE interface for Claude/Gemini/OpenAI: chat(), caption_image(),
                              # rewrite_system_prompt(), suggest_option() -- used by generation AND the UI's
                              # "Rewrite with LLM" / "✦ Suggest" buttons

  evaluation/
    heuristics.py            # zero-cost, rule-based reports for parsing/chunking/embedding/retrieval
    ragas_eval.py              # the actual `ragas` PyPI package for generation-stage metrics:
                                #   reference-free: Faithfulness, AnswerRelevancy, ContextPrecisionWithoutReference
                                #   +with-reference: ContextPrecisionWithReference, ContextRecall,
                                #                    AnswerCorrectness, SemanticSimilarity

cli/main.py    # parse | chunk | run [--step] | show-registry -- every stage runnable standalone
```

## A design decision worth explaining: where does RAGAS actually run?

Full RAGAS metrics need a real *answer* to judge against — they don't
cleanly apply to, say, chunking in isolation. So the eval-driven-development
principle is split two ways, both giving "concise report + recommendation"
at every stage, but through different mechanisms:

- **Parsing / Chunking / Embedding / Retrieval** → `heuristics.py`. Fast,
  deterministic, rule-based, **zero LLM tokens spent**. E.g. chunking
  reports mean/std token size and flags high variance; retrieval reports
  score spread and flags a wide gap between top and bottom results.
- **Generation** → `ragas_eval.py`. This is where a real answer exists, so
  the actual `ragas` package runs Faithfulness / AnswerRelevancy /
  ContextPrecision (+ the reference-based set if a gold answer is given).

This means most of the pipeline's "evaluate every step" requirement costs
nothing to run, and the one expensive LLM-judge call happens exactly once,
at the end, where it's most meaningful — directly serving requirement 8
(use tokens judiciously).

## A real dependency conflict we hit and fixed

Installing `ragas` fresh pulls in its own compatible `langchain-core` /
`langchain` / `langchain-community` versions and imports cleanly. Installing
`langchain-openai` or `langchain-anthropic` *first* pulls newer, incompatible
`langchain-core` pins and breaks `ragas`'s imports. **Fix: install `ragas`
first, then `langchain-anthropic` on top** — that's the order baked into
`requirements.txt`. This is exactly the fragility flagged before choosing
the real `ragas` package over a custom reimplementation — it's real, but
resolved.

Also: `ragas` 0.4.x's own metric classes (`Faithfulness`, `AnswerRelevancy`,
etc.) take a native provider client directly via `ragas.llms.llm_factory`
— **no langchain wrapper needed at the metric level** for Claude/OpenAI.
`langchain-anthropic` is still required as a transitive import inside
`ragas`'s judge-construction path in this version.

## What's tested vs. what needs your Mac's internet + real keys

This sandbox has restricted network access (no `huggingface.co`, no live
LLM APIs) and no real API keys, so testing split into two tiers:

✅ **Fully tested, offline, real data:**
- All 6 parsers/chunkers on the 5-format sample corpus (Milestone 1 + this one)
- `Pipeline.run_parsing()` and `run_chunking()` end-to-end, including their
  heuristic eval reports
- The registry (`show-registry` lists all 24 registered methods across 6 stages)
- The step-by-step acknowledgment gate (`--step`, tested with piped input —
  confirmed it actually halts on "n", not just a soft skip)
- `ragas` + `langchain-anthropic` import cleanly together (the conflict above, resolved)
- The CLI's error handling (confirmed it prints all completed stage reports
  before a clear failure message, not a lost trace + raw stack)

⚠️ **Structurally correct, needs your Mac to fully verify:**
- `run_embedding()` onward — MiniLM/BGE-small download from `huggingface.co`
  (blocked here, fine on your Mac)
- Gemini/OpenAI embedding and all generation calls — need real API keys
- `ragas_eval.get_judge()` — needs a real key to actually score

## Try it yourself

```bash
pip install -r requirements.txt
brew install tesseract   # macOS, for OCR

python3 -m cli.main show-registry
python3 -m cli.main chunk sample_corpus --method markdown_structure --chunk-size 60 --overlap 10

# Full pipeline, all at once:
python3 -m cli.main run sample_corpus -q "How does self-attention work?" \
    --claude-key sk-ant-... 

# Step-by-step, with acknowledgment required at each stage:
python3 -m cli.main run sample_corpus -q "How does self-attention work?" \
    --claude-key sk-ant-... --step
```

## Next milestone (not built yet)

FastAPI app streaming `StageReport`s over SSE, and wiring the React UI
(`rag-lab-mockup.jsx`) to this real backend instead of simulated data.
