from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Every dropdown/numeric-input choice from the UI, in one place. This
    is what gets saved/loaded as a shareable JSON config (per the platform's
    'export config' feature) and what the CLI's flags map onto 1:1."""

    # Parsing
    parsing_method: str = "assorted"
    extract_images: bool = True
    ocr: bool = False

    # Chunking
    chunking_method: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Embedding
    embedding_method: str = "minilm-l6"
    image_embedding_method: str = "caption-text-embed"  # or "clip-local"

    # Retrieval
    retrieval_method: str = "hybrid-rrf"
    top_k: int = 10
    result_mode: str = "text-only"  # "text-only" | "joint" | "separate-merge"

    # Web search (Tavily) -- governs Retrieval AND the ReAct loop identically:
    # off, no stage ever calls it; on, it's blended at Retrieval AND
    # re-queried on every ReAct iteration (not just the first pass).
    web_enabled: bool = False

    # Reranking
    reranking_method: str = "cross-encoder"
    rerank_keep_top: int = 4

    # ReAct refinement loop (optional) -- sits after Reranking, before
    # Generation. When off, the pipeline behaves exactly as Milestone 2.
    react_enabled: bool = False
    react_max_iterations: int = 3      # range 1-5
    react_judge_method: str = "gemini-2.5-flash"  # distinct from judge_provider (RAGAS judge) below

    # Generation
    generation_method: str = "claude-sonnet"
    system_prompt: str = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "Cite chunk ids like [#47]. If the context is insufficient, say so."
    )
    vision_grounded: bool = False

    # API keys (kept in memory only -- never persisted with the config export)
    api_keys: dict = field(default_factory=dict)  # {"claude": "...", "gemini": "...", "openai": "...", "tavily": "..."}

    # Evaluation (RAGAS judge -- distinct from react_judge_method above).
    # None means "auto": match whatever provider generation used, so RAGAS
    # never needs a key the user didn't already supply.
    judge_provider: str | None = None

    # Image captioning (used by the caption-text-embed image embedding method)
    caption_provider: str = "claude"

    def to_shareable_json(self) -> dict:
        """Everything except api_keys -- safe to export/share."""
        d = self.__dict__.copy()
        d.pop("api_keys", None)
        return d
