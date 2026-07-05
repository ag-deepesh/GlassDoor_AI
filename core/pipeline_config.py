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
    image_candidate_k: int = 8  # candidates fetched BEFORE reranking cuts to rerank_keep_top --
                                 # must stay wider than keep_top or reranking has nothing to choose between

    # Web search (Tavily) -- governs Retrieval AND the ReAct loop identically:
    # off, no stage ever calls it; on, it's blended at Retrieval AND
    # re-queried on every ReAct iteration (not just the first pass).
    web_enabled: bool = False

    # Reranking -- rerank_keep_top is a GLOBAL cap shared by text and images
    # together (not "top text + unlimited images"); vision_rerank_enabled adds
    # a second-pass vision-LLM relevance check on top of the cross-encoder/
    # caption-surrogate score, using caption_provider's key (already vision-
    # capable -- see _VISION_CAPABLE in generation.py).
    reranking_method: str = "cross-encoder"
    rerank_keep_top: int = 4
    vision_rerank_enabled: bool = False

    # ReAct refinement loop (optional) -- sits after Reranking, before
    # Generation. When off, the pipeline behaves exactly as Milestone 2.
    react_enabled: bool = False
    react_max_iterations: int = 3      # range 1-5
    react_judge_method: str = "gemini-2.5-flash"  # distinct from judge_provider (RAGAS judge) below

    # Generation
    generation_method: str = "claude-sonnet"
    system_prompt: str = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "Cite text chunk ids like [#47], and image ids like [img#12] whenever a figure "
        "actually informed your answer -- omit an image tag if you didn't rely on it. "
        "If the context is insufficient, say so."
    )
    vision_grounded: bool = False

    # Generation sampling params -- always explicit, never provider defaults
    # silently applied, so the trace/UI always shows what was actually used.
    temperature: float = 0.7
    top_p: float = 1.0
    gen_top_k: int = 40      # distinct from retrieval's top_k; unsupported by openai/groq, ignored there
    max_tokens: int = 1024

    # Chat history -- role-based {"role": "user"|"assistant", "content": str}
    # turns from earlier in the conversation, oldest first. Kept client-side
    # by the UI; the pipeline just forwards whatever it's given to the LLM.
    history: list = field(default_factory=list)

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
