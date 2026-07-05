"""
Unified data schemas.

Design rule: every parser (pdf/docx/pptx/md/assorted) must return a ParsedDoc.
Everything downstream (chunking, embedding, retrieval) only ever sees this
shape, so a new file format only ever requires a new parser -- nothing else
in the pipeline changes. This is what makes "Assorted" possible.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class TextBlock:
    text: str
    page: int | None = None          # 1-indexed page/slide number, None if n/a
    section_path: str = ""           # e.g. "Heading 1 > Heading 2"
    source_ocr: bool = False         # True if this text came from OCR, not a native text layer


@dataclass
class ImageAsset:
    path: Path                       # where the extracted image bytes live on disk
    page: int | None = None
    caption: str | None = None       # filled in later by the captioning embedder
    bbox: tuple[float, float, float, float] | None = None
    id: str = ""                     # assigned by the parser: "{doc_id}_img{n}"


@dataclass
class TableAsset:
    markdown: str                    # table rendered as markdown, easiest for LLMs to consume
    page: int | None = None
    id: str = ""


@dataclass
class ParsedDoc:
    doc_id: str
    source_path: Path
    format: Literal["pdf", "docx", "pptx", "md", "txt"]
    text_blocks: list[TextBlock] = field(default_factory=list)
    images: list[ImageAsset] = field(default_factory=list)
    tables: list[TableAsset] = field(default_factory=list)
    n_pages: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.text_blocks if b.text.strip())

    def stats(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "format": self.format,
            "n_pages": self.n_pages,
            "n_text_blocks": len(self.text_blocks),
            "n_words": len(self.full_text.split()),
            "n_images": len(self.images),
            "n_tables": len(self.tables),
            "n_ocr_blocks": sum(1 for b in self.text_blocks if b.source_ocr),
        }


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page: int | None = None
    n_tokens: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class TraceEvent:
    """Emitted by every pipeline stage -- this is what the UI's trace panel
    and the CLI's --verbose output both render, so they never drift apart."""
    stage: str
    method: str
    input_summary: str
    output_summary: str
    latency_ms: float
    tokens: int = 0
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class StageReport:
    """What every pipeline stage returns, uniformly -- this IS the
    evaluation-driven-development pattern: a stage's output never travels
    alone, it always comes with what the output looked like, how it scored,
    and what to try next. The UI's per-stage panel and the CLI's --step
    output both render this same object, so they can never drift apart."""
    stage: str
    method: str
    output_preview: str          # short, human-readable summary of what this stage produced
    trace: TraceEvent
    eval_reference_free: dict | None = None   # {"scores": {...}, "recommendation": "..."}
    eval_with_reference: dict | None = None
    acknowledged: bool = False   # set True once the user has reviewed it in step-by-step mode


@dataclass
class StageError:
    """What a stage yields instead of a StageReport when it fails -- the
    structured shape behind every error card in the UI. Never raised past
    already-yielded StageReports: a failure here still leaves every earlier
    stage's report standing, it just marks where the run stopped."""
    stage: str
    method: str
    what_failed: str   # short, human-readable: what broke (not a raw traceback)
    hint: str           # one-line, actionable: what to try next
