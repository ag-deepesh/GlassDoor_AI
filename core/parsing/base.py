from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from core.schemas import ParsedDoc


class BaseParser(ABC):
    """Every parser takes a file path (+ shared options) and returns a ParsedDoc.
    Subclasses only implement `parse` -- nothing else in the pipeline needs
    to know this class exists beyond the registry entry."""

    def __init__(self, extract_images: bool = True, ocr: bool = False, assets_dir: Path | None = None):
        self.extract_images = extract_images
        self.ocr = ocr
        self.assets_dir = assets_dir or Path("assets")
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def parse(self, path: Path, doc_id: str) -> ParsedDoc: ...
