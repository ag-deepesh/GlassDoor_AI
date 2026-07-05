"""
The "Assorted" parsing option.

A mixed corpus (pdf + docx + pptx + md together) is the normal case for
real use, not the exception -- so Assorted isn't a separate parser, it's a
one-line dispatch table. Importing this module registers "pdf", "docx",
"pptx", "md", and "txt" as a side effect (each submodule self-registers),
so Assorted can look them all up by extension.
"""
from __future__ import annotations
from pathlib import Path

from core.registry import register, get
from core.schemas import ParsedDoc
from core.parsing.base import BaseParser

# Import for registration side-effects.
from core.parsing import pdf_parser, docx_parser, pptx_parser, md_parser  # noqa: F401

EXT_TO_METHOD = {
    ".pdf": "pdf", ".docx": "docx", ".pptx": "pptx",
    ".md": "md", ".markdown": "md", ".txt": "txt",
}


@register("parsing", "assorted")
class AssortedParser(BaseParser):
    def parse(self, path: Path, doc_id: str) -> ParsedDoc:
        method = EXT_TO_METHOD.get(path.suffix.lower())
        if method is None:
            raise ValueError(
                f"Assorted parsing has no handler for '{path.suffix}'. "
                f"Supported: {sorted(EXT_TO_METHOD)}"
            )
        parser_cls = get("parsing", method)
        parser = parser_cls(extract_images=self.extract_images, ocr=self.ocr, assets_dir=self.assets_dir)
        return parser.parse(path, doc_id)


def parse_corpus(paths: list[Path], extract_images: bool = True, ocr: bool = False,
                  assets_dir: Path | None = None) -> list[ParsedDoc]:
    """Convenience: parse a whole folder of mixed formats in one call."""
    router = AssortedParser(extract_images=extract_images, ocr=ocr, assets_dir=assets_dir)
    return [router.parse(p, doc_id=p.stem) for p in paths]
