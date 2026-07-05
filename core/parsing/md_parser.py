from __future__ import annotations
from pathlib import Path
from markdown_it import MarkdownIt

from core.registry import register
from core.schemas import ParsedDoc, TextBlock, ImageAsset
from core.parsing.base import BaseParser

_md = MarkdownIt()


@register("parsing", "md")
class MdParser(BaseParser):
    def parse(self, path: Path, doc_id: str) -> ParsedDoc:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        tokens = _md.parse(raw)

        text_blocks: list[TextBlock] = []
        images: list[ImageAsset] = []
        section_path = ""
        img_n = 0
        expecting_heading_text = False  # True right after a heading_open token

        for tok in tokens:
            if tok.type == "heading_open":
                expecting_heading_text = True
                continue
            if tok.type != "inline" or not tok.content.strip():
                continue

            if expecting_heading_text:
                section_path = tok.content.strip()
                expecting_heading_text = False
                continue  # heading text itself isn't a body block

            text_blocks.append(TextBlock(text=tok.content.strip(), section_path=section_path))

            if self.extract_images:
                for child in tok.children or []:
                    if child.type == "image":
                        candidate = path.parent / child.attrs.get("src", "")
                        if candidate.exists():
                            img_n += 1
                            img_id = f"{doc_id}_img{img_n}"
                            out_path = self.assets_dir / f"{img_id}{candidate.suffix}"
                            out_path.write_bytes(candidate.read_bytes())
                            images.append(ImageAsset(path=out_path, id=img_id,
                                                      caption=child.attrs.get("alt")))

        return ParsedDoc(
            doc_id=doc_id, source_path=path, format="md",
            text_blocks=text_blocks, images=images, n_pages=1,
            meta={"parser": "markdown-it-py"},
        )


@register("parsing", "txt")
class TxtParser(BaseParser):
    def parse(self, path: Path, doc_id: str) -> ParsedDoc:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return ParsedDoc(
            doc_id=doc_id, source_path=path, format="txt",
            text_blocks=[TextBlock(text=raw.strip())], n_pages=1,
            meta={"parser": "plain-text"},
        )
