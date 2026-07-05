from __future__ import annotations
from pathlib import Path
import fitz  # PyMuPDF

from core.registry import register
from core.schemas import ParsedDoc, TextBlock, ImageAsset
from core.parsing.base import BaseParser
from core.parsing.ocr import ocr_image, needs_ocr, ocr_available


@register("parsing", "pdf")
class PDFParser(BaseParser):
    def parse(self, path: Path, doc_id: str) -> ParsedDoc:
        doc = fitz.open(path)
        text_blocks: list[TextBlock] = []
        images: list[ImageAsset] = []
        img_n = 0

        for page_idx, page in enumerate(doc, start=1):
            native_text = page.get_text().strip()
            source_ocr = False

            if self.ocr and ocr_available() and needs_ocr(native_text):
                # Render the whole page to an image and OCR it -- this is the
                # "scanned page with no text layer" case.
                pix = page.get_pixmap(dpi=200)
                tmp = self.assets_dir / f"{doc_id}_p{page_idx}_render.png"
                pix.save(tmp)
                recovered = ocr_image(tmp)
                if recovered:
                    native_text = recovered
                    source_ocr = True

            if native_text:
                text_blocks.append(TextBlock(text=native_text, page=page_idx, source_ocr=source_ocr))

            if self.extract_images:
                for img_ref in page.get_images(full=True):
                    xref = img_ref[0]
                    try:
                        base = doc.extract_image(xref)
                    except Exception:
                        continue
                    img_n += 1
                    img_id = f"{doc_id}_img{img_n}"
                    out_path = self.assets_dir / f"{img_id}.{base['ext']}"
                    out_path.write_bytes(base["image"])
                    images.append(ImageAsset(path=out_path, page=page_idx, id=img_id))

        return ParsedDoc(
            doc_id=doc_id, source_path=path, format="pdf",
            text_blocks=text_blocks, images=images, n_pages=doc.page_count,
            meta={"parser": "PyMuPDF"},
        )
