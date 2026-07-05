"""
OCR fallback for scanned pages (text trapped in a picture, not a real text layer).

Kept as a standalone helper -- any parser calls `ocr_image(path)` when it
detects a page/slide has near-zero native text next to an image. This keeps
OCR an explicit, toggleable choice (per the "OCR toggle" requirement) rather
than baked silently into one parser.
"""
from __future__ import annotations
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

MIN_NATIVE_CHARS = 20  # below this, treat the page as "scanned" and worth OCR-ing


def ocr_available() -> bool:
    return _OCR_AVAILABLE


def ocr_image(image_path: Path) -> str:
    """Run Tesseract on one image file, return recovered text (empty string on failure)."""
    if not _OCR_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(Image.open(image_path)).strip()
    except Exception:
        return ""


def needs_ocr(native_text: str) -> bool:
    return len(native_text.strip()) < MIN_NATIVE_CHARS
