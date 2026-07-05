from __future__ import annotations
from core.registry import register
from core.schemas import ImageAsset
from core.embedding.base import BaseEmbedder
from core.llm_clients import get_client


class BaseImageEmbedder:
    """Separate registry stage ('image_embedding') from text embedding,
    because the two options here have fundamentally different mechanics --
    one calls a vision LLM, the other loads a local vision model -- even
    though both end up producing vectors comparable to text chunks."""

    def embed_images(self, images: list[ImageAsset]) -> list[list[float]]: ...


@register("image_embedding", "caption-text-embed")
class CaptionThenEmbed(BaseImageEmbedder):
    """Default: caption each image once via a vision LLM (cached on the
    ImageAsset), then embed the caption with the SAME text embedder used
    for chunks -- so images and text share one vector space and can be
    ranked together with plain cosine similarity. One vision call per
    image at ingest time, never repeated per query."""

    def __init__(self, text_embedder: BaseEmbedder, caption_provider: str, caption_api_key: str,
                 caption_model: str | None = None):
        self._text_embedder = text_embedder
        self._client = get_client(caption_provider, caption_api_key)
        self._model = caption_model

    def embed_images(self, images: list[ImageAsset]) -> list[list[float]]:
        captions = []
        for img in images:
            if not img.caption:  # cache: only caption once, ever
                kwargs = {"model": self._model} if self._model else {}
                img.caption = self._client.caption_image(img.path, **kwargs).text.strip()
            captions.append(img.caption)
        return self._text_embedder.embed(captions)


@register("image_embedding", "clip-local")
class CLIPEmbedder(BaseImageEmbedder):
    """Advanced/local option: CLIP embeds the image pixels directly, no
    captioning step and no API cost -- but text queries and images only
    compare validly if the QUERY is also embedded with this same CLIP model
    (not the text embedder used for chunks). Best for pure visual-similarity
    search ("find charts like this one"), not for mixing scores directly
    with text-chunk cosine similarities without calibration."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("clip-ViT-B-32")

    def embed_images(self, images: list[ImageAsset]) -> list[list[float]]:
        from PIL import Image
        self._ensure_loaded()
        pil_images = [Image.open(img.path) for img in images]
        return self._model.encode(pil_images, normalize_embeddings=True).tolist()

    def embed_query_text(self, query: str) -> list[float]:
        """CLIP has its own text tower -- use this, not the chunk text
        embedder, to query against CLIP image vectors."""
        self._ensure_loaded()
        return self._model.encode([query], normalize_embeddings=True)[0].tolist()
