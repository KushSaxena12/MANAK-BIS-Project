import os
from pathlib import Path

from fastembed import TextEmbedding

# Persistent, project-local cache dir (survives restarts/redeploys, unlike a
# container's default ~/.cache which may not be preserved).
_CACHE_DIR = Path(__file__).resolve().parents[2] / ".model_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class EmbeddingService:
    """
    Generates semantic embeddings using BAAI/bge-small-en-v1.5 (ONNX, via fastembed).

    The model is loaded once when this service is created. Weights are
    cached under app/.model_cache so subsequent restarts load from disk
    instead of re-downloading.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.model_name = model_name

        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(_CACHE_DIR),
        )

    def embed(self, text: str) -> list[float]:
        """
        Generate a normalized embedding for a single query.
        """

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        embedding = list(self.model.embed([text.strip()]))[0]

        return embedding.tolist()

    def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate normalized embeddings for multiple texts.
        """

        if not texts:
            return []

        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned_texts:
            return []

        embeddings = list(self.model.embed(cleaned_texts))

        return [e.tolist() for e in embeddings]