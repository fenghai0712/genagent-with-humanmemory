"""Embedding provider abstraction. Default: sentence-transformers (local)."""

import hashlib
import threading
from typing import Optional


class EmbeddingProvider:
    """Wraps a sentence-transformers model. Falls back to hash-based vectors if
    the model can't be loaded."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):
        self.model_name = model_name
        self._model = None
        self._dim: Optional[int] = None
        self._lock = threading.Lock()

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            try:
                self._dim = self._model.get_embedding_dimension()
            except AttributeError:
                self._dim = self._model.get_sentence_embedding_dimension()
        except Exception:
            self._model = None
            self._dim = None

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        return 768  # fallback default

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed(self, text: str | list[str], normalize: bool = True) -> list[float] | list[list[float]]:
        """Return embedding vector(s). Single string -> list[float], list -> list[list[float]].
        Vectors are L2-normalized by default for consistent distance scaling across models."""
        if self._model is not None:
            with self._lock:
                result = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=normalize)
            if isinstance(text, str):
                return result.tolist()
            return [v.tolist() for v in result]
        return self._fallback(text, normalize)

    def _fallback(self, text: str | list[str], normalize: bool = True) -> list[float] | list[list[float]]:
        """Hash-based pseudo-embedding. Deterministic but not semantic."""
        import math

        def _hash_vec(t: str) -> list[float]:
            h = hashlib.sha256(t.encode()).digest()
            dim = self.dim
            vec = []
            for i in range(dim):
                b = h[i % len(h)]
                vec.append(((b / 255.0) * 2.0 - 1.0) * 0.1)
            if normalize:
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 0:
                    vec = [x / norm for x in vec]
            return vec

        if isinstance(text, str):
            return _hash_vec(text)
        return [_hash_vec(t) for t in text]


# Singleton shared across encoder and retrieval engine
_default_provider: Optional[EmbeddingProvider] = None
_default_lock = threading.Lock()


def get_embedding_provider(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                           device: str = "cpu") -> EmbeddingProvider:
    global _default_provider
    if _default_provider is None:
        with _default_lock:
            if _default_provider is None:
                _default_provider = EmbeddingProvider(model_name, device)
    return _default_provider


def reset_embedding_provider():
    global _default_provider
    _default_provider = None
