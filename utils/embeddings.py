from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np


@lru_cache(maxsize=2)
def _load_embedder(model_name: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: Iterable[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embeds a list of texts as float32 vectors.
    """
    embedder = _load_embedder(model_name)
    vectors = embedder.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)
    return np.asarray(vectors, dtype=np.float32)


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / max(temperature, 1e-9)
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / np.sum(exp)

