from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .embeddings import embed_texts, l2_normalize


Json = dict[str, Any]


@dataclass
class VectorSearchResult:
    score: float
    text: str
    metadata: Json


class VectorStore:
    """
    Simple FAISS-backed semantic search.
    Stores raw document text + metadata alongside the index.
    """

    def __init__(self, index: faiss.Index, items: list[dict[str, Any]], embed_model: str) -> None:
        self.index = index
        self.items = items
        self.embed_model = embed_model
        self.dim = index.d

    @classmethod
    def load_or_create(
        cls,
        index_dir: Path,
        documents: list[str],
        metadatas: list[Json],
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        force_rebuild: bool = False,
    ) -> "VectorStore":
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "metadata.json"

        if not force_rebuild and index_path.exists() and meta_path.exists():
            index = faiss.read_index(str(index_path))
            items = json.loads(meta_path.read_text(encoding="utf-8"))
            return cls(index=index, items=items, embed_model=embed_model)

        # Build from scratch.
        if not documents:
            # Create a minimal index; will error on search/add if used without data.
            raise ValueError("No documents provided for vector store build.")

        vectors = embed_texts(documents, model_name=embed_model)
        vectors = l2_normalize(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        items = []
        for i, (text, md) in enumerate(zip(documents, metadatas)):
            items.append({"id": i, "text": text, "metadata": md})

        faiss.write_index(index, str(index_path))
        meta_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

        return cls(index=index, items=items, embed_model=embed_model)

    def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        if self.index.ntotal == 0:
            return []

        q_vec = embed_texts([query], model_name=self.embed_model)
        q_vec = l2_normalize(q_vec)[0:1]

        scores, ids = self.index.search(q_vec, top_k)
        scores = scores[0].tolist()
        ids = ids[0].tolist()

        results: list[VectorSearchResult] = []
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(self.items):
                continue
            item = self.items[idx]
            results.append(VectorSearchResult(score=float(score), text=item["text"], metadata=item["metadata"]))
        return results

    def add_document(self, text: str, metadata: Json, index_dir: Path | None = None) -> None:
        """
        Adds a single document to the FAISS index and persists the updated index.
        """
        vec = embed_texts([text], model_name=self.embed_model)
        vec = l2_normalize(vec)

        if vec.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: index dim={self.dim}, embed dim={vec.shape[1]}")

        new_id = len(self.items)
        self.index.add(vec.astype(np.float32))
        self.items.append({"id": new_id, "text": text, "metadata": metadata})

        if index_dir is not None:
            index_path = index_dir / "index.faiss"
            meta_path = index_dir / "metadata.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            import faiss  # noqa: F401

            faiss.write_index(self.index, str(index_path))
            meta_path.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8")

