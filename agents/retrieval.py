from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.vector_store import VectorStore


Json = dict[str, Any]


def _normalize_category(cat: str) -> str:
    # The provided dataset uses "return" but your system routes into "refund".
    if cat == "return":
        return "refund"
    return cat


def build_document_corpus(knowledge_base: list[Json], past_tickets: list[Json]) -> tuple[list[str], list[Json]]:
    """
    Build RAG documents from knowledge base and past tickets.
    """
    docs: list[str] = []
    metadatas: list[Json] = []

    for kb_item in knowledge_base:
        cat = _normalize_category(str(kb_item.get("category", "")))
        title = kb_item.get("title", "")
        content = kb_item.get("content", "")
        steps = kb_item.get("resolution_steps", [])
        steps_str = "\n".join([f"- {s}" for s in steps]) if isinstance(steps, list) else str(steps)

        doc_text = (
            f"Type: knowledge_base\n"
            f"Category: {cat}\n"
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"Resolution Steps:\n{steps_str}\n"
        )
        docs.append(doc_text)
        metadatas.append(
            {
                "source": "knowledge_base",
                "category": cat,
                "title": title,
                "kb_id": kb_item.get("id"),
                "content": content,
                "resolution_steps": steps if isinstance(steps, list) else [str(steps)],
            }
        )

    for ticket in past_tickets:
        cat = _normalize_category(str(ticket.get("category", "")))
        query = ticket.get("query", "")
        resolution = ticket.get("resolution", "")
        status = ticket.get("status", "")
        created_at = ticket.get("created_at", "")

        doc_text = (
            f"Type: past_ticket\n"
            f"Category: {cat}\n"
            f"Status: {status}\n"
            f"CreatedAt: {created_at}\n"
            f"Query: {query}\n"
            f"Resolution: {resolution}\n"
        )
        docs.append(doc_text)
        metadatas.append(
            {
                "source": "past_ticket",
                "category": cat,
                "ticket_id": ticket.get("ticket_id"),
                "status": status,
                "created_at": created_at,
                "query": query,
                "resolution": resolution,
            }
        )

    return docs, metadatas


@dataclass
class RetrievalResult:
    category: str
    score: float
    text: str
    metadata: Json


class Retriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        hits = self.vector_store.search(query, top_k=top_k)
        results: list[RetrievalResult] = []
        for h in hits:
            cat = str(h.metadata.get("category", ""))
            results.append(RetrievalResult(category=cat, score=h.score, text=h.text, metadata=h.metadata))
        return results

