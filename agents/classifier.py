from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.embeddings import embed_texts, l2_normalize, softmax
from utils.logger import call_gemini_json, safe_json_loads


Json = dict[str, Any]


INTENT_CATEGORIES = ["payment_issue", "refund", "delivery_delay", "account_issue", "order_cancel"]
CATEGORY_MAP = {"return": "refund"}


def _normalize_category(cat: str) -> str:
    return CATEGORY_MAP.get(cat, cat)


def _extract_order_id(text: str) -> str | None:
    # Example format: ORD00012
    m = re.search(r"\b(ORD\d{5})\b", text or "")
    return m.group(1) if m else None


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    method: str  # gemini | embeddings


class IntentClassifier:
    """
    Intent classifier with a Gemini option and a semantic-embedding fallback.
    """

    def __init__(self, knowledge_base: list[Json], prototype_temperature: float = 0.8) -> None:
        self.prototype_temperature = prototype_temperature
        self.prototypes = self._build_prototypes(knowledge_base)

        # Pre-embed prototypes for deterministic fallback.
        self._prototype_embeddings: dict[str, np.ndarray] = {}
        for cat, proto_text in self.prototypes.items():
            v = embed_texts([proto_text])
            v = l2_normalize(v)[0]
            self._prototype_embeddings[cat] = v

    def _build_prototypes(self, knowledge_base: list[Json]) -> dict[str, str]:
        prototypes: dict[str, list[str]] = {c: [] for c in INTENT_CATEGORIES}
        for kb_item in knowledge_base:
            cat = _normalize_category(str(kb_item.get("category", "")))
            if cat not in prototypes:
                continue
            title = kb_item.get("title", "")
            content = kb_item.get("content", "")
            keywords = kb_item.get("keywords", [])
            steps = kb_item.get("resolution_steps", [])
            kw_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
            steps_str = "\n".join([f"- {s}" for s in steps]) if isinstance(steps, list) else str(steps)
            prototypes[cat].append(f"{title}\n{content}\nKeywords: {kw_str}\nResolution steps:\n{steps_str}")

        # If a category has no docs, create a minimal prototype from the category name.
        merged: dict[str, str] = {}
        for cat, parts in prototypes.items():
            if parts:
                merged[cat] = "\n\n".join(parts)
            else:
                merged[cat] = f"Intent category: {cat}. Customer support issue."
        return merged

    def classify(self, query: str, logger=None) -> ClassificationResult:
        order_id = _extract_order_id(query)

        # Prefer Gemini if configured.
        gemini_prompt = (
            "Classify the customer support query into one of exactly these categories:\n"
            "- payment_issue\n"
            "- refund\n"
            "- delivery_delay\n"
            "- account_issue\n"
            "- order_cancel\n\n"
            "Return ONLY valid JSON with keys:\n"
            "{\n"
            "  \"category\": <one of the categories>,\n"
            "  \"confidence\": <number between 0 and 1>,\n"
            "  \"rationale\": <short string>\n"
            "}\n\n"
            f"Query: {query}\n"
            f"Detected order_id (if any): {order_id}\n"
        )

        gemini_resp = call_gemini_json(gemini_prompt, schema_hint="category, confidence, rationale")
        if gemini_resp:
            parsed = safe_json_loads(gemini_resp)
            if parsed and isinstance(parsed.get("confidence"), (int, float)) and parsed.get("category") in INTENT_CATEGORIES:
                cat = str(parsed["category"])
                conf = float(parsed["confidence"])
                if logger:
                    logger.step(
                        "intent_classification_gemini",
                        input_data={"query": query},
                        output_data=parsed,
                        confidence=conf,
                        streaming_msg=f"Intent detected: {cat} (confidence: {conf:.2f})",
                        meta={"order_id": order_id},
                    )
                return ClassificationResult(category=cat, confidence=max(0.0, min(1.0, conf)), method="gemini")

        # Fallback: semantic similarity between query and category prototypes.
        q_vec = embed_texts([query])[0:1]
        q_vec = l2_normalize(q_vec)[0]

        sims = []
        for cat in INTENT_CATEGORIES:
            p_vec = self._prototype_embeddings[cat]
            sims.append(float(np.dot(q_vec, p_vec)))
        sims_arr = np.asarray(sims, dtype=np.float32)

        # Softmax gives a probabilistic confidence distribution.
        probs = softmax(sims_arr, temperature=self.prototype_temperature)
        best_idx = int(np.argmax(probs))
        best_cat = INTENT_CATEGORIES[best_idx]
        best_conf = float(probs[best_idx])

        if logger:
            logger.step(
                "intent_classification_embeddings",
                input_data={"query": query, "order_id": order_id},
                output_data={"category": best_cat, "confidence": best_conf, "prototype_similarities": dict(zip(INTENT_CATEGORIES, sims_arr.tolist()))},
                confidence=best_conf,
                streaming_msg=f"Intent detected: {best_cat} (confidence: {best_conf:.2f})",
                meta={"order_id": order_id},
            )

        return ClassificationResult(category=best_cat, confidence=best_conf, method="embeddings")

