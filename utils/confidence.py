from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoutingDecision:
    final_confidence: float
    decision: str  # auto_resolve | ask_clarification | escalate_admin
    badge_color: str  # for UI


def scale_cosine_to_unit(sim: float) -> float:
    """
    Cosine similarity is [-1, 1]. Convert to [0, 1] for routing.
    """
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def compute_final_confidence(intent_confidence: float, retrieval_cosine_similarity: float) -> float:
    """
    Blend classifier confidence and retrieval similarity.
    Weights: retrieval 0.6, intent 0.4.
    """
    retrieval_score = scale_cosine_to_unit(retrieval_cosine_similarity)
    w_intent = 0.4
    w_retrieval = 0.6
    final = (w_retrieval * retrieval_score) + (w_intent * intent_confidence)
    return max(0.0, min(1.0, final))


def route_by_confidence(final_confidence: float) -> RoutingDecision:
    if final_confidence > 0.65:
        return RoutingDecision(final_confidence=final_confidence, decision="auto_resolve", badge_color="green")
    if final_confidence >= 0.4:
        return RoutingDecision(final_confidence=final_confidence, decision="ask_clarification", badge_color="orange")
    return RoutingDecision(final_confidence=final_confidence, decision="escalate_admin", badge_color="red")

