from __future__ import annotations

import re
from typing import Any

from utils.logger import call_gemini_json, safe_json_loads


Json = dict[str, Any]


def _strip_md(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _sanitize_output_text(text: str) -> str:
    t = (text or "").strip()
    # Never show legacy generic guidance text.
    t = t.replace(
        "Next step: Follow the verified guidance above. If anything changes, reply with your updated order/payment status.",
        "Next step: Reply with your latest order/payment update and I'll guide the exact action.",
    )
    # Prevent nested old ticket resolution blobs from polluting user-facing response.
    t = re.sub(r"Evidence matched:\s*Type:\s*past_ticket.*?(?=\nTool results|\n\nTool results|$)", "Evidence matched: relevant policy and similar case context.", t, flags=re.IGNORECASE | re.DOTALL)
    return t


def _fallback_response(
    query: str,
    category: str,
    routing_decision: str,
    retrieved_snippets: list[Json],
    tool_results: list[Json],
    clarification_question: str | None,
) -> Json:
    top = retrieved_snippets[0]["text"] if retrieved_snippets else ""
    top_meta = retrieved_snippets[0].get("metadata", {}) if retrieved_snippets else {}

    def _find_tool_result(tool_name: str) -> dict[str, Any] | None:
        for tr in tool_results or []:
            if tr.get("tool_name") == tool_name:
                res = tr.get("result")
                return res if isinstance(res, dict) else None
        return None

    def _lower(v: Any) -> str:
        return "" if v is None else str(v).lower()

    # Build an evidence string from the top retrieved item.
    evidence = _strip_md(top)[:520]
    resolution_steps = top_meta.get("resolution_steps", []) if isinstance(top_meta, dict) else []
    if isinstance(resolution_steps, str):
        resolution_steps = [resolution_steps]

    actions_taken: list[str] = []
    for tr in tool_results:
        tool_name = tr.get("tool_name", "")
        ok = tr.get("result", {}).get("ok", True)
        actions_taken.append(f"{tool_name} ({'succeeded' if ok else 'failed'})")

    if not actions_taken and routing_decision in {"auto_resolve", "ask_clarification"}:
        actions_taken.append("Matched your case to relevant policy guidance (no tool call required yet).")

    # Build a high-level tool results block for transparency.
    tool_block = ""
    if tool_results:
        tool_block = "\n\nTool results (high level):\n"
        for tr in tool_results[:3]:
            tname = tr.get("tool_name", "")
            res = tr.get("result", {}) if isinstance(tr.get("result", {}), dict) else {}
            summary = res.get("payment_status") or res.get("order_status") or res.get("account_status") or res.get("refund_status") or res.get("error") or ""
            tool_block += f"- {tname}: {summary}\n"

    # Decision-maker deterministic overrides (use tool outputs, not retrieved "past ticket authority").
    check_payment = _find_tool_result("check_payment_status")
    refund = _find_tool_result("initiate_refund")
    delivery_order = _find_tool_result("get_order_details")
    account = _find_tool_result("get_user_account_status")
    unblock = _find_tool_result("unblock_account")

    if category in {"payment_issue", "refund"} and check_payment:
        pst = _lower(check_payment.get("payment_status"))
        ost = _lower(check_payment.get("order_status"))
        oid = str(check_payment.get("order_id") or "")
        age = check_payment.get("age_hours")

        order_confirmed = ost in {"confirmed", "shipped", "delivered"}
        q = (query or "").lower()
        if category == "refund":
            user_mentions_refund = True
        else:
            user_mentions_refund = any(
                k in q
                for k in [
                    "refund",
                    "not refunded",
                    "didn't get refund",
                    "didnt get refund",
                    "money debited but not refunded",
                    "money deducted but not refunded",
                    "amount not returned",
                    "not returned",
                    "charged back",
                ]
            )
        refund_ok = bool(refund and refund.get("ok") is True)
        refund_status = str(refund.get("refund_status") or "") if refund else ""

        if order_confirmed:
            if ost == "confirmed" and user_mentions_refund:
                final = (
                    f"I checked your order `{oid}` and confirmed that payment was successful and the order is already placed (order_status=`confirmed`).\n\n"
                    f"- payment_status: `{pst}`\n"
                    f"- order_status: `{ost}`\n\n"
                    "Since the order is confirmed, a refund is not applicable for this specific issue.\n"
                    "If you meant a cancellation or return request, tell me and I'll help you initiate the correct process."
                    f"{tool_block}"
                )
            elif ost == "shipped" and any(
                k in q
                for k in [
                    "not received",
                    "not recived",
                    "not delivered",
                    "where is my order",
                    "order not received",
                    "still not received",
                ]
            ):
                final = (
                    f"I checked order `{oid}` and confirmed the payment is successful and the order is currently in `shipped` state.\n\n"
                    f"- payment_status: `{pst}`\n"
                    f"- order_status: `{ost}`\n\n"
                    "This means the package is in transit. Please wait about 1 day for delivery before we trigger a delivery exception workflow.\n"
                    "If it is still not delivered after that, reply here and I'll escalate with courier-proof checks."
                    f"{tool_block}"
                )
            else:
                final = (
                    f"I checked your transaction for order `{oid}` and confirmed the payment/order state is consistent.\n\n"
                    f"- payment_status: `{pst}`\n"
                    f"- order_status: `{ost}`\n"
                    f"\nIf you still don't see an order confirmation, refresh your account/app view and check email/SMS for confirmation.\n"
                    f"{tool_block}"
                )
            return {
                "explanation": "Payment verified and the order is present in the system.",
                "actions_taken": actions_taken,
                "next_steps": "If you expected a refund, tell me whether you wanted cancellation or a return; otherwise confirm where you expected the confirmation (email/SMS/app).",
                "final_message": _sanitize_output_text(final),
            }

        # Core missing-order decision point.
        if pst == "paid":
            if refund_ok:
                final = (
                    f"I checked your transaction for order `{oid}`.\n\n"
                    f"- payment_status: `paid`\n"
                    f"- order_status: `{ost}` (order confirmation not found)\n\n"
                    "Because payment is marked complete but the order confirmation is missing, I initiated an auto-refund.\n"
                    f"Refund status in our system: `{refund_status or 'refund_requested'}`.\n"
                    "Refunds typically return within 5–7 business days (depends on your payment provider)."
                    f"{tool_block}"
                )
                return {
                    "explanation": "Tool check showed paid but order confirmation missing; refund was initiated.",
                    "actions_taken": actions_taken,
                    "next_steps": "If the refund doesn't appear after 7 business days, reply here for escalation.",
                    "final_message": _sanitize_output_text(final),
                }

            # Refund tool missing: be proactive but honest.
            final = (
                f"I checked your transaction for order `{oid}` and verified payment is marked `paid`, "
                f"but the order isn't confirmed yet (order_status=`{ost}`).\n\n"
                "Next step: the system will proceed on an auto-refund path if the order still doesn't appear. "
                "Please wait for reconciliation."
                f"{tool_block}"
            )
            return {
                "explanation": "Payment verified as paid, but refund initiation did not appear in tool results.",
                "actions_taken": actions_taken,
                "next_steps": "If the order confirmation still doesn't show up within 24 hours, reply here and I'll initiate the refund.",
                "final_message": _sanitize_output_text(final),
            }

        # Non-paid statuses: guide based on observed state.
        if refund_ok:
            final = (
                f"I checked your transaction for order `{oid}`.\n\n"
                f"- payment_status: `{pst}`\n"
                f"- order_status: `{ost}`\n\n"
                f"I initiated a refund in our system (refund_status=`{refund_status or 'refund_requested'}`).\n"
                "Refunds typically return within 5–7 business days (depends on your payment provider)."
                f"{tool_block}"
            )
            return {
                "explanation": "Tool results show a refund was initiated based on your refund request.",
                "actions_taken": actions_taken,
                "next_steps": "If the refund isn't visible after 7 business days, reply here and we will escalate.",
                "final_message": _sanitize_output_text(final),
            }

        final = (
            f"I checked your transaction for order `{oid}`.\n\n"
            f"- payment_status: `{pst}`\n"
            f"- order_status: `{ost}`\n"
            f"- age_hours: `{age}`\n\n"
            "If the order confirmation still doesn't appear, we follow policy-based refund/reconciliation steps."
            f"{tool_block}"
        )
        return {
            "explanation": "Used tool results to determine payment/order mismatch state.",
            "actions_taken": actions_taken,
            "next_steps": "Reply with any updated payment/order status you see.",
            "final_message": _sanitize_output_text(final),
        }

    if category == "account_issue" and account:
        acc_status = _lower(account.get("account_status"))

        if acc_status == "blocked":
            unblock_ok = bool(unblock and unblock.get("ok") is True)
            if unblock_ok:
                final = (
                    "I checked your account and it is currently blocked.\n\n"
                    "I initiated an unblock request. After a short propagation delay, try logging in again.\n"
                    f"{tool_block}"
                )
                return {
                    "explanation": "Account was blocked; unblock was initiated based on tool results.",
                    "actions_taken": actions_taken,
                    "next_steps": "Wait a few minutes and try logging in again.",
                    "final_message": _sanitize_output_text(final),
                }
            final = (
                "I checked your account and it is currently blocked.\n\n"
                "I attempted to unblock it, but it did not complete successfully.\n"
                "Next step: contact support with the time you tried to log in, and we'll resolve it manually."
                f"{tool_block}"
            )
            return {
                "explanation": "Account is blocked and unblock initiation failed or unavailable.",
                "actions_taken": actions_taken,
                "next_steps": "Contact support; include login attempt time.",
                "final_message": _sanitize_output_text(final),
            }

        final = (
            f"I checked your account and it looks active (account_status=`{acc_status or 'unknown'}`).\n\n"
            "If login still fails, share the exact error message and I'll help you troubleshoot."
            f"{tool_block}"
        )
        return {
            "explanation": "Account status looked active; provided troubleshooting next steps.",
            "actions_taken": actions_taken,
            "next_steps": "Share the exact login error text.",
            "final_message": _sanitize_output_text(final),
        }

    if category == "delivery_delay" and delivery_order:
        ost = _lower(delivery_order.get("order_status"))
        oid = str(delivery_order.get("order_id") or "")
        age = delivery_order.get("age_hours")

        if routing_decision == "ask_clarification" and clarification_question:
            final = (
                f"I checked your delivery record for order `{oid}`.\n\n"
                f"- order_status: `{ost}`\n"
                f"- age_hours: `{age}`\n\n"
                f"{clarification_question}"
                f"{tool_block}"
            )
            return {
                "explanation": "Delivery check completed, but one detail is required to proceed safely.",
                "actions_taken": actions_taken,
                "next_steps": f"Answer: {clarification_question}",
                "final_message": _sanitize_output_text(final),
            }

        if ost == "delivered":
            final = (
                f"I checked order `{oid}` and it is marked `delivered` in our system.\n\n"
                "If you didn't receive it, reply here and we'll request delivery-proof details (who received it / delivery location)."
                f"{tool_block}"
            )
            return {
                "explanation": "Order is marked delivered; provided next steps if user reports not receiving.",
                "actions_taken": actions_taken,
                "next_steps": "Share what happened at delivery.",
                "final_message": _sanitize_output_text(final),
            }

        final = (
            f"I checked order `{oid}` delivery status.\n\n"
            f"- order_status: `{ost}`\n"
            f"- age_hours: `{age}`\n\n"
            "No delivery exception is triggered yet. If the status doesn't improve, reply and I'll guide you on the next action."
            f"{tool_block}"
        )
        return {
            "explanation": "Used tool results to determine delivery status and next monitoring steps.",
            "actions_taken": actions_taken,
            "next_steps": "Reply with any courier updates you see.",
            "final_message": _sanitize_output_text(final),
        }

    if routing_decision == "ask_clarification" and clarification_question:
        final = (
            "I can help. I'm asking for this specific detail because I couldn't complete a safe automated check without it.\n\n"
            f"{clarification_question}"
            f"{tool_block}"
        )
        return {
            "explanation": evidence or "I found relevant support guidance based on similar past cases.",
            "actions_taken": actions_taken,
            "next_steps": f"Answer: {clarification_question}",
            "final_message": _sanitize_output_text(final),
        }

    if routing_decision == "escalate_admin":
        next_steps = "Next step: I'm escalating this to a support agent for manual review."
        checked_hint = tool_block if tool_block else "\n\nTool results: (none available)"
        final = (
            "I checked the available records and I can't safely complete the resolution automatically for this ticket.\n"
            f"{checked_hint}\n"
            "A support agent will review the case and confirm the correct next action."
        )
        return {
            "explanation": evidence or "I found some relevant policy details, but the confidence is not high enough for automation.",
            "actions_taken": actions_taken,
            "next_steps": next_steps,
            "final_message": _sanitize_output_text(final),
        }

    # auto_resolve
    policy_block = ""
    if resolution_steps:
        bullets = "\n".join([f"- {s}" for s in resolution_steps[:5]])
        policy_block = f"\n\nRecommended resolution steps:\n{bullets}"

    # tool_block already built above for ask/escalate and reused here.

    next_steps = "Next step: Tell me what outcome you expected (refund/cancellation/return or just confirmation) and I'll guide the correct next action."

    final = (
        f"Here's what I found for your request.\n\n"
        f"- Category: {category}\n"
        f"- Evidence matched: {evidence}\n"
        f"{tool_block}"
        f"{policy_block}\n\n"
        + "Actions taken:\n- "
        + "\n- ".join(actions_taken)
        + f"\n\n{next_steps}"
    )

    return {
        "explanation": evidence or "I used retrieved guidance from our support knowledge base and past tickets.",
        "actions_taken": actions_taken,
        "next_steps": next_steps,
        "final_message": _sanitize_output_text(final),
    }


def generate_response(
    query: str,
    category: str,
    routing_decision: str,
    retrieved_snippets: list[Json],
    tool_results: list[Json],
    clarification_question: str | None,
    final_confidence: float,
) -> Json:
    """
    Generates a structured response:
    - explanation
    - actions_taken
    - next_steps
    - final_message
    """
    gemini_prompt = (
        "You are an AI customer support agent.\n"
        "Generate a helpful response.\n\n"
        "Return ONLY valid JSON with keys:\n"
        "{\n"
        "  \"explanation\": <string>,\n"
        "  \"actions_taken\": <array of strings>,\n"
        "  \"next_steps\": <string>,\n"
        "  \"final_message\": <string>\n"
        "}\n\n"
        "Context:\n"
        f"- Query: {query}\n"
        f"- Intent category: {category}\n"
        f"- Routing decision: {routing_decision}\n"
        f"- Final confidence: {final_confidence:.2f}\n\n"
        f"Retrieved evidence snippets (array): {retrieved_snippets[:5]}\n\n"
        f"Tool results (array): {tool_results[:5]}\n\n"
        f"Clarification question (if any): {clarification_question}\n\n"
        "Requirements:\n"
        "- Be specific and grounded in the retrieved evidence/tool results.\n"
        "- If routing_decision is ask_clarification, focus on the missing info request.\n"
        "- If routing_decision is escalate_admin, tell user it will be reviewed by a human.\n"
    )

    # Deterministic fallback is higher quality for this demo.
    fallback = _fallback_response(
        query=query,
        category=category,
        routing_decision=routing_decision,
        retrieved_snippets=retrieved_snippets,
        tool_results=tool_results,
        clarification_question=clarification_question,
    )

    # If tool results exist, deterministic response is more reliable than LLM restyling.
    # This prevents generic outputs like "Follow the verified guidance above."
    if tool_results:
        fallback["final_message"] = _sanitize_output_text(str(fallback.get("final_message", "")))
        return fallback

    # Gemini is used only as an optional stylistic upgrader when no tool signal exists.
    gemini_resp = call_gemini_json(
        gemini_prompt,
        schema_hint="explanation, actions_taken, next_steps, final_message",
    )
    parsed = safe_json_loads(gemini_resp) if gemini_resp else None
    if (
        parsed
        and isinstance(parsed.get("final_message"), str)
        and parsed.get("final_message").strip()
        and len(parsed.get("final_message").strip()) >= 40
        and "Handled based on" not in parsed.get("final_message", "")
        and "Follow the verified guidance above" not in parsed.get("final_message", "")
    ):
        parsed["final_message"] = _sanitize_output_text(str(parsed.get("final_message", "")))
        return parsed

    fallback["final_message"] = _sanitize_output_text(str(fallback.get("final_message", "")))
    return fallback

