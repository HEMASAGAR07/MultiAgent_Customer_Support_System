from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TypedDict

from dateutil import parser as dt_parser
from langgraph.graph import END, StateGraph

from agents.classifier import ClassificationResult, IntentClassifier
from agents.planner import Planner
from agents.retrieval import Retriever
from agents.response_generator import generate_response
from memory.memory_store import MemoryStore
from tools.account_tools import get_user_account_status, unblock_account
from tools.order_tools import get_order_details, get_user_orders
from tools.payment_tools import check_payment_status, initiate_refund
from utils.confidence import compute_final_confidence, route_by_confidence
from utils.logger import Logger
from utils.vector_store import VectorStore


Json = dict[str, Any]
StreamCallback = Callable[[Json], None]

ORDER_ID_RE = re.compile(r"\b(ORD\d{5})\b")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_order_id(text: str) -> str | None:
    m = ORDER_ID_RE.search(text or "")
    return m.group(1) if m else None


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return dt_parser.parse(str(value))
    except Exception:
        return None


def _hours_since(dt_value: Any) -> float | None:
    dt_obj = _parse_dt(dt_value)
    if not dt_obj:
        return None
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - dt_obj).total_seconds() / 3600.0


class TicketGraphState(TypedDict, total=False):
    user_id: str
    query: str
    query_for_agents: str

    order_id_in_query: Optional[str]
    order_id: Optional[str]

    intent_category: str
    intent_confidence: float

    retrieved_snippets: list[Json]
    retrieval_cosine: float
    retrieval_similarity_unit: float

    final_conf: float
    routing_decision: str

    plan_type: str
    tool_calls: list[Json]

    tool_results: list[Json]
    effective_decision: str
    effective_conf: float
    clarification_question: Optional[str]

    final_response: Json
    final_message: str
    ticket: Json
    saved: Json


class LangGraphSupportActionAgent:
    """
    LangGraph-powered orchestrator.
    Keeps your existing modular components (classifier, RAG retriever, planner rules, JSON tools)
    but executes them as an explicit LangGraph node graph for clear agent architecture.
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        retriever: Retriever,
        planner: Planner,
        memory_store: MemoryStore,
        vector_store: VectorStore,
        vector_index_dir: str,
    ) -> None:
        self.classifier = classifier
        self.retriever = retriever
        self.planner = planner
        self.memory_store = memory_store
        self.vector_store = vector_store
        self.vector_index_dir = vector_index_dir

    def handle_ticket(self, user_id: str, query: str, stream_callback: StreamCallback | None = None) -> Json:
        logger = Logger(stream_callback=stream_callback)

        def _find_order_id_in_history(u_id: str, fallback: Optional[str]) -> Optional[str]:
            # If the user didn't provide an order id in the current message, reuse from last tickets.
            try:
                tickets = self.memory_store.get_user_tickets(u_id)
                # Search from newest to oldest.
                tickets_sorted = sorted(tickets, key=lambda x: str(x.get("created_at", "")), reverse=True)[:10]
                for t in tickets_sorted:
                    for msg in (t.get("messages") or []):
                        if msg.get("role") != "user":
                            continue
                        oid = _extract_order_id(str(msg.get("content", "")))
                        if oid:
                            return oid
            except Exception:
                pass
            return fallback

        def prepare_context(state: TicketGraphState) -> TicketGraphState:
            logger.step("prepare_context", input_data={"user_id": state["user_id"]}, streaming_msg="Understanding request and checking context...")

            order_id_in_query = _extract_order_id(state["query"])
            order_id = _find_order_id_in_history(state["user_id"], order_id_in_query)
            query_for_agents = state["query"]
            if order_id and not order_id_in_query:
                # Inject so classifier/tools can use it without asking user.
                query_for_agents = f"{state['query']}\nOrder: {order_id}"

            return {
                **state,
                "order_id_in_query": order_id_in_query,
                "order_id": order_id,
                "query_for_agents": query_for_agents,
            }

        def classify_intent(state: TicketGraphState) -> TicketGraphState:
            logger.step("classify_intent", input_data={"query": state["query_for_agents"]}, streaming_msg="Detected intent and confidence...")
            classification: ClassificationResult = self.classifier.classify(state["query_for_agents"], logger=logger)
            return {
                **state,
                "intent_category": classification.category,
                "intent_confidence": float(classification.confidence),
            }

        def retrieve_rag(state: TicketGraphState) -> TicketGraphState:
            logger.step("retrieval_start", input_data={"query": state["query_for_agents"]}, streaming_msg="Retrieving relevant knowledge and similar cases...")
            retrieved = self.retriever.retrieve(state["query_for_agents"], top_k=5)
            retrieval_cosine = float(retrieved[0].score) if retrieved else -0.2
            retrieval_similarity_unit = (retrieval_cosine + 1.0) / 2.0
            retrieved_snippets = [
                {"category": r.category, "score": r.score, "metadata": r.metadata, "text": r.text}
                for r in retrieved
            ]
            logger.step(
                "retrieval_results",
                input_data={"top_k": 5},
                output_data={"top_results": [{"category": r.category, "score": r.score} for r in retrieved]},
                confidence=retrieval_similarity_unit,
                streaming_msg=f"Found {len(retrieved)} similar cases (best similarity: {retrieval_similarity_unit:.2f})",
            )
            return {
                **state,
                "retrieved_snippets": retrieved_snippets,
                "retrieval_cosine": retrieval_cosine,
                "retrieval_similarity_unit": retrieval_similarity_unit,
            }

        def route_confidence(state: TicketGraphState) -> TicketGraphState:
            intent_score = state["intent_confidence"]
            retrieval_score = state["retrieval_cosine"]
            final_conf = compute_final_confidence(intent_score, retrieval_score)
            routing = route_by_confidence(final_conf)
            logger.step(
                "confidence_routing",
                input_data={"intent": intent_score, "retrieval_cosine": retrieval_score},
                output_data={"final_confidence": final_conf, "decision": routing.decision},
                confidence=final_conf,
                streaming_msg=f"Confidence routing: {routing.decision.replace('_', ' ')} (confidence: {final_conf:.2f})",
            )
            return {**state, "final_conf": final_conf, "routing_decision": routing.decision}

        def plan_tools(state: TicketGraphState) -> TicketGraphState:
            logger.step("plan_tools", streaming_msg="Planning tool usage using deterministic rules...")
            plan = self.planner.plan(
                category=state["intent_category"],
                query=state["query_for_agents"],
                user_id=state["user_id"],
                routing=type("R", (), {"decision": state["routing_decision"]})(),  # minimal adapter
                retrieved_categories=[r.get("category") for r in (state["retrieved_snippets"] or [])[:3]],
                logger=logger,
            )

            tool_calls: list[Json] = []
            if plan.plan_type == "auto_tool":
                for tc in plan.tool_calls:
                    tool_calls.append({"tool_name": tc.tool_name, "args": tc.args})

            return {**state, "plan_type": plan.plan_type, "tool_calls": tool_calls}

        def execute_plan_tools(state: TicketGraphState) -> TicketGraphState:
            tool_results: list[Json] = []
            logger.step("execute_tools", streaming_msg="Executing planned tool calls...")

            def _exec_one(tool_name: str, args: dict[str, Any]) -> Json:
                logger.step(
                    "tool_call",
                    input_data=args,
                    output_data=None,
                    tool_name=tool_name,
                    streaming_msg=f"Checking via tool: {tool_name}...",
                )
                if tool_name == "check_payment_status":
                    return check_payment_status(args.get("order_id"), stream_logger=logger, memory_store=self.memory_store)
                if tool_name == "get_order_details":
                    return get_order_details(args.get("order_id"), stream_logger=logger)
                if tool_name == "get_user_orders":
                    return get_user_orders(args.get("user_id"), stream_logger=logger)
                if tool_name == "get_user_account_status":
                    return get_user_account_status(args.get("user_id"), stream_logger=logger)
                if tool_name == "unblock_account":
                    return unblock_account(args.get("user_id"), stream_logger=logger)
                if tool_name == "initiate_refund":
                    return initiate_refund(args.get("order_id"), reason=args.get("reason") or "policy refund", stream_logger=logger)
                return {"ok": False, "error": f"Unknown tool {tool_name}"}

            for call in state.get("tool_calls") or []:
                tool_name = call.get("tool_name")
                args = call.get("args") or {}
                res = _exec_one(str(tool_name), args)
                tool_results.append({"tool_name": str(tool_name), "result": res})
                logger.step("tool_result", input_data=args, output_data=res, tool_name=str(tool_name), streaming_msg=f"{tool_name} completed.")

            return {**state, "tool_results": tool_results}

        def post_checks(state: TicketGraphState) -> TicketGraphState:
            """
            Mandatory deterministic checks:
            - account_issue -> check user status (and unblock if blocked)
            - payment_issue -> check orders/transactions (and refund if policy reached)
            - delivery_delay -> check order status
            """
            logger.step("post_checks", streaming_msg="Applying mandatory DB/tool checks before responding...")
            tool_results = state.get("tool_results") or []

            def _tool_result(name: str) -> Optional[Json]:
                for tr in tool_results:
                    if tr.get("tool_name") == name:
                        return tr.get("result")
                return None

            intent_category = state["intent_category"]
            effective_decision = state["routing_decision"]
            effective_conf = state["final_conf"]
            clarification_question: Optional[str] = None

            # ACCOUNT
            if intent_category == "account_issue":
                logger.step("detected_login_issue", streaming_msg="Detected login/account issue: verifying account status...")
                acc = _tool_result("get_user_account_status") or get_user_account_status(state["user_id"], stream_logger=logger)
                if not acc.get("ok"):
                    effective_decision = "escalate_admin"
                    effective_conf = 0.35
                else:
                    acc_status = str(acc.get("account_status", "")).lower()
                    if acc_status == "blocked":
                        logger.step("account_blocked", streaming_msg="Account is blocked: initiating unblock request...")
                        unblock_res = unblock_account(state["user_id"], stream_logger=logger)
                        tool_results.append({"tool_name": "unblock_account", "result": unblock_res})
                        effective_decision = "auto_resolve"
                        effective_conf = max(effective_conf, 0.75)
                    else:
                        logger.step("account_ok", streaming_msg=f"Account status: {acc_status or 'active/unknown'}. No unblock needed.")
                        effective_decision = "auto_resolve"
                        effective_conf = max(effective_conf, 0.65)

            # PAYMENT
            if intent_category == "payment_issue":
                logger.step("payment_check", streaming_msg="Detected payment issue: checking orders and transactions...")

                order_id = state.get("order_id")
                q = (state.get("query_for_agents") or "").lower()
                user_mentions_refund = any(
                    k in q
                    for k in [
                        "refund",
                        "not refunded",
                        "didn't get refund",
                        "didnt get refund",
                        "money debited but not refunded",
                        "money deducted but not refunded",
                        "charged back",
                        "amount not returned",
                        "not returned",
                        "i need a refund",
                    ]
                )
                candidate_order_ids: list[str] = []
                if order_id:
                    candidate_order_ids = [order_id]
                else:
                    orders_res = _tool_result("get_user_orders") or get_user_orders(state["user_id"], stream_logger=logger)
                    for o in orders_res.get("orders") or []:
                        if o.get("order_id"):
                            candidate_order_ids.append(str(o["order_id"]))

                candidate_order_ids = candidate_order_ids[:3]
                if not candidate_order_ids:
                    effective_decision = "ask_clarification"
                    effective_conf = 0.5
                    clarification_question = (
                        "I couldn't find matching orders for your message in the system. "
                        "I need your order id (format `ORD00012`) so I can check the payment record and decide whether a refund is needed."
                    )
                else:
                    checks: list[Json] = []
                    for oid in candidate_order_ids:
                        res = check_payment_status(oid, stream_logger=logger, memory_store=self.memory_store)
                        tool_results.append({"tool_name": "check_payment_status", "result": res})
                        checks.append(res)

                    # Pick best candidate check by “missing order” signals.
                    def _score(r: Json) -> float:
                        if not r.get("ok"):
                            return -1.0
                        pst = str(r.get("payment_status") or "").lower()
                        ost = str(r.get("order_status") or "").lower()
                        age = r.get("age_hours")
                        s = 0.0
                        if pst in {"deducted", "pending", "failed"}:
                            s += 0.6
                        if ost not in {"confirmed", "shipped", "delivered"}:
                            s += 0.3
                        if isinstance(age, (int, float)) and age >= 24.0:
                            s += 0.5
                        return s

                    best = max(checks, key=_score) if checks else {}
                    if best.get("ok") is True:
                        best_oid = str(best.get("order_id") or candidate_order_ids[0])
                        pst = str(best.get("payment_status") or "").lower()
                        ost = str(best.get("order_status") or "").lower()
                        age = best.get("age_hours")

                        order_confirmed = ost in {"confirmed", "shipped", "delivered"}

                        # Payment verified, but order confirmation is missing:
                        # This is the "Money deducted but no order confirmation" case.
                        if not order_confirmed and pst in {"paid", "deducted", "pending", "failed"}:
                            logger.step(
                                "payment_verified_but_order_missing",
                                streaming_msg=f"Verified payment={pst}, but order_status={ost} (no order confirmation). Taking resolution action...",
                            )

                            # If payment was fully successful (`paid`), trigger refund immediately in this demo.
                            # Otherwise, wait until policy window (24h) before refund.
                            should_refund = False
                            reason = "Order was not confirmed after payment."
                            if pst == "paid":
                                should_refund = True
                                reason = "Payment succeeded but order confirmation is missing (refund requested)."
                            elif isinstance(age, (int, float)) and age >= 24.0:
                                should_refund = True
                                reason = "Order not confirmed after policy window (refund requested)."

                            if should_refund:
                                logger.step("policy_refund", streaming_msg="Initiating refund based on verified payment/order mismatch...")
                                refund_res = initiate_refund(best_oid, reason=reason, stream_logger=logger)
                                tool_results.append({"tool_name": "initiate_refund", "result": refund_res})
                                effective_decision = "auto_resolve"
                                effective_conf = 0.9
                            else:
                                wait = max(0.0, 24.0 - float(age or 0.0))
                                logger.step(
                                    "policy_wait",
                                    streaming_msg=f"Payment verified but order confirmation is pending. Waiting window remaining: ~{wait:.1f} hours, then refund if still missing.",
                                )
                                effective_decision = "auto_resolve"
                                effective_conf = max(effective_conf, 0.75)
                        else:
                            logger.step("order_exists", streaming_msg=f"Order/payment already in a valid state: order_status={ost}.")
                        # Contradiction handling:
                        # If the system says the order is confirmed, but the user explicitly expects a refund,
                        # we should correct the expectation (no neutral "follow guidance" response).
                        if user_mentions_refund and ost == "confirmed":
                            effective_decision = "auto_resolve"
                            effective_conf = max(effective_conf, 0.88)
                        else:
                            effective_decision = "auto_resolve"
                            effective_conf = max(effective_conf, 0.65)

            # REFUND (same state-validation, but for explicit refund intent category)
            if intent_category == "refund":
                logger.step("refund_check", streaming_msg="Detected refund issue: checking payment + order state...")
                # For refund intent, the user expects a refund, but we still validate against system truth.
                q = (state.get("query_for_agents") or "").lower()
                user_mentions_refund = True if q else True

                order_id = state.get("order_id")
                candidate_order_ids: list[str] = []
                if order_id:
                    candidate_order_ids = [order_id]
                else:
                    orders_res = _tool_result("get_user_orders") or get_user_orders(state["user_id"], stream_logger=logger)
                    for o in orders_res.get("orders") or []:
                        if o.get("order_id"):
                            candidate_order_ids.append(str(o["order_id"]))

                candidate_order_ids = candidate_order_ids[:3]
                if not candidate_order_ids:
                    effective_decision = "ask_clarification"
                    effective_conf = 0.5
                    clarification_question = (
                        "I couldn't find matching orders for your refund request. "
                        "Please share your order id (format `ORD00012`) so I can verify the payment state."
                    )
                else:
                    checks: list[Json] = []
                    for oid in candidate_order_ids:
                        res = check_payment_status(oid, stream_logger=logger, memory_store=self.memory_store)
                        tool_results.append({"tool_name": "check_payment_status", "result": res})
                        checks.append(res)

                    def _score(r: Json) -> float:
                        if not r.get("ok"):
                            return -1.0
                        pst = str(r.get("payment_status") or "").lower()
                        ost = str(r.get("order_status") or "").lower()
                        age = r.get("age_hours")
                        s = 0.0
                        if pst in {"refunded"}:
                            s += 1.0
                        if pst in {"paid", "deducted", "pending", "failed"}:
                            s += 0.6
                        if ost == "confirmed":
                            s += 0.4
                        if isinstance(age, (int, float)) and age >= 24.0:
                            s += 0.2
                        return s

                    best = max(checks, key=_score) if checks else {}
                    if best.get("ok") is True:
                        best_oid = str(best.get("order_id") or candidate_order_ids[0])
                        pst = str(best.get("payment_status") or "").lower()
                        ost = str(best.get("order_status") or "").lower()
                        age = best.get("age_hours")

                        # Contradiction handling:
                        # If the order is already confirmed/placed, refund isn't applicable for this issue.
                        if ost == "confirmed" and user_mentions_refund:
                            logger.step(
                                "refund_not_applicable_contradiction",
                                streaming_msg="System shows order confirmed, but user expects refund → correcting expectation."
                            )
                            effective_decision = "auto_resolve"
                            effective_conf = max(effective_conf, 0.92)
                        else:
                            # Refund path: if the payment isn't already refunded, initiate refund.
                            if pst != "refunded":
                                logger.step(
                                    "refund_initiating_from_intent",
                                    streaming_msg="Refund intent validated; initiating refund in mock system..."
                                )
                                refund_res = initiate_refund(
                                    best_oid,
                                    reason="Refund requested; verified payment/order state requires refund.",
                                    stream_logger=logger,
                                )
                                tool_results.append({"tool_name": "initiate_refund", "result": refund_res})
                                effective_decision = "auto_resolve"
                                effective_conf = max(effective_conf, 0.86)
                            else:
                                logger.step("refund_already_done", streaming_msg="System shows payment already refunded.")
                                effective_decision = "auto_resolve"
                                effective_conf = max(effective_conf, 0.82)

            # DELIVERY
            if intent_category == "delivery_delay":
                logger.step("delivery_check", streaming_msg="Detected delivery delay: checking order status...")
                order_id = state.get("order_id")
                if not order_id:
                    orders_res = get_user_orders(state["user_id"], stream_logger=logger)
                    for o in orders_res.get("orders") or []:
                        if o.get("order_id"):
                            order_id = str(o["order_id"])
                            break

                if not order_id:
                    effective_decision = "ask_clarification"
                    effective_conf = 0.5
                    clarification_question = (
                        "I couldn't find matching orders for your message in the system. "
                        "Please share your order id (format `ORD00012`) so I can check the delivery status and (if needed) request delivery-proof details."
                    )
                else:
                    order_res = get_order_details(order_id, stream_logger=logger)
                    tool_results.append({"tool_name": "get_order_details", "result": order_res})
                    ost = str(order_res.get("order_status") or "").lower()
                    age = order_res.get("age_hours")
                    if ost == "delivered":
                        # If the user claims they didn't receive the package, we need clarification
                        # (delivery proof / recipient confirmation) even though the order is marked delivered.
                        q = (state.get("query_for_agents") or "").lower()
                        # If the user reports any mismatch with delivery, ask for proof/clarification.
                        user_disagrees = any(
                            s in q
                            for s in [
                                "not received",
                                "didn't receive",
                                "didnt receive",
                                "still not",
                                "not delivered",
                                "haven't received",
                                "wrong item",
                                "wrong",
                                "different",
                                "diffirent",
                                "mismatch",
                                "not the same",
                            ]
                        )
                        if user_disagrees:
                            effective_decision = "ask_clarification"
                            effective_conf = 0.55
                            # Tailor the question based on what the user complained about.
                            if any(s in q for s in ["wrong item", "wrong", "different", "diffirent", "mismatch", "not the same"]):
                                clarification_question = (
                                    "Our system shows the order as DELIVERED, but your message suggests a delivery mismatch (wrong/different item). "
                                    "Please tell me: which item you received vs which item you ordered (and if possible, share item names/variants or a photo). "
                                    "Once I have that, I'll guide the correct resolution path."
                                )
                            else:
                                clarification_question = (
                                    "Our system shows the order as DELIVERED, but you report you didn't receive it. "
                                    "Please confirm: was it received by someone else at your address (family/guard/neighbor), or was it left at a mailbox/doorstep? "
                                    "Tell me what happened and I'll guide the next step."
                                )
                            logger.step("delivery_proof_needed", streaming_msg="System marks delivered, but user reports not received → asking for delivery proof details...")
                        else:
                            effective_decision = "auto_resolve"
                            effective_conf = max(effective_conf, 0.7)
                    elif isinstance(age, (int, float)) and age >= 48.0:
                        logger.step("delivery_overdue", streaming_msg="Delivery looks overdue (~48h). Escalation recommended...")
                        effective_decision = "auto_resolve"
                        effective_conf = max(effective_conf, 0.75)
                    else:
                        logger.step("delivery_pending", streaming_msg="Delivery checks complete. Monitoring next courier update...")
                        effective_decision = "auto_resolve"
                        effective_conf = max(effective_conf, 0.65)

            return {
                **state,
                "tool_results": tool_results,
                "effective_decision": effective_decision,
                "effective_conf": float(effective_conf),
                "clarification_question": clarification_question,
            }

        def generate_and_finalize(state: TicketGraphState) -> TicketGraphState:
            def _tool_result(name: str) -> Optional[Json]:
                for tr in (state.get("tool_results") or []):
                    if tr.get("tool_name") == name and isinstance(tr.get("result"), dict):
                        return tr.get("result")
                return None

            def _clean_message(msg: str) -> str:
                text = str(msg or "")
                # Remove legacy/generic content that is not useful to users.
                text = text.replace(
                    "Next step: Follow the verified guidance above. If anything changes, reply with your updated order/payment status.",
                    "Next step: Reply with your latest status and I'll guide the exact next action.",
                )
                if "Evidence matched: Type: past_ticket" in text:
                    text = re.sub(
                        r"Evidence matched:\s*Type:\s*past_ticket.*?(?=\nTool results|\n\nTool results|$)",
                        "Evidence matched: relevant policy and similar case context.",
                        text,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                return text.strip()

            final_response = generate_response(
                query=state["query_for_agents"],
                category=state["intent_category"],
                routing_decision=state["effective_decision"],
                retrieved_snippets=state.get("retrieved_snippets") or [],
                tool_results=state.get("tool_results") or [],
                clarification_question=state.get("clarification_question"),
                final_confidence=float(state["effective_conf"]),
            )
            final_message = _clean_message(str(final_response.get("final_message", "")))

            # Hard deterministic guardrail at orchestration level.
            # If a generic response slips through, rebuild a useful answer from verified tool data.
            generic_markers = [
                "Here's what I found for your request.",
                "Follow the verified guidance above",
                "Category:",
                "Evidence matched:",
            ]
            looks_generic = any(m in final_message for m in generic_markers)
            if looks_generic:
                category = state["intent_category"]
                query_l = str(state.get("query_for_agents") or "").lower()
                pay = _tool_result("check_payment_status")
                acc = _tool_result("get_user_account_status")

                if category == "account_issue" and acc and acc.get("ok"):
                    acc_status = str(acc.get("account_status") or "unknown").lower()
                    if acc_status == "active":
                        final_message = (
                            "I checked your account status in the system and it is currently ACTIVE.\n\n"
                            "Since the account is not blocked, this issue is likely due to credentials/session mismatch.\n"
                            "Please retry login once, and if it still fails, share the exact error message so I can troubleshoot the next step precisely."
                        )
                    elif acc_status == "blocked":
                        final_message = (
                            "I checked your account status and it is BLOCKED.\n\n"
                            "I have initiated the unblock flow. Please wait a few minutes and try logging in again."
                        )

                elif category in {"payment_issue", "refund"} and pay and pay.get("ok"):
                    oid = str(pay.get("order_id") or "")
                    pst = str(pay.get("payment_status") or "").lower()
                    ost = str(pay.get("order_status") or "").lower()
                    if ost == "shipped" and any(k in query_l for k in ["not received", "not recived", "order not received", "not delivered"]):
                        final_message = (
                            f"I checked order `{oid}` and it is currently in `shipped` state with payment `{pst}`.\n\n"
                            "This means the package is in transit. Please wait about 1 day for delivery.\n"
                            "If it is still not delivered after 1 day, reply here and I'll raise a delivery exception."
                        )
                    elif ost == "confirmed" and any(k in query_l for k in ["refund", "not refunded", "money debited", "money deducted"]):
                        final_message = (
                            f"I checked order `{oid}` and confirmed payment is successful and the order is `confirmed`.\n\n"
                            "A refund is not applicable for this state. "
                            "If you meant cancellation or return, tell me and I'll initiate the correct flow."
                        )
                    else:
                        final_message = (
                            f"I verified your order `{oid}` with payment status `{pst}` and order status `{ost}`.\n\n"
                            "Based on this verified state, I'll continue with the most appropriate next action. "
                            "Reply with any new update you see."
                        )

            final_response["final_message"] = final_message

            logger.step("final_response", streaming_msg="Final response generated.")

            status = (
                "resolved"
                if state["effective_decision"] == "auto_resolve"
                else "needs_clarification"
                if state["effective_decision"] == "ask_clarification"
                else "escalated"
            )

            ticket: Json = {
                "ticket_id": None,
                "user_id": state["user_id"],
                "query": state["query_for_agents"],
                "category": state["intent_category"],
                "priority": "medium",
                "status": status,
                "created_at": _now_iso(),
                "resolution": final_message,
                "explanation": final_response.get("explanation", ""),
                "actions_taken": final_response.get("actions_taken", []),
                "next_steps": final_response.get("next_steps", ""),
                "final_decision": state["effective_decision"],
                "intent_confidence": state["intent_confidence"],
                "retrieval_similarity_unit": state.get("retrieval_similarity_unit", 0.0),
                "final_confidence": float(state["effective_conf"]),
                "confidence_thresholds": {"auto": 0.65, "clarify": 0.4},
                "tool_called": len(state.get("tool_results") or []) > 0,
                "tool_calls": state.get("tool_calls") or [],
                "agent_trace": logger.to_trace(),
                "messages": [
                    {"role": "user", "content": state["query_for_agents"], "created_at": _now_iso()},
                    {"role": "assistant", "content": final_message, "created_at": _now_iso()},
                ],
            }

            saved = self.memory_store.add_ticket(ticket)

            # Best-effort: add to vector store for future RAG.
            try:
                doc_text = (
                    f"Type: past_ticket\nCategory: {state['intent_category']}\n"
                    f"Status: {status}\nCreatedAt: {saved.get('created_at')}\nQuery: {state['query_for_agents']}\n"
                    f"Resolution: {final_message}\n"
                )
                self.vector_store.add_document(
                    text=doc_text,
                    metadata={"source": "past_ticket", "category": state["intent_category"], "ticket_id": saved.get("ticket_id"), "status": status},
                    index_dir=self.vector_index_dir,
                )
            except Exception:
                pass

            logger.step("done", output_data={"ticket_id": saved.get("ticket_id"), "status": status}, streaming_msg=f"Ticket completed: {saved.get('ticket_id')} ({status})")
            return {**state, "final_response": final_response, "final_message": final_message, "ticket": ticket, "saved": saved}

        # Build LangGraph.
        graph = StateGraph(TicketGraphState)
        graph.add_node("prepare_context", prepare_context)
        graph.add_node("classify_intent", classify_intent)
        graph.add_node("retrieve_rag", retrieve_rag)
        graph.add_node("route_confidence", route_confidence)
        graph.add_node("plan_tools", plan_tools)
        graph.add_node("execute_plan_tools", execute_plan_tools)
        graph.add_node("post_checks", post_checks)
        graph.add_node("generate_and_finalize", generate_and_finalize)

        graph.set_entry_point("prepare_context")
        graph.add_edge("prepare_context", "classify_intent")
        graph.add_edge("classify_intent", "retrieve_rag")
        graph.add_edge("retrieve_rag", "route_confidence")
        graph.add_edge("route_confidence", "plan_tools")

        def _cond(state: TicketGraphState) -> str:
            return "execute_plan_tools" if state.get("plan_type") == "auto_tool" else "post_checks"

        graph.add_conditional_edges("plan_tools", _cond, {"execute_plan_tools": "execute_plan_tools", "post_checks": "post_checks"})
        graph.add_edge("execute_plan_tools", "post_checks")
        graph.add_edge("post_checks", "generate_and_finalize")
        graph.add_edge("generate_and_finalize", END)

        app = graph.compile()

        init_state: TicketGraphState = {
            "user_id": user_id,
            "query": query,
            "query_for_agents": query,
        }

        result = app.invoke(init_state)
        return result["saved"]

