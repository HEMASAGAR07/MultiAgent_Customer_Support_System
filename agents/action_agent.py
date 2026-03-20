from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from dateutil import parser as dt_parser

from agents.classifier import IntentClassifier, ClassificationResult
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


def _parse_dt(value: Any) -> datetime | None:
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


class SupportActionAgent:
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

        def _find_order_id_in_history(u_id: str, fallback: str | None) -> str | None:
            # If the user didn't provide an order_id in the current message, look in their last tickets/messages.
            try:
                tickets = self.memory_store.get_user_tickets(u_id)
                for t in sorted(tickets, key=lambda x: str(x.get("created_at", "")), reverse=True)[:8]:
                    for msg in (t.get("messages") or []):
                        if msg.get("role") != "user":
                            continue
                        oid = _extract_order_id(str(msg.get("content", "")))
                        if oid:
                            return oid
            except Exception:
                pass
            return fallback

        logger.step(
            "start",
            input_data={"user_id": user_id, "query": query},
            streaming_msg="Understanding your request and checking records...",
        )

        # Context awareness (order id reuse)
        order_id_in_query = _extract_order_id(query)
        order_id = _find_order_id_in_history(user_id, order_id_in_query)

        query_for_agents = query
        if order_id and not order_id_in_query:
            # Feed the order id into the agent pipeline so it can use tools without asking again.
            query_for_agents = f"{query}\nOrder: {order_id}"

        # 1) Intent classification
        classification: ClassificationResult = self.classifier.classify(query_for_agents, logger=logger)
        intent_category = classification.category
        intent_confidence = float(classification.confidence)

        # 2) Retrieval (RAG)
        logger.step("retrieval_start", input_data={"query": query_for_agents}, streaming_msg="Retrieving relevant knowledge and similar cases...")
        retrieved = self.retriever.retrieve(query_for_agents, top_k=5)
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

        logger.step(
            "retrieved_documents",
            input_data={"query": query_for_agents},
            output_data={"retrieved_snippets": retrieved_snippets},
            streaming_msg="Retrieved documents ready for planning.",
        )

        # 3) Confidence routing (badge/initial decision only)
        final_conf = compute_final_confidence(intent_confidence, retrieval_cosine)
        routing = route_by_confidence(final_conf)

        logger.step(
            "confidence_routing",
            input_data={"intent_category": intent_category, "intent_confidence": intent_confidence, "retrieval_cosine": retrieval_cosine},
            output_data={"final_confidence": final_conf, "decision": routing.decision},
            confidence=final_conf,
            streaming_msg=f"Confidence routing: {routing.decision.replace('_', ' ')} (confidence: {final_conf:.2f})",
            meta={"intent_confidence": intent_confidence, "retrieval_similarity_unit": retrieval_similarity_unit},
        )

        # 4) Rule-based planner (tool-first)
        logger.step(
            "planning_start",
            streaming_msg="Planning the next actions with a deterministic rule engine...",
        )

        plan = self.planner.plan(
            category=intent_category,
            query=query_for_agents,
            user_id=user_id,
            routing=routing,
            retrieved_categories=[r.category for r in retrieved[:3]],
            logger=logger,
        )

        tool_results: list[Json] = []
        tool_calls: list[Json] = []

        def _exec_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
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

        if plan.plan_type == "auto_tool":
            for tc in plan.tool_calls:
                res = _exec_tool(tc.tool_name, tc.args)
                tool_calls.append({"tool_name": tc.tool_name, "args": tc.args})
                tool_results.append({"tool_name": tc.tool_name, "result": res})
                logger.step(
                    "tool_result",
                    input_data=tc.args,
                    output_data=res,
                    tool_name=tc.tool_name,
                    streaming_msg=f"{tc.tool_name} completed.",
                )

        # 5) Execute additional mandatory checks based on intent and tool outputs
        clarification_question: str | None = None
        effective_decision = routing.decision
        effective_conf = final_conf
        action_taken_summary: list[str] = []

        def _extract_tool_result(name: str) -> Json | None:
            for tr in tool_results:
                if tr.get("tool_name") == name:
                    return tr.get("result")
            return None

        # ACCOUNT ISSUE: always check users DB and unblock if needed.
        if intent_category == "account_issue":
            logger.step("account_issue_check", streaming_msg="Detected login/account issue: verifying account status...")
            acc = _extract_tool_result("get_user_account_status") or {}
            acc_status = str(acc.get("account_status", "")).lower()
            if acc_status == "blocked":
                logger.step("account_blocked", streaming_msg="Account is blocked: initiating unblock request...")
                unblock_res = _exec_tool("unblock_account", {"user_id": user_id})
                tool_calls.append({"tool_name": "unblock_account", "args": {"user_id": user_id}})
                tool_results.append({"tool_name": "unblock_account", "result": unblock_res})
                action_taken_summary.append("unblock_account")
                effective_decision = "auto_resolve"
                effective_conf = max(effective_conf, 0.8)
            else:
                logger.step("account_status_ok", streaming_msg=f"Account status looks good (status: {acc_status or 'unknown'}).")
                effective_decision = "auto_resolve"
                effective_conf = max(effective_conf, 0.65)

        # PAYMENT / REFUND: always check transactions/orders before responding.
        if intent_category in {"payment_issue", "refund"}:
            logger.step("payment_issue_check", streaming_msg="Detected payment-related issue: checking orders/transactions before answering...")

            # Candidate orders:
            candidate_order_ids: list[str] = []
            if order_id:
                candidate_order_ids = [order_id]
            else:
                user_orders_res = _extract_tool_result("get_user_orders") or {}
                for o in user_orders_res.get("orders") or []:
                    if o.get("order_id"):
                        candidate_order_ids.append(str(o["order_id"]))

            # If the planner didn't fetch orders and we don't have an order id, we must still fetch.
            if not candidate_order_ids and not order_id:
                logger.step("payment_missing_order_fetch", streaming_msg="No order id in message: fetching your orders first...")
                user_orders_res = _exec_tool("get_user_orders", {"user_id": user_id})
                tool_calls.append({"tool_name": "get_user_orders", "args": {"user_id": user_id}})
                tool_results.append({"tool_name": "get_user_orders", "result": user_orders_res})
                for o in user_orders_res.get("orders") or []:
                    if o.get("order_id"):
                        candidate_order_ids.append(str(o["order_id"]))

            # Pick best candidates for payment checks.
            # If we have user_orders payload, filter for "not yet confirmed" states.
            order_candidates_with_meta: list[tuple[str, str | None, str | None, str | None]] = []
            if not order_id:
                user_orders_res = _extract_tool_result("get_user_orders") or {}
                for o in user_orders_res.get("orders") or []:
                    oid = o.get("order_id")
                    if not oid:
                        continue
                    order_candidates_with_meta.append(
                        (
                            str(oid),
                            str(o.get("order_status") or ""),
                            str(o.get("payment_status") or ""),
                            str(o.get("created_at") or ""),
                        )
                    )
            else:
                order_candidates_with_meta = [(order_id, None, None, None)]

            preferred: list[str] = []
            for oid, ost, pst, _created in order_candidates_with_meta:
                ost_l = str(ost or "").lower()
                pst_l = str(pst or "").lower()
                if pst_l in {"deducted", "pending", "failed"} and ost_l not in {"confirmed", "shipped", "delivered"}:
                    preferred.append(oid)
            candidate_order_ids = preferred or candidate_order_ids
            candidate_order_ids = candidate_order_ids[:3]

            if not candidate_order_ids:
                # No candidate orders exist in DB -> fallback to clarification/escalation.
                if routing.decision == "ask_clarification":
                    effective_decision = "ask_clarification"
                    clarification_question = "I couldn't find matching orders in the system. Please share your order id (format `ORD00012`) so I can look up payment/transactions."
                else:
                    effective_decision = "escalate_admin"
                effective_conf = min(0.35, effective_conf)
            else:
                # Check payment status for candidates.
                payment_checks: list[Json] = []
                for oid in candidate_order_ids:
                    res = _exec_tool("check_payment_status", {"order_id": oid})
                    tool_calls.append({"tool_name": "check_payment_status", "args": {"order_id": oid}})
                    tool_results.append({"tool_name": "check_payment_status", "result": res})
                    payment_checks.append(res)

                # Choose best check result (most relevant).
                def _score_check(r: Json) -> float:
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

                best_check = max(payment_checks, key=_score_check) if payment_checks else {}
                if best_check and best_check.get("ok") is True:
                    best_oid = str(best_check.get("order_id") or candidate_order_ids[0])
                    payment_status = str(best_check.get("payment_status") or "").lower()
                    order_status = str(best_check.get("order_status") or "").lower()
                    age_hours = best_check.get("age_hours")

                    # Decide refund vs wait vs confirm.
                    if payment_status in {"deducted", "pending", "failed"} and order_status not in {"confirmed", "shipped", "delivered"}:
                        if isinstance(age_hours, (int, float)) and age_hours >= 24.0:
                            logger.step("payment_refund_initiate", streaming_msg="Payment indicates missing order after policy window: initiating refund...")
                            reason = "Order not confirmed after payment window (policy refund)"
                            refund_res = _exec_tool("initiate_refund", {"order_id": best_oid, "reason": reason})
                            tool_calls.append({"tool_name": "initiate_refund", "args": {"order_id": best_oid, "reason": reason}})
                            tool_results.append({"tool_name": "initiate_refund", "result": refund_res})
                            action_taken_summary.append("initiate_refund")
                            effective_decision = "auto_resolve"
                            effective_conf = max(effective_conf, 0.85)
                        else:
                            wait_for = max(0.0, 24.0 - float(age_hours or 0.0))
                            logger.step("payment_wait_window", streaming_msg=f"Best match looks pending/deducted. Waiting window remaining: ~{wait_for:.1f} hours.")
                            effective_decision = "auto_resolve"
                            effective_conf = max(effective_conf, 0.7)
                    elif order_status in {"confirmed", "shipped", "delivered"}:
                        logger.step("payment_order_exists", streaming_msg=f"Order status already exists: {order_status}. No refund needed.")
                        effective_decision = "auto_resolve"
                        effective_conf = max(effective_conf, 0.7)
                    else:
                        effective_decision = "auto_resolve"
                        effective_conf = max(effective_conf, 0.6)
                else:
                    effective_decision = routing.decision if routing.decision != "escalate_admin" else "ask_clarification"
                    effective_conf = min(0.45, effective_conf)

        # DELIVERY DELAY: always check order status before asking anything.
        if intent_category == "delivery_delay":
            logger.step("delivery_issue_check", streaming_msg="Detected delivery delay: checking your order status first...")
            candidate_order_ids: list[str] = []
            if order_id:
                candidate_order_ids = [order_id]
            else:
                user_orders_res = _extract_tool_result("get_user_orders") or {}
                for o in user_orders_res.get("orders") or []:
                    if o.get("order_id"):
                        candidate_order_ids.append(str(o["order_id"]))

            if not candidate_order_ids and not order_id:
                logger.step("delivery_missing_order_fetch", streaming_msg="No order id in message: fetching your orders first...")
                user_orders_res = _exec_tool("get_user_orders", {"user_id": user_id})
                tool_calls.append({"tool_name": "get_user_orders", "args": {"user_id": user_id}})
                tool_results.append({"tool_name": "get_user_orders", "result": user_orders_res})
                for o in user_orders_res.get("orders") or []:
                    if o.get("order_id"):
                        candidate_order_ids.append(str(o["order_id"]))

            # Filter for likely delayed orders (shipped/confirmed, not delivered/cancelled).
            preferred: list[str] = []
            if not order_id:
                user_orders_res = _extract_tool_result("get_user_orders") or {}
                for o in user_orders_res.get("orders") or []:
                    oid = o.get("order_id")
                    if not oid:
                        continue
                    ost_l = str(o.get("order_status") or "").lower()
                    if ost_l in {"shipped", "confirmed"}:
                        preferred.append(str(oid))
            candidate_order_ids = (preferred or candidate_order_ids)[:1]

            if not candidate_order_ids:
                if routing.decision == "ask_clarification":
                    effective_decision = "ask_clarification"
                    clarification_question = "I couldn't find matching orders in the system. Please share your order id so I can check delivery status."
                else:
                    effective_decision = "escalate_admin"
                effective_conf = min(0.35, effective_conf)
            else:
                oid = candidate_order_ids[0]
                order_res = _exec_tool("get_order_details", {"order_id": oid})
                tool_calls.append({"tool_name": "get_order_details", "args": {"order_id": oid}})
                tool_results.append({"tool_name": "get_order_details", "result": order_res})

                ost = str(order_res.get("order_status") or "").lower()
                age_hours = order_res.get("age_hours")

                if ost == "delivered":
                    logger.step("delivery_delivered", streaming_msg="Order already delivered according to system records.")
                    effective_decision = "auto_resolve"
                    effective_conf = max(effective_conf, 0.7)
                elif isinstance(age_hours, (int, float)) and age_hours >= 48.0:
                    logger.step("delivery_overdue", streaming_msg=f"Delivery looks overdue (~{float(age_hours):.1f}h). Requesting logistics escalation...")
                    effective_decision = "auto_resolve"
                    effective_conf = max(effective_conf, 0.75)
                else:
                    remaining = None
                    if isinstance(age_hours, (int, float)):
                        remaining = max(0.0, 48.0 - float(age_hours))
                    logger.step(
                        "delivery_pending",
                        streaming_msg=f"Delivery delay checks complete. Remaining time before escalation: {remaining:.1f}h" if remaining is not None else "Delivery delay checks complete.",
                    )
                    effective_decision = "auto_resolve"
                    effective_conf = max(effective_conf, 0.65)

        # Default: if we didn't override based on tools, use routing decision.
        effective_decision = effective_decision or routing.decision

        logger.step(
            "final_effective_decision",
            input_data={"routing_decision": routing.decision, "effective_decision": effective_decision},
            output_data={"effective_conf": effective_conf},
            streaming_msg=f"Final decision after checks: {effective_decision.replace('_', ' ')}",
        )

        # 6) Generate response (meaningful + actionable)
        final_response = generate_response(
            query=query,
            category=intent_category,
            routing_decision=effective_decision,
            retrieved_snippets=retrieved_snippets,
            tool_results=tool_results,
            clarification_question=clarification_question,
            final_confidence=effective_conf,
        )

        final_message = final_response["final_message"]

        logger.step(
            "response_generated",
            input_data={"routing_decision": effective_decision},
            output_data=final_response,
            streaming_msg="Final response generated.",
        )

        status = "resolved" if effective_decision == "auto_resolve" else "needs_clarification" if effective_decision == "ask_clarification" else "escalated"

        ticket: Json = {
            "ticket_id": None,
            "user_id": user_id,
            "query": query,
            "category": intent_category,
            "priority": "medium",
            "status": status,
            "created_at": _now_iso(),
            "resolution": final_message,
            "explanation": final_response.get("explanation", ""),
            "actions_taken": final_response.get("actions_taken", []),
            "next_steps": final_response.get("next_steps", ""),
            "final_decision": effective_decision,
            "intent_confidence": intent_confidence,
            "retrieval_similarity_unit": retrieval_similarity_unit,
            "final_confidence": effective_conf,
            "confidence_thresholds": {"auto": 0.65, "clarify": 0.4},
            "tool_called": len(tool_calls) > 0,
            "tool_calls": tool_calls,
            "agent_trace": logger.to_trace(),
            "messages": [
                {"role": "user", "content": query, "created_at": _now_iso()},
                {"role": "assistant", "content": final_message, "created_at": _now_iso()},
            ],
        }

        saved = self.memory_store.add_ticket(ticket)

        # 7) Update vector store (best effort)
        try:
            doc_text = f"Type: past_ticket\nCategory: {intent_category}\nStatus: {status}\nCreatedAt: {saved.get('created_at')}\nQuery: {query}\nResolution: {final_message}\n"
            self.vector_store.add_document(
                text=doc_text,
                metadata={
                    "source": "past_ticket",
                    "category": intent_category,
                    "ticket_id": saved.get("ticket_id"),
                    "status": status,
                },
                index_dir=self.vector_index_dir,
            )
        except Exception:
            pass

        logger.step(
            "done",
            output_data={"ticket_id": saved.get("ticket_id"), "status": status},
            streaming_msg=f"Ticket completed: {saved.get('ticket_id')} ({status})",
        )

        return saved

