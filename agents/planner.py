from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from utils.confidence import RoutingDecision


Json = dict[str, Any]


ORDER_ID_RE = re.compile(r"\b(ORD\d{5})\b")


def extract_order_id(text: str) -> str | None:
    m = ORDER_ID_RE.search(text or "")
    return m.group(1) if m else None


@dataclass
class ToolCall:
    tool_name: str
    args: Json


@dataclass
class Plan:
    plan_type: str  # auto_tool | auto_no_tool | ask_clarification | escalate_admin
    tool_calls: list[ToolCall]
    clarification_question: str | None = None


class Planner:
    """
    Plans what to do after confidence routing:
    - auto: may call tools depending on intent + missing order_id
    - clarify: ask for missing order_id
    - escalate: route to admin
    """

    def plan(
        self,
        category: str,
        query: str,
        user_id: str,
        routing: RoutingDecision,
        retrieved_categories: list[str],
        logger=None,
    ) -> Plan:
        """
        Lightweight rule engine:
        - Decide tools to call deterministically from (intent category, identifiers).
        - Never ask for order_id if we can look it up via get_user_orders(user_id).
        """
        order_id = extract_order_id(query)

        if logger:
            logger.step(
                "planning_routing",
                input_data={
                    "category": category,
                    "query": query,
                    "user_id": user_id,
                    "routing": routing.decision,
                    "order_id": order_id,
                    "retrieved_categories": retrieved_categories,
                },
                output_data=None,
                streaming_msg="Applying deterministic support rules before escalation...",
                meta={},
            )

        # Rules per intent.
        if category == "account_issue":
            if logger:
                logger.step(
                    "rule_engine_account_issue",
                    input_data={"user_id": user_id},
                    output_data=None,
                    streaming_msg="Detected login/account issue: checking user database...",
                    meta={},
                )
            return Plan(
                plan_type="auto_tool",
                tool_calls=[ToolCall(tool_name="get_user_account_status", args={"user_id": user_id})],
            )

        if category in {"payment_issue", "refund"}:
            if order_id:
                if logger:
                    logger.step(
                        "rule_engine_payment_with_order",
                        input_data={"order_id": order_id},
                        output_data=None,
                        streaming_msg="Payment issue detected: checking transactions for the order...",
                        meta={},
                    )
                if category == "payment_issue":
                    return Plan(
                        plan_type="auto_tool",
                        tool_calls=[ToolCall(tool_name="check_payment_status", args={"order_id": order_id})],
                    )
                # refund
                return Plan(
                    plan_type="auto_tool",
                    tool_calls=[
                        ToolCall(tool_name="get_order_details", args={"order_id": order_id}),
                        ToolCall(tool_name="check_payment_status", args={"order_id": order_id}),
                    ],
                )

            # No order id in text: look up candidate orders from DB instead of asking the user.
            if logger:
                logger.step(
                    "rule_engine_payment_without_order",
                    input_data={"user_id": user_id},
                    output_data=None,
                    streaming_msg="No order id provided: fetching your orders first...",
                    meta={},
                )
            return Plan(
                plan_type="auto_tool",
                tool_calls=[ToolCall(tool_name="get_user_orders", args={"user_id": user_id})],
            )

        if category == "delivery_delay":
            if order_id:
                if logger:
                    logger.step(
                        "rule_engine_delivery_with_order",
                        input_data={"order_id": order_id},
                        output_data=None,
                        streaming_msg="Delivery delay detected: checking order status...",
                        meta={},
                    )
                return Plan(
                    plan_type="auto_tool",
                    tool_calls=[ToolCall(tool_name="get_order_details", args={"order_id": order_id})],
                )

            if logger:
                logger.step(
                    "rule_engine_delivery_without_order",
                    input_data={"user_id": user_id},
                    output_data=None,
                    streaming_msg="No order id provided: fetching your recent orders for delivery checks...",
                    meta={},
                )
            return Plan(
                plan_type="auto_tool",
                tool_calls=[ToolCall(tool_name="get_user_orders", args={"user_id": user_id})],
            )

        if category == "order_cancel":
            if order_id:
                return Plan(plan_type="auto_tool", tool_calls=[ToolCall(tool_name="get_order_details", args={"order_id": order_id})])
            return Plan(plan_type="auto_tool", tool_calls=[ToolCall(tool_name="get_user_orders", args={"user_id": user_id})])

        # Unknown category fallback.
        if routing.decision == "escalate_admin":
            return Plan(plan_type="escalate_admin", tool_calls=[])
        return Plan(plan_type="auto_no_tool", tool_calls=[])

    # Intentionally no _plan_auto: Planner is rule-based and tool-first.

