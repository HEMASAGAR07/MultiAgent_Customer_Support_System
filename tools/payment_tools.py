from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dateutil import parser as dt_parser


Json = dict[str, Any]


def _get_project_data_dir() -> Path:
    # project/tools/payment_tools.py -> parents[1]=project
    return Path(__file__).resolve().parents[1] / "data"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return default
    return json.loads(raw)


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_transaction_id(transactions: list[Json]) -> str:
    max_num = 0
    for t in transactions:
        tid = str(t.get("transaction_id", ""))
        m = re.match(r"TXN(\d+)", tid)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"TXN{max_num + 1:05d}"


def check_payment_status(order_id: str | None, stream_logger=None, memory_store=None) -> Json:
    """
    Mock API tool:
    Returns payment/order state by reading JSON datasets.
    """
    data_dir = _get_project_data_dir()
    orders = _load_json(data_dir / "orders.json", [])
    txns = _load_json(data_dir / "transactions.json", [])

    if not order_id:
        return {"ok": False, "error": "Missing order_id"}

    order = next((o for o in orders if str(o.get("order_id")) == order_id), None)
    txns_for_order = [t for t in txns if str(t.get("order_id")) == order_id]

    if stream_logger:
        stream_logger.step(
            "check_payment_status_lookup",
            input_data={"order_id": order_id},
            output_data={"order_found": bool(order), "txn_count": len(txns_for_order)},
            streaming_msg="Looking up payment status in mock gateway...",
        )

    if not order:
        return {"ok": False, "error": f"Order {order_id} not found", "order_id": order_id, "payment_status": None}

    # Derive payment status from order and/or transactions.
    order_payment_status = str(order.get("payment_status", ""))
    transaction_statuses = sorted({str(t.get("status", "")) for t in txns_for_order})

    created_at = order.get("created_at")
    age_hours = None
    try:
        if created_at:
            dt_obj = dt_parser.parse(str(created_at))
            # assume naive => UTC
            if dt_obj.tzinfo is None:
                from datetime import timezone

                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            from datetime import datetime

            age_hours = (datetime.now(timezone.utc) - dt_obj).total_seconds() / 3600.0
    except Exception:
        age_hours = None

    return {
        "ok": True,
        "order_id": order_id,
        "order_status": order.get("order_status"),
        "payment_status": order_payment_status,
        "transaction_statuses": transaction_statuses,
        "created_at": created_at,
        "age_hours": age_hours,
    }


def initiate_refund(order_id: str | None, reason: str, stream_logger=None) -> Json:
    """
    Mock API tool:
    Updates order/payment state to 'refunded' and appends a new transaction record.
    """
    data_dir = _get_project_data_dir()
    orders_path = data_dir / "orders.json"
    txns_path = data_dir / "transactions.json"

    orders = _load_json(orders_path, [])
    transactions = _load_json(txns_path, [])

    if not order_id:
        return {"ok": False, "error": "Missing order_id"}

    order = next((o for o in orders if str(o.get("order_id")) == order_id), None)
    if not order:
        return {"ok": False, "error": f"Order {order_id} not found", "order_id": order_id}

    if stream_logger:
        stream_logger.step(
            "initiate_refund_prepare",
            input_data={"order_id": order_id, "reason": reason},
            output_data={"old_payment_status": order.get("payment_status")},
            streaming_msg="Initiating refund in mock system...",
        )

    order["payment_status"] = "refunded"
    order["order_status"] = order.get("order_status") or "cancelled"

    txn = {
        "transaction_id": _next_transaction_id(transactions),
        "order_id": order_id,
        "status": "success",
        "amount": order.get("amount"),
        "gateway": "RefundGateway",
        "timestamp": order.get("created_at"),
        "reason": reason,
    }
    transactions.append(txn)

    _save_json(orders_path, orders)
    _save_json(txns_path, transactions)

    return {
        "ok": True,
        "order_id": order_id,
        "refund_status": "initiated",
        "new_payment_status": order.get("payment_status"),
        "transaction_id": txn["transaction_id"],
    }

