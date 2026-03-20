from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dateutil import parser as dt_parser


Json = dict[str, Any]


def _get_project_data_dir() -> Path:
    # project/tools/order_tools.py -> parents[1]=project
    return Path(__file__).resolve().parents[1] / "data"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return default
    return json.loads(raw)


def get_order_details(order_id: str | None, stream_logger=None) -> Json:
    """
    Mock API tool:
    Returns order info from orders.json.
    """
    data_dir = _get_project_data_dir()
    orders = _load_json(data_dir / "orders.json", [])

    if not order_id:
        return {"ok": False, "error": "Missing order_id"}

    order = next((o for o in orders if str(o.get("order_id")) == order_id), None)
    if stream_logger:
        stream_logger.step(
            "get_order_details_lookup",
            input_data={"order_id": order_id},
            output_data={"order_found": bool(order)},
            streaming_msg="Fetching order details from mock order system...",
        )

    if not order:
        return {"ok": False, "error": f"Order {order_id} not found", "order_id": order_id}

    created_at = order.get("created_at")
    age_hours = None
    try:
        dt_obj = dt_parser.parse(str(created_at)) if created_at else None
        if dt_obj:
            if dt_obj.tzinfo is None:
                from datetime import timezone, datetime

                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            from datetime import datetime as dt

            age_hours = (dt.now(dt_obj.tzinfo) - dt_obj).total_seconds() / 3600.0
    except Exception:
        age_hours = None

    return {
        "ok": True,
        "order_id": order_id,
        "order_status": order.get("order_status"),
        "payment_status": order.get("payment_status"),
        "amount": order.get("amount"),
        "created_at": created_at,
        "age_hours": age_hours,
    }


def get_user_orders(user_id: str | None, stream_logger=None) -> Json:
    """
    Mock API tool:
    Returns orders belonging to a user (from orders.json).
    """
    data_dir = _get_project_data_dir()
    orders = _load_json(data_dir / "orders.json", [])

    if not user_id:
        return {"ok": False, "error": "Missing user_id"}

    user_orders = [o for o in orders if str(o.get("user_id")) == str(user_id)]

    # Sort by created_at desc if present.
    try:
        user_orders = sorted(
            user_orders,
            key=lambda o: dt_parser.parse(str(o.get("created_at"))) if o.get("created_at") else dt_parser.parse("1970-01-01"),
            reverse=True,
        )
    except Exception:
        pass

    if stream_logger:
        stream_logger.step(
            "get_user_orders_lookup",
            input_data={"user_id": user_id},
            output_data={"order_count": len(user_orders)},
            streaming_msg="Fetching your recent orders from our system...",
            meta={},
        )

    return {
        "ok": True,
        "user_id": user_id,
        "orders": [
            {
                "order_id": o.get("order_id"),
                "order_status": o.get("order_status"),
                "payment_status": o.get("payment_status"),
                "amount": o.get("amount"),
                "created_at": o.get("created_at"),
            }
            for o in user_orders
        ],
    }

