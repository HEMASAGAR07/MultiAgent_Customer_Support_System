from __future__ import annotations

import json
from pathlib import Path
from typing import Any


Json = dict[str, Any]


def _get_project_data_dir() -> Path:
    # project/tools/account_tools.py -> parents[1]=tools, parents[2]=project
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


def get_user_account_status(user_id: str | None, stream_logger=None) -> Json:
    """
    Mock API tool:
    Reads users.json and returns the current account status.
    """
    data_dir = _get_project_data_dir()
    users = _load_json(data_dir / "users.json", [])

    if not user_id:
        return {"ok": False, "error": "Missing user_id"}

    user = next((u for u in users if str(u.get("user_id")) == str(user_id)), None)
    if stream_logger:
        stream_logger.step(
            "get_user_account_status_lookup",
            input_data={"user_id": user_id},
            output_data={"user_found": bool(user)},
            streaming_msg="Checking your account status in our system...",
            meta={},
        )

    if not user:
        return {"ok": False, "error": f"User {user_id} not found", "user_id": user_id}

    return {
        "ok": True,
        "user_id": user_id,
        "account_status": user.get("account_status"),
        "created_at": user.get("created_at"),
    }


def unblock_account(user_id: str | None, stream_logger=None) -> Json:
    """
    Mock API tool:
    Updates users.json -> account_status = "active".
    """
    data_dir = _get_project_data_dir()
    users_path = data_dir / "users.json"

    users = _load_json(users_path, [])

    if not user_id:
        return {"ok": False, "error": "Missing user_id"}

    user = next((u for u in users if str(u.get("user_id")) == str(user_id)), None)
    if not user:
        return {"ok": False, "error": f"User {user_id} not found", "user_id": user_id}

    prev_status = user.get("account_status")
    user["account_status"] = "active"
    user["unblock_requested_at"] = user.get("unblock_requested_at") or None

    if stream_logger:
        stream_logger.step(
            "unblock_account_update",
            input_data={"user_id": user_id, "prev_status": prev_status},
            output_data={"new_status": user.get("account_status")},
            streaming_msg="Initiating unblock request (mock)...",
        )

    _save_json(users_path, users)

    return {
        "ok": True,
        "user_id": user_id,
        "prev_status": prev_status,
        "new_status": user.get("account_status"),
        "unblock_status": "initiated",
        "estimated_access_hours": 2,
    }

