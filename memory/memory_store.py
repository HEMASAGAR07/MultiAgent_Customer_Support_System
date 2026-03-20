from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from filelock import FileLock


Json = dict[str, Any]


class MemoryStore:
    """
    File-backed store for tickets and associated agent traces.
    This enables admin observability across Streamlit sessions.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.tickets_path = data_dir / "support_tickets.json"
        self._lock_path = data_dir / "support_tickets.lock"
        self._lock = FileLock(str(self._lock_path))

    def load_tickets(self) -> list[Json]:
        if not self.tickets_path.exists():
            return []
        raw = self.tickets_path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        return json.loads(raw)

    def save_tickets(self, tickets: list[Json]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tickets_path.write_text(json.dumps(tickets, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_next_ticket_id(self, existing: list[Json]) -> str:
        max_num = 0
        for t in existing:
            tid = t.get("ticket_id", "")
            if isinstance(tid, str) and tid.startswith("TKT"):
                try:
                    max_num = max(max_num, int(tid.replace("TKT", "")))
                except Exception:
                    pass
        return f"TKT{max_num + 1:04d}"

    def add_ticket(self, ticket: Json) -> Json:
        with self._lock:
            tickets = self.load_tickets()
            ticket_id = ticket.get("ticket_id")
            if not ticket_id:
                ticket["ticket_id"] = self.get_next_ticket_id(tickets)
            tickets.append(ticket)
            self.save_tickets(tickets)
        return ticket

    def get_user_tickets(self, user_id: str) -> list[Json]:
        tickets = self.load_tickets()
        return [t for t in tickets if t.get("user_id") == user_id]

    def get_recent_user_history(self, user_id: str, limit: int = 5) -> list[Json]:
        tickets = self.get_user_tickets(user_id)
        # Sort by created_at desc if present, else keep original.
        def _key(t: Json) -> str:
            return str(t.get("created_at", "")) or ""

        tickets_sorted = sorted(tickets, key=_key, reverse=True)
        return tickets_sorted[:limit]

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

