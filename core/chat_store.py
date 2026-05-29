"""Persistent chat sessions stored as JSON in the application's AppData area."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from .app_data import CHATS_DIR, SESSIONS_DIR, atomic_write_json, read_json


_STATE_FILE = SESSIONS_DIR / "chat_state.json"


class ChatSessionStore:
    """Persists the existing per-profile conversation model without changing UI records."""

    def __init__(self) -> None:
        state = read_json(_STATE_FILE, {})
        self._state: dict[str, Any] = state if isinstance(state, dict) else {}
        self._state.setdefault("sessions_by_profile", {})
        self._state.setdefault("last_profile_id", "")

    def last_profile_id(self) -> str:
        value = self._state.get("last_profile_id", "")
        return value if isinstance(value, str) else ""

    def set_last_profile_id(self, profile_id: str) -> None:
        self._state["last_profile_id"] = profile_id
        self._save_state()

    def load_profile(self, profile_id: str) -> tuple[list[dict], list[tuple[str, str]]]:
        session_id = self._session_id_for(profile_id, create=False)
        if not session_id:
            return [], []
        data = read_json(CHATS_DIR / f"{session_id}.json", {})
        if not isinstance(data, dict):
            return [], []

        records: list[dict] = []
        for message in data.get("messages", []):
            if not isinstance(message, dict):
                continue
            record = dict(message)
            record["text"] = str(record.pop("content", record.get("text", "")) or "")
            record.pop("streaming", None)
            records.append(record)

        history: list[tuple[str, str]] = []
        for turn in data.get("history", []):
            if isinstance(turn, list) and len(turn) == 2:
                history.append((str(turn[0]), str(turn[1])))
        if not history:
            history = self._history_from_records(records)
        return records, history

    def save_profile(
        self,
        profile_id: str,
        profile_name: str,
        records: list[dict],
        history: list[tuple[str, str]],
    ) -> None:
        session_id = self._session_id_for(profile_id, create=True)
        if not session_id:
            return
        existing = read_json(CHATS_DIR / f"{session_id}.json", {})
        created_at = existing.get("created_at") if isinstance(existing, dict) else None
        messages = [self._serialized_message(record) for record in records]
        title = self._title_from_messages(messages) or profile_name or "Chat"
        data = {
            "id": session_id,
            "profile_id": profile_id,
            "title": title,
            "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "messages": messages,
            "history": [[user, assistant] for user, assistant in history],
        }
        atomic_write_json(CHATS_DIR / f"{session_id}.json", data)
        self._save_state()

    def _session_id_for(self, profile_id: str, *, create: bool) -> str:
        sessions = self._state.setdefault("sessions_by_profile", {})
        session_id = sessions.get(profile_id)
        if isinstance(session_id, str) and session_id:
            return session_id
        if not create:
            return ""
        session_id = uuid.uuid4().hex
        sessions[profile_id] = session_id
        return session_id

    def _save_state(self) -> None:
        atomic_write_json(_STATE_FILE, self._state)

    @staticmethod
    def _serialized_message(record: dict) -> dict:
        stored = {
            key: value for key, value in record.items()
            if key in {
                "role", "sender", "time", "profile_kind", "tool_name",
                "detail", "output", "ok",
            }
        }
        stored["role"] = str(record.get("role", "assistant"))
        stored["content"] = str(record.get("text", "") or "")
        return stored

    @staticmethod
    def _title_from_messages(messages: list[dict]) -> str:
        for message in messages:
            if message.get("role") == "user" and message.get("content", "").strip():
                first_line = message["content"].strip().splitlines()[0]
                return first_line[:80]
        return ""

    @staticmethod
    def _history_from_records(records: list[dict]) -> list[tuple[str, str]]:
        turns: list[tuple[str, str]] = []
        pending_user = ""
        for record in records:
            role = record.get("role")
            if role == "user":
                pending_user = str(record.get("text", ""))
            elif role == "assistant" and pending_user:
                turns.append((pending_user, str(record.get("text", ""))))
                pending_user = ""
        return turns
