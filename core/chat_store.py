"""Persistent profile-scoped chat sessions stored in AppData."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .app_data import CHATS_DIR, SESSIONS_DIR, atomic_write_json, read_json


_STATE_FILE = SESSIONS_DIR / "chat_state.json"
DEFAULT_RENDER_LIMIT = 50


@dataclass
class ChatSession:
    id: str
    title: str
    profile_kind: str
    project_path: str = ""
    created_at: str = ""
    updated_at: str = ""
    pinned: bool = False
    archived: bool = False
    deleted: bool = False
    message_count: int = 0
    last_message_preview: str = ""
    profile_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatSession":
        now = datetime.now().isoformat(timespec="seconds")
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            title=str(data.get("title") or "Chat"),
            profile_kind=str(data.get("profile_kind") or data.get("profile_id") or "generic"),
            project_path=str(data.get("project_path") or ""),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
            pinned=bool(data.get("pinned", False)),
            archived=bool(data.get("archived", False)),
            deleted=bool(data.get("deleted", False)),
            message_count=int(data.get("message_count") or 0),
            last_message_preview=str(data.get("last_message_preview") or ""),
            profile_id=str(data.get("profile_id") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChatSessionStore:
    """Index + per-session chat storage.

    The index keeps lightweight metadata only. Session files keep messages.
    `save_profile/load_profile` remain as a compatibility layer for older tests
    and previously saved one-chat-per-profile data.
    """

    def __init__(self) -> None:
        state = read_json(_STATE_FILE, {})
        self._state: dict[str, Any] = state if isinstance(state, dict) else {}
        self._state.setdefault("sessions_by_profile", {})
        self._state.setdefault("last_profile_id", "")
        self._state.setdefault("last_active_by_profile_kind", {})
        self._index: dict[str, Any] = self._load_index()

    def last_profile_id(self) -> str:
        value = self._state.get("last_profile_id", "")
        return value if isinstance(value, str) else ""

    def set_last_profile_id(self, profile_id: str) -> None:
        self._state["last_profile_id"] = profile_id
        self._save_state()

    def create_session(
        self,
        profile_kind: str,
        title: str,
        project_path: str = "",
        *,
        profile_id: str = "",
        set_active: bool = True,
    ) -> ChatSession:
        now = datetime.now().isoformat(timespec="seconds")
        session = ChatSession(
            id=uuid.uuid4().hex,
            title=title or self.default_title(profile_kind),
            profile_kind=str(profile_kind),
            project_path=str(project_path or ""),
            created_at=now,
            updated_at=now,
            profile_id=str(profile_id or ""),
        )
        self._index.setdefault("sessions", {})[session.id] = session.to_dict()
        self._write_session_file(session, [])
        self._save_index()
        if set_active:
            self.set_last_active_session(profile_kind, session.id, profile_id=profile_id)
        return session

    def list_sessions(self, profile_kind: str | None = None, project_path: str | None = None) -> list[ChatSession]:
        sessions: list[ChatSession] = []
        for raw in self._index.get("sessions", {}).values():
            if not isinstance(raw, dict):
                continue
            session = ChatSession.from_dict(raw)
            if session.deleted or session.archived:
                continue
            if profile_kind is not None and session.profile_kind != profile_kind:
                continue
            if project_path is not None and session.project_path != project_path:
                continue
            sessions.append(session)
        sessions.sort(key=lambda s: (s.pinned, s.updated_at), reverse=True)
        return sessions

    def load_session(
        self,
        session_id: str,
        *,
        message_limit: int | None = DEFAULT_RENDER_LIMIT,
    ) -> tuple[list[dict], list[tuple[str, str]]]:
        data = read_json(self._session_path(session_id), {})
        if not isinstance(data, dict):
            return [], []
        raw_messages = data.get("messages", [])
        if not isinstance(raw_messages, list):
            raw_messages = []
        if message_limit is not None and message_limit > 0:
            raw_messages = raw_messages[-message_limit:]
        records = [self._record_from_message(m) for m in raw_messages if isinstance(m, dict)]
        history = self._history_from_records(records)
        if not history:
            raw_history = data.get("history", [])
            if isinstance(raw_history, list):
                for turn in raw_history:
                    if isinstance(turn, list) and len(turn) == 2:
                        history.append((str(turn[0]), str(turn[1])))
        return records, history

    def save_message(self, session_id: str, record: dict) -> None:
        data = self._read_session_file(session_id)
        messages = data.setdefault("messages", [])
        stored = self._serialized_message(record)
        stored.setdefault("id", str(record.get("id") or uuid.uuid4().hex))
        record["id"] = stored["id"]
        messages.append(stored)
        self._write_session_data(session_id, data)

    def update_message(self, session_id: str, record: dict) -> None:
        data = self._read_session_file(session_id)
        messages = data.setdefault("messages", [])
        stored = self._serialized_message(record)
        message_id = str(record.get("id") or stored.get("id") or "")
        replaced = False
        if message_id:
            stored["id"] = message_id
            for index, message in enumerate(messages):
                if isinstance(message, dict) and str(message.get("id") or "") == message_id:
                    messages[index] = stored
                    replaced = True
                    break
        if not replaced:
            stored.setdefault("id", uuid.uuid4().hex)
            record["id"] = stored["id"]
            messages.append(stored)
        self._write_session_data(session_id, data)

    def rename_session(self, session_id: str, title: str) -> None:
        session = self._get_session(session_id)
        if session is None:
            return
        session.title = (title or session.title).strip() or session.title
        session.updated_at = datetime.now().isoformat(timespec="seconds")
        self._index["sessions"][session_id] = session.to_dict()
        self._save_index()

    def clear_session(self, session_id: str) -> None:
        session = self._get_session(session_id)
        if session is None:
            return
        session.message_count = 0
        session.last_message_preview = ""
        session.updated_at = datetime.now().isoformat(timespec="seconds")
        self._index["sessions"][session_id] = session.to_dict()
        self._write_session_file(session, [])
        self._save_index()

    def delete_session(self, session_id: str) -> None:
        session = self._get_session(session_id)
        if session is None:
            return
        session.deleted = True
        session.updated_at = datetime.now().isoformat(timespec="seconds")
        self._index["sessions"][session_id] = session.to_dict()
        try:
            self._session_path(session_id).unlink(missing_ok=True)
        except OSError:
            pass
        for key, value in list(self._state.get("last_active_by_profile_kind", {}).items()):
            if value == session_id:
                self._state["last_active_by_profile_kind"].pop(key, None)
        for key, value in list(self._state.get("sessions_by_profile", {}).items()):
            if value == session_id:
                self._state["sessions_by_profile"].pop(key, None)
        self._save_index()
        self._save_state()

    def get_last_active_session(self, profile_kind: str, *, profile_id: str = "") -> str:
        active = self._state.setdefault("last_active_by_profile_kind", {})
        for key in self._active_keys(profile_kind, profile_id):
            value = active.get(key)
            if isinstance(value, str) and self._get_session(value):
                return value
        legacy = self._state.setdefault("sessions_by_profile", {}).get(profile_id)
        if isinstance(legacy, str) and self._get_session(legacy):
            return legacy
        return ""

    def set_last_active_session(self, profile_kind: str, session_id: str, *, profile_id: str = "") -> None:
        active = self._state.setdefault("last_active_by_profile_kind", {})
        for key in self._active_keys(profile_kind, profile_id):
            active[key] = session_id
        if profile_id:
            self._state.setdefault("sessions_by_profile", {})[profile_id] = session_id
        self._save_state()

    def load_profile(self, profile_id: str) -> tuple[list[dict], list[tuple[str, str]]]:
        session_id = self._session_id_for(profile_id, create=False)
        if not session_id:
            return [], []
        return self.load_session(session_id, message_limit=DEFAULT_RENDER_LIMIT)

    def save_profile(
        self,
        profile_id: str,
        profile_name: str,
        records: list[dict],
        history: list[tuple[str, str]],
    ) -> None:
        session_id = self._session_id_for(profile_id, create=True, title=profile_name)
        data = self._read_session_file(session_id)
        session = self._get_session(session_id)
        messages = [self._serialized_message(record) for record in records]
        title = self._title_from_messages(messages) or profile_name or (session.title if session else "Chat")
        now = datetime.now().isoformat(timespec="seconds")
        if session is None:
            session = ChatSession(
                id=session_id,
                title=title,
                profile_kind=profile_id,
                profile_id=profile_id,
                created_at=str(data.get("created_at") or now),
                updated_at=now,
            )
        session.title = title
        session.updated_at = now
        session.message_count = len(messages)
        session.last_message_preview = self._last_preview(messages)
        self._index.setdefault("sessions", {})[session_id] = session.to_dict()
        data.update({
            "id": session_id,
            "profile_id": profile_id,
            "profile_kind": session.profile_kind,
            "title": title,
            "created_at": session.created_at,
            "updated_at": now,
            "messages": messages,
            "history": [[user, assistant] for user, assistant in history],
        })
        atomic_write_json(self._session_path(session_id), data)
        self._save_index()
        self._save_state()

    def _session_id_for(self, profile_id: str, *, create: bool, title: str = "") -> str:
        sessions = self._state.setdefault("sessions_by_profile", {})
        session_id = sessions.get(profile_id)
        if isinstance(session_id, str) and session_id:
            if session_id not in self._index.get("sessions", {}):
                self._import_legacy_session(session_id, profile_id, title)
            return session_id
        if not create:
            return ""
        session = self.create_session(profile_id, title or "Chat", profile_id=profile_id)
        sessions[profile_id] = session.id
        return session.id

    def _import_legacy_session(self, session_id: str, profile_id: str, title: str = "") -> None:
        data = read_json(self._session_path(session_id), {})
        if not isinstance(data, dict):
            return
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        now = datetime.now().isoformat(timespec="seconds")
        session = ChatSession(
            id=session_id,
            title=str(data.get("title") or title or profile_id or "Chat"),
            profile_kind=str(data.get("profile_kind") or profile_id),
            profile_id=str(data.get("profile_id") or profile_id),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
            message_count=len(messages),
            last_message_preview=self._last_preview(messages),
        )
        self._index.setdefault("sessions", {})[session_id] = session.to_dict()
        self._save_index()

    def _read_session_file(self, session_id: str) -> dict[str, Any]:
        data = read_json(self._session_path(session_id), {})
        return data if isinstance(data, dict) else {}

    def _write_session_file(self, session: ChatSession, messages: list[dict]) -> None:
        data = {
            "id": session.id,
            "title": session.title,
            "profile_kind": session.profile_kind,
            "profile_id": session.profile_id,
            "project_path": session.project_path,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": messages,
        }
        atomic_write_json(self._session_path(session.id), data)

    def _write_session_data(self, session_id: str, data: dict[str, Any]) -> None:
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            messages = []
            data["messages"] = messages
        now = datetime.now().isoformat(timespec="seconds")
        data["updated_at"] = now
        session = self._get_session(session_id)
        if session:
            session.updated_at = now
            session.message_count = len(messages)
            session.last_message_preview = self._last_preview(messages)
            self._index.setdefault("sessions", {})[session_id] = session.to_dict()
            data.setdefault("title", session.title)
            data.setdefault("profile_kind", session.profile_kind)
            data.setdefault("profile_id", session.profile_id)
        atomic_write_json(self._session_path(session_id), data)
        self._save_index()

    def _get_session(self, session_id: str) -> ChatSession | None:
        raw = self._index.get("sessions", {}).get(session_id)
        if not isinstance(raw, dict):
            return None
        session = ChatSession.from_dict(raw)
        return None if session.deleted else session

    def _load_index(self) -> dict[str, Any]:
        data = read_json(self._index_file(), {})
        if not isinstance(data, dict):
            data = {}
        data.setdefault("version", 1)
        data.setdefault("sessions", {})
        return data

    def _save_index(self) -> None:
        atomic_write_json(self._index_file(), self._index)

    def _save_state(self) -> None:
        atomic_write_json(_STATE_FILE, self._state)

    @staticmethod
    def _index_file() -> Path:
        return CHATS_DIR / "index.json"

    @staticmethod
    def _session_path(session_id: str) -> Path:
        return CHATS_DIR / f"{session_id}.json"

    @staticmethod
    def _active_keys(profile_kind: str, profile_id: str = "") -> list[str]:
        keys = []
        if profile_id:
            keys.append(f"profile:{profile_id}")
        keys.append(f"kind:{profile_kind}")
        return keys

    @staticmethod
    def default_title(profile_kind: str) -> str:
        return {
            "coder": "Новый кодер-чат",
            "companion": "Новый чат с Лерой",
            "researcher": "Новый поиск",
        }.get(str(profile_kind), "Новый чат")

    @staticmethod
    def _serialized_message(record: dict) -> dict:
        stored = {
            key: value for key, value in record.items()
            if key in {
                "id", "role", "sender", "time", "profile_kind", "tool_name",
                "detail", "output", "ok",
            }
        }
        stored["id"] = str(stored.get("id") or uuid.uuid4().hex)
        stored["role"] = str(record.get("role", "assistant"))
        stored["content"] = str(record.get("text", "") or "")
        return stored

    @staticmethod
    def _record_from_message(message: dict) -> dict:
        record = dict(message)
        record["text"] = str(record.pop("content", record.get("text", "")) or "")
        record.pop("streaming", None)
        return record

    @staticmethod
    def _title_from_messages(messages: list[dict]) -> str:
        for message in messages:
            if message.get("role") == "user" and str(message.get("content", "")).strip():
                first_line = str(message["content"]).strip().splitlines()[0]
                return first_line[:80]
        return ""

    @staticmethod
    def _last_preview(messages: list[dict]) -> str:
        for message in reversed(messages):
            text = str(message.get("content") or message.get("text") or "").strip()
            if text:
                return text.splitlines()[0][:120]
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
