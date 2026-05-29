"""Companion state and long-term memory helpers.

This module is intentionally small and side-effect light. The chat pipeline can
ask it for compact prompt context, while UI code can review/edit memories.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .app_data import MEMORY_DIR, atomic_write_json, read_json


COMPANION_MEMORY_FILE = MEMORY_DIR / "companion_memory.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded(value: Any, default: int = 5) -> int:
    try:
        return max(0, min(10, int(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class CompanionMemory:
    id: str
    text: str
    category: str = "preference"
    confidence: float = 0.7
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    source: str = "user"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompanionMemory":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            text=str(data.get("text") or "").strip(),
            category=str(data.get("category") or "preference"),
            confidence=float(data.get("confidence", 0.7) or 0.7),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
            source=str(data.get("source") or "user"),
        )


@dataclass
class CompanionState:
    persona_name: str = "Лера"
    base_traits: str = "мягкая, живая, внимательная, с лёгкой игривостью"
    mood: str = "спокойное"
    relationship_state: str = "близкая, тёплая динамика"
    user_preferences: str = ""
    boundaries: str = ""
    memories: list[CompanionMemory] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)
    long_term_facts: list[str] = field(default_factory=list)
    project_interests: list[str] = field(default_factory=list)
    roleplay_mode: bool = False
    support_mode: bool = False
    idea_mode: bool = False
    companion_mode: str = "chat"
    tenderness: int = 7
    playfulness: int = 6
    initiative: int = 5
    romance: int = 5
    humor: int = 5
    autonomy: int = 5
    speaking_style: str = "живой разговорный"
    memory_enabled: bool = True

    @classmethod
    def from_persona(
        cls,
        persona: dict[str, Any] | None,
        memories: list[CompanionMemory] | None = None,
    ) -> "CompanionState":
        data = dict(persona or {})
        mode = str(data.get("companion_mode") or "chat")
        return cls(
            persona_name=str(data.get("character_name") or data.get("persona_name") or "Лера"),
            base_traits=str(data.get("base_traits") or data.get("personality") or cls.base_traits),
            mood=str(data.get("current_mood") or data.get("mood") or "спокойное"),
            relationship_state=str(data.get("relationship_to_user") or "близкая, тёплая динамика"),
            user_preferences=str(data.get("user_preferences") or ""),
            boundaries=str(data.get("boundaries") or ""),
            memories=list(memories or []),
            recent_topics=_split_lines(data.get("recent_topics")),
            long_term_facts=_split_lines(data.get("long_term_facts")),
            project_interests=_split_lines(data.get("project_interests")),
            roleplay_mode=mode == "roleplay" or _truthy(data.get("roleplay_mode")),
            support_mode=mode == "support" or _truthy(data.get("support_mode")),
            idea_mode=mode in {"ideas", "project_brainstorm"} or _truthy(data.get("idea_mode")),
            companion_mode=mode,
            tenderness=_bounded(data.get("tenderness"), 7),
            playfulness=_bounded(data.get("playfulness"), 6),
            initiative=_bounded(data.get("initiative"), 5),
            romance=_bounded(data.get("romance"), 5),
            humor=_bounded(data.get("humor"), 5),
            autonomy=_bounded(data.get("autonomy"), 5),
            speaking_style=str(data.get("speaking_style") or "живой разговорный"),
            memory_enabled=_truthy(data.get("memory_enabled"), default=True),
        )


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "да", "вкл"}


def _split_lines(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


class CompanionMemoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or COMPANION_MEMORY_FILE

    def load(self) -> list[CompanionMemory]:
        data = read_json(self.path, {"version": 1, "memories": []})
        memories = data.get("memories", []) if isinstance(data, dict) else []
        result: list[CompanionMemory] = []
        for item in memories:
            if isinstance(item, dict):
                memory = CompanionMemory.from_dict(item)
                if memory.text:
                    result.append(memory)
        return result

    def save(self, memories: list[CompanionMemory]) -> bool:
        return atomic_write_json(
            self.path,
            {"version": 1, "memories": [asdict(memory) for memory in memories if memory.text]},
        )

    def add(self, text: str, category: str = "preference", source: str = "user") -> CompanionMemory | None:
        text = _normalize_memory_text(text)
        if not text:
            return None
        memories = self.load()
        for memory in memories:
            if memory.text.casefold() == text.casefold():
                memory.updated_at = _now_iso()
                memory.confidence = max(memory.confidence, 0.8)
                self.save(memories)
                return memory
        memory = CompanionMemory(
            id=uuid.uuid4().hex,
            text=text,
            category=category,
            confidence=0.8,
            source=source,
        )
        memories.append(memory)
        self.save(memories)
        return memory

    def delete(self, memory_id: str) -> bool:
        memories = self.load()
        kept = [memory for memory in memories if memory.id != memory_id]
        if len(kept) == len(memories):
            return False
        return self.save(kept)

    def clear(self) -> bool:
        return self.save([])


REMEMBER_RE = re.compile(
    r"(?:запомни|remember(?:\s+that)?|важно:\s*)(?P<fact>.+)",
    re.IGNORECASE | re.DOTALL,
)


def extract_explicit_memory(text: str) -> str:
    match = REMEMBER_RE.search(text or "")
    if not match:
        return ""
    return _normalize_memory_text(match.group("fact"))


def _normalize_memory_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip(" :;,.。\n\t")
    return text[:500]


_SMOKE_MARKER_RE = re.compile(r"\b(?:msg|nmsgr)-[a-z0-9-]+\b", re.IGNORECASE)


def normalize_companion_reply_for_repeat(text: str) -> str:
    """Normalize replies enough to catch replay with only turn markers changed."""
    value = _SMOKE_MARKER_RE.sub("<marker>", str(text or ""))
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value


_WRAPPER_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "«": "»",
    "“": "”",
    '"': '"',
    "'": "'",
    "`": "`",
    "*": "*",
    "_": "_",
}


def normalize_companion_echo_text(text: str) -> str:
    """Normalize user/assistant text for deterministic echo detection."""
    value = str(text or "").strip()
    changed = True
    while changed and len(value) >= 2:
        changed = False
        value = value.strip()
        first, last = value[:1], value[-1:]
        if _WRAPPER_PAIRS.get(first) == last:
            value = value[1:-1].strip()
            changed = True
    value = re.sub(r"^[\s:;,.!?！？。…—-]+|[\s:;,.!?！？。…—-]+$", "", value)
    value = re.sub(r"[^\w]+", "", value, flags=re.UNICODE).casefold()
    return value


def companion_echo_similarity(user_text: str, response_text: str) -> float:
    user_norm = normalize_companion_echo_text(user_text)
    response_norm = normalize_companion_echo_text(response_text)
    if not user_norm and not response_norm:
        return 1.0
    if not user_norm or not response_norm:
        return 0.0
    if user_norm == response_norm:
        return 1.0
    return SequenceMatcher(None, user_norm, response_norm).ratio()


def is_companion_echo_response(user_text: str, response_text: str, *, threshold: float = 0.92) -> tuple[bool, float]:
    """Return whether response is just an echo of the latest user message."""
    user_norm = normalize_companion_echo_text(user_text)
    response_norm = normalize_companion_echo_text(response_text)
    if not user_norm or not response_norm:
        return False, 0.0
    similarity = companion_echo_similarity(user_text, response_text)
    if user_norm == response_norm:
        return True, similarity
    # Short responses that contain essentially only the user's text are echoes:
    # "Привет" -> "*Привет*", "(Привет)", "Привет!".
    if len(response_norm) <= max(len(user_norm) + 6, int(len(user_norm) * 1.25)) and (
        user_norm in response_norm or response_norm in user_norm
    ):
        return True, similarity
    if similarity >= threshold and len(response_norm) <= max(12, int(len(user_norm) * 1.35)):
        return True, similarity
    return False, similarity


def compact_companion_history(
    history: list[tuple[str, str]] | None,
    *,
    max_turns: int = 18,
    repeat_threshold: int = 2,
) -> list[tuple[str, str]]:
    """
    Keep recent companion history, but remove repeated assistant scripts.

    Long dialogs can poison the prompt when the same assistant response appears
    again and again. Replacing the prose with a textual marker still gives the
    model a phrase to copy, so repeated assistant turns are omitted from the
    prompt entirely. The latest live user message is pinned separately by the
    prompt builder.
    """
    recent = list(history or [])[-max_turns:]
    counts: dict[str, int] = {}
    for _user, assistant in recent:
        key = normalize_companion_reply_for_repeat(assistant)
        if key:
            counts[key] = counts.get(key, 0) + 1

    compacted: list[tuple[str, str]] = []
    for user, assistant in recent:
        key = normalize_companion_reply_for_repeat(assistant)
        if key and counts.get(key, 0) >= repeat_threshold:
            continue
        compacted.append((user, assistant))
    return compacted


def build_companion_context(
    persona: dict[str, Any] | None,
    *,
    memory_store: CompanionMemoryStore | None = None,
    max_memories: int = 12,
) -> str:
    """Return compact non-chatty context for the companion system prompt."""
    store = memory_store or CompanionMemoryStore()
    memories = store.load()
    state = CompanionState.from_persona(persona, memories=memories)

    lines = [
        "=== Companion State v2 ===",
        f"Mode: {state.companion_mode}",
        f"Mood: {state.mood}",
        f"Relationship: {state.relationship_state}",
        f"Traits: {state.base_traits}",
        (
            "Tuning: "
            f"tenderness={state.tenderness}/10, playfulness={state.playfulness}/10, "
            f"initiative={state.initiative}/10, romance={state.romance}/10, "
            f"humor={state.humor}/10, autonomy={state.autonomy}/10"
        ),
        f"Style: {state.speaking_style}",
    ]
    if state.boundaries:
        lines.append(f"User boundaries/preferences: {state.boundaries}")
    if state.user_preferences:
        lines.append(f"User preferences: {state.user_preferences}")
    if state.project_interests:
        lines.append("Project interests: " + "; ".join(state.project_interests[:6]))
    if state.memory_enabled and state.memories:
        lines.append("Long-term memories:")
        for memory in state.memories[-max_memories:]:
            lines.append(f"- [{memory.category}; {memory.confidence:.1f}] {memory.text}")
    else:
        lines.append("Long-term memories: disabled or empty.")
    lines.append(
        "Use this as background continuity, not as a script. Do not store every message; "
        "only explicit stable facts should become memories."
    )
    lines.extend(
        [
            "Live response contract:",
            "- You are Lera, not the user.",
            "- Answer the latest user message as user input, not as text to imitate.",
            "- Never repeat the user's message as the whole reply.",
            "- Never reply only by wrapping the user's text in parentheses, quotes, or asterisks.",
            "- Always add new content: a reaction, feeling, question, idea, or continuation.",
            "- If the user says 'Привет', greet naturally; if the user says 'Тут?', answer that you are here.",
            "- Do not roleplay as the user and do not continue stale assistant replies unless explicitly asked.",
        ]
    )
    return "\n".join(lines)
