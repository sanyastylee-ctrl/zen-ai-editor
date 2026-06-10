from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import app_data
from core.diagnostics import write_log


CHECKPOINT_VERSION = 1


@dataclass
class AgentRunCheckpoint:
    run_id: str
    profile_id: str
    session_id: str
    project_root: str
    user_task: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    phase: str = ""
    stop_reason: str = ""
    resumable: bool = False
    dismissed: bool = False
    auto_resume_count: int = 0
    max_auto_resume_count: int = 5
    ledger: list[dict] = field(default_factory=list)
    current_goal_id: str = ""
    file_goals: list[dict] = field(default_factory=list)
    dependency_goals: list[str] = field(default_factory=list)
    dependency_state: dict = field(default_factory=dict)
    command_goals: list[str] = field(default_factory=list)
    command_goals_done: list[str] = field(default_factory=list)
    repair: dict = field(default_factory=dict)
    completed_tool_hashes: list[str] = field(default_factory=list)
    failed_tool_hashes: list[str] = field(default_factory=list)
    last_successful_action: dict = field(default_factory=dict)
    last_failed_action: dict = field(default_factory=dict)
    pending_next_action: str = ""
    python_cmd: str = ""
    access_mode: str = "safe_access"
    profile_snapshot: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    continuation_state: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "session_id": self.session_id,
            "project_root": self.project_root,
            "user_task": self.user_task,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "phase": self.phase,
            "stop_reason": self.stop_reason,
            "resumable": self.resumable,
            "dismissed": self.dismissed,
            "auto_resume_count": self.auto_resume_count,
            "max_auto_resume_count": self.max_auto_resume_count,
            "ledger": self.ledger,
            "current_goal_id": self.current_goal_id,
            "file_goals": self.file_goals,
            "dependency_goals": self.dependency_goals,
            "dependency_state": self.dependency_state,
            "command_goals": self.command_goals,
            "command_goals_done": self.command_goals_done,
            "repair": self.repair,
            "completed_tool_hashes": self.completed_tool_hashes,
            "failed_tool_hashes": self.failed_tool_hashes,
            "last_successful_action": self.last_successful_action,
            "last_failed_action": self.last_failed_action,
            "pending_next_action": self.pending_next_action,
            "python_cmd": self.python_cmd,
            "access_mode": self.access_mode,
            "profile_snapshot": self.profile_snapshot,
            "summary": self.summary,
            "continuation_state": self.continuation_state,
        }

    @classmethod
    def from_json(cls, data: dict) -> "AgentRunCheckpoint":
        if not isinstance(data, dict):
            raise ValueError("checkpoint is not an object")
        return cls(
            run_id=str(data.get("run_id") or ""),
            profile_id=str(data.get("profile_id") or ""),
            session_id=str(data.get("session_id") or ""),
            project_root=str(data.get("project_root") or ""),
            user_task=str(data.get("user_task") or ""),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            phase=str(data.get("phase") or ""),
            stop_reason=str(data.get("stop_reason") or ""),
            resumable=bool(data.get("resumable", False)),
            dismissed=bool(data.get("dismissed", False)),
            auto_resume_count=int(data.get("auto_resume_count") or 0),
            max_auto_resume_count=int(data.get("max_auto_resume_count") or 5),
            ledger=list(data.get("ledger") or []),
            current_goal_id=str(data.get("current_goal_id") or ""),
            file_goals=list(data.get("file_goals") or []),
            dependency_goals=list(data.get("dependency_goals") or []),
            dependency_state=dict(data.get("dependency_state") or {}),
            command_goals=list(data.get("command_goals") or []),
            command_goals_done=list(data.get("command_goals_done") or []),
            repair=dict(data.get("repair") or {}),
            completed_tool_hashes=list(data.get("completed_tool_hashes") or []),
            failed_tool_hashes=list(data.get("failed_tool_hashes") or []),
            last_successful_action=dict(data.get("last_successful_action") or {}),
            last_failed_action=dict(data.get("last_failed_action") or {}),
            pending_next_action=str(data.get("pending_next_action") or ""),
            python_cmd=str(data.get("python_cmd") or ""),
            access_mode=str(data.get("access_mode") or "safe_access"),
            profile_snapshot=dict(data.get("profile_snapshot") or {}),
            summary=dict(data.get("summary") or {}),
            continuation_state=dict(data.get("continuation_state") or {}),
        )


def checkpoint_path(run_id: str) -> Path:
    safe = "".join(ch for ch in str(run_id or "") if ch.isalnum() or ch in {"-", "_"})[:80]
    if not safe:
        safe = hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()[:16]
    return app_data.AGENT_RUNS_DIR / f"{safe}.json"


def save_checkpoint(checkpoint: AgentRunCheckpoint) -> bool:
    checkpoint.updated_at = time.time()
    path = checkpoint_path(checkpoint.run_id)
    ok = app_data.atomic_write_json(path, checkpoint.to_json())
    write_log(
        f"[agent_checkpoint_saved] run_id={checkpoint.run_id} "
        f"path={path} resumable={checkpoint.resumable} reason={checkpoint.stop_reason!r}"
    )
    return ok


def load_checkpoint(run_id: str) -> AgentRunCheckpoint | None:
    path = checkpoint_path(run_id)
    data = app_data.read_json(path, None)
    if not isinstance(data, dict):
        write_log(f"[agent_checkpoint_load_failed] run_id={run_id} path={path}")
        return None
    try:
        checkpoint = AgentRunCheckpoint.from_json(data)
    except (TypeError, ValueError) as exc:
        write_log(f"[agent_checkpoint_load_failed] run_id={run_id} error={exc}")
        return None
    write_log(f"[agent_checkpoint_loaded] run_id={checkpoint.run_id} path={path}")
    return checkpoint


def latest_resumable_checkpoint(*, profile_id: str = "", project_root: str = "", session_id: str = "") -> AgentRunCheckpoint | None:
    best: AgentRunCheckpoint | None = None
    try:
        paths = list(app_data.AGENT_RUNS_DIR.glob("*.json"))
    except OSError:
        return None
    normalized_root = str(project_root or "")
    for path in paths:
        data = app_data.read_json(path, None)
        if not isinstance(data, dict):
            continue
        try:
            checkpoint = AgentRunCheckpoint.from_json(data)
        except (TypeError, ValueError):
            continue
        if not checkpoint.resumable or checkpoint.dismissed:
            continue
        if profile_id and checkpoint.profile_id != profile_id:
            continue
        if session_id and checkpoint.session_id != session_id:
            continue
        if normalized_root and checkpoint.project_root != normalized_root:
            continue
        if best is None or checkpoint.updated_at > best.updated_at:
            best = checkpoint
    return best


def mark_checkpoint_done(run_id: str) -> None:
    checkpoint = load_checkpoint(run_id)
    if checkpoint is None:
        return
    checkpoint.resumable = False
    checkpoint.stop_reason = "done"
    save_checkpoint(checkpoint)


def mark_checkpoint_dismissed(run_id: str) -> None:
    checkpoint = load_checkpoint(run_id)
    if checkpoint is None:
        return
    checkpoint.dismissed = True
    checkpoint.resumable = False
    save_checkpoint(checkpoint)
