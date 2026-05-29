"""
Base classes for project-scoped agent tools.

Every filesystem path accepted by a tool is resolved inside the current
ProjectManager root. Absolute paths and ``..`` escapes outside the project are
rejected before the tool touches the disk.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.projects import ProjectManager


@dataclass
class ToolCall:
    name: str
    args: dict[str, str]
    raw: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class ToolResult:
    ok: bool
    output: str
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    critical: bool = False

    @classmethod
    def error(cls, message: str, *, critical: bool = False) -> "ToolResult":
        return cls(ok=False, output=f"[error: {message}]", critical=critical)


class Tool:
    name = ""
    description = ""
    mutates_project = False
    runs_command = False

    def __init__(self, project_root: str | None = None) -> None:
        self.project_root = os.path.realpath(
            project_root or ProjectManager.instance().current
        )

    def execute(self, call: ToolCall) -> ToolResult:
        raise NotImplementedError

    def preview(self, call: ToolCall) -> str:
        return ""

    def safe_path(self, path: str, *, allow_missing: bool = False) -> str:
        if not path or not path.strip():
            candidate = self.project_root
        else:
            raw = path.strip()
            if os.path.isabs(raw):
                candidate = os.path.realpath(raw)
            else:
                candidate = os.path.realpath(os.path.join(self.project_root, raw))

        try:
            common = os.path.commonpath([self.project_root, candidate])
        except ValueError as e:
            raise ValueError("path outside project") from e
        if common != self.project_root:
            raise ValueError("path outside project")
        if not allow_missing and not os.path.exists(candidate):
            raise FileNotFoundError(path)
        return candidate

    def relpath(self, path: str) -> str:
        return os.path.relpath(path, self.project_root).replace(os.sep, "/")
