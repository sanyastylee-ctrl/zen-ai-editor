from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re


@dataclass(frozen=True)
class TracebackFrame:
    path: str
    line: int
    function: str = ""


@dataclass(frozen=True)
class TracebackInfo:
    error_type: str
    message: str
    frames: tuple[TracebackFrame, ...]

    @property
    def relevant_files(self) -> tuple[str, ...]:
        files: list[str] = []
        for frame in self.frames:
            if frame.path and frame.path not in files:
                files.append(frame.path)
        return tuple(files)

    def summary(self) -> str:
        files = ", ".join(
            f"{frame.path}:{frame.line}" if frame.line else frame.path
            for frame in self.frames[-4:]
        )
        detail = f"{self.error_type}: {self.message}".strip(": ")
        return f"{detail} at {files}" if files else detail


_FRAME_RE = re.compile(
    r'^\s*File\s+"(?P<path>[^"]+)",\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>.+))?\s*$',
    re.MULTILINE,
)
_ERROR_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*Error|[A-Za-z_][\w.]*Exception):\s*(?P<message>.*)$")


def parse_traceback(output: str, project_root: str | os.PathLike[str] | None = None) -> TracebackInfo | None:
    text = output or ""
    if "Traceback (most recent call last)" not in text:
        return None

    root = Path(project_root).resolve() if project_root else None
    frames: list[TracebackFrame] = []
    for match in _FRAME_RE.finditer(text):
        raw_path = match.group("path")
        rel_path = _relativize_traceback_path(raw_path, root)
        frames.append(
            TracebackFrame(
                path=rel_path,
                line=int(match.group("line") or 0),
                function=(match.group("func") or "").strip(),
            )
        )

    error_type = "TracebackError"
    message = ""
    for line in reversed(text.strip().splitlines()):
        match = _ERROR_RE.match(line.strip())
        if match:
            error_type = match.group("type")
            message = match.group("message").strip()
            break

    return TracebackInfo(error_type=error_type, message=message, frames=tuple(frames))


def _relativize_traceback_path(raw_path: str, root: Path | None) -> str:
    path_text = (raw_path or "").replace("\\", "/")
    if not root:
        return path_text
    try:
        path = Path(raw_path).resolve()
        return path.relative_to(root).as_posix()
    except Exception:
        return path_text
