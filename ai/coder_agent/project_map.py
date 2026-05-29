from __future__ import annotations

import os
from pathlib import Path


IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".zen_ai",
    "build",
    "dist",
    "models",
}

IGNORED_SUFFIXES = {
    ".gguf",
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
    ".onnx",
}


def build_project_map(
    project_root: str | os.PathLike[str],
    *,
    max_depth: int = 3,
    max_entries: int = 400,
) -> str:
    """Return a compact, deterministic project tree for the coder prompt."""

    root = Path(project_root)
    lines = ["Current project tree (depth 3):"]
    entries_seen = 0

    def should_skip(path: Path, *, is_dir: bool) -> bool:
        name = path.name
        if is_dir:
            return name in IGNORED_DIRS or name.startswith(".")
        return name.startswith(".") or path.suffix.lower() in IGNORED_SUFFIXES

    def visit(path: Path, prefix: str = "", depth: int = 0) -> None:
        nonlocal entries_seen
        if depth > max_depth or entries_seen >= max_entries:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError:
            return
        visible = [
            entry for entry in entries
            if not should_skip(entry, is_dir=entry.is_dir())
        ]
        for i, entry in enumerate(visible):
            if entries_seen >= max_entries:
                lines.append(prefix + "[project map truncated]")
                return
            last = i == len(visible) - 1
            lines.append(prefix + ("`-- " if last else "|-- ") + entry.name)
            entries_seen += 1
            if entry.is_dir():
                visit(entry, prefix + ("    " if last else "|   "), depth + 1)

    visit(root)
    return "\n".join(lines)
