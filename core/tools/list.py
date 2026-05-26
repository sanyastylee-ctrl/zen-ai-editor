from __future__ import annotations

import os

from .base import Tool, ToolCall, ToolResult


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files below a project directory."

    SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", ".zen_ai"}

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            root = self.safe_path(call.args.get("path", ""))
            if not os.path.isdir(root):
                return ToolResult.error("not a directory")
            max_depth = int(call.args.get("max_depth", "3") or "3")
            lines = self._walk(root, max_depth=max_depth)
            rel = "." if root == self.project_root else self.relpath(root)
            return ToolResult(
                ok=True,
                title=f"List: {rel}",
                output="\n".join(lines) if lines else "[empty]",
                meta={"path": rel, "count": len(lines)},
            )
        except ValueError:
            return ToolResult.error("path outside project")
        except Exception as e:
            return ToolResult.error(str(e))

    def _walk(self, root: str, *, max_depth: int) -> list[str]:
        lines: list[str] = []

        def visit(path: str, prefix: str = "", depth: int = 0) -> None:
            if depth > max_depth:
                lines.append(prefix + "...")
                return
            try:
                entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
            except OSError:
                return
            entries = [
                e for e in entries
                if not e.name.startswith(".") and (not e.is_dir() or e.name not in self.SKIP_DIRS)
            ]
            for i, entry in enumerate(entries):
                last = i == len(entries) - 1
                pointer = "`-- " if last else "|-- "
                lines.append(prefix + pointer + entry.name)
                if entry.is_dir():
                    visit(entry.path, prefix + ("    " if last else "|   "), depth + 1)

        visit(root)
        return lines
