from __future__ import annotations

import difflib
import os

from .base import Tool, ToolCall, ToolResult


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write complete UTF-8 file content inside the current project."
    mutates_project = True

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            path = self.safe_path(call.args.get("path", ""), allow_missing=True)
            content = call.args.get("content", "")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(content)
            rel = self.relpath(path)
            return ToolResult(
                ok=True,
                title=f"Write: {rel}",
                output=f"[ok: wrote {rel}, {len(content)} chars]",
                meta={"path": rel, "chars": len(content)},
            )
        except ValueError:
            return ToolResult.error("path outside project")
        except Exception as e:
            return ToolResult.error(str(e), critical=True)

    def preview(self, call: ToolCall) -> str:
        try:
            path = self.safe_path(call.args.get("path", ""), allow_missing=True)
            old = ""
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    old = f.read()
            new = call.args.get("content", "")
            rel = self.relpath(path)
            return "".join(difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            ))
        except Exception as e:
            return f"[preview error: {e}]"
