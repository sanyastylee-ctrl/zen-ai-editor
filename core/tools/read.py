from __future__ import annotations

import os

from .base import Tool, ToolCall, ToolResult


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file from the current project."

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            path = self.safe_path(call.args.get("path", ""))
            if not os.path.isfile(path):
                return ToolResult.error("not a file")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            max_chars = int(call.args.get("max_chars", "60000") or "60000")
            truncated = len(content) > max_chars
            if truncated:
                content = content[:max_chars]
            suffix = "\n\n[truncated]" if truncated else ""
            rel = self.relpath(path)
            return ToolResult(
                ok=True,
                title=f"Read: {rel}",
                output=f"# {rel}\n{content}{suffix}",
                meta={"path": rel, "lines": content.count("\n") + 1},
            )
        except ValueError:
            return ToolResult.error("path outside project")
        except Exception as e:
            return ToolResult.error(str(e))
