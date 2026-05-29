from __future__ import annotations

import hashlib
import os

from core.diagnostics import write_log

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
            content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
            file_size = os.path.getsize(path)
            preview = content[:200].replace("\r", "\\r").replace("\n", "\\n")
            write_log(
                "[agent_read_file] "
                f"source=filesystem path={path} exists=True size={file_size} "
                f"preview={preview!r}"
            )
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
                meta={
                    "path": rel,
                    "absolute_path": path,
                    "source": "filesystem",
                    "size": file_size,
                    "content_hash": content_hash,
                    "lines": content.count("\n") + 1,
                },
            )
        except ValueError:
            return ToolResult.error("path outside project")
        except Exception as e:
            return ToolResult.error(str(e))
