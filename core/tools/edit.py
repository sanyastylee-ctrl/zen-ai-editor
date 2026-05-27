from __future__ import annotations

import difflib

from .base import Tool, ToolCall, ToolResult


class EditFileTool(Tool):
    name = "edit_file"
    description = "Replace one exact old_str block in a project file with new_str."
    mutates_project = True

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            if "new_str" not in call.args:
                return ToolResult.error("missing new_str", critical=True)
            path = self.safe_path(call.args.get("path", ""))
            old_str = call.args.get("old_str", "")
            new_str = call.args.get("new_str", "")
            if not old_str:
                return ToolResult.error("missing old_str", critical=True)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                old = f.read()
            count = old.count(old_str)
            if count == 0:
                return ToolResult.error("old_str not found")
            if count > 1:
                return ToolResult.error("old_str is not unique")
            new = old.replace(old_str, new_str, 1)
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(new)
            rel = self.relpath(path)
            return ToolResult(
                ok=True,
                title=f"Edit: {rel}",
                output=f"[ok: edited {rel}]",
                meta={"path": rel},
            )
        except ValueError:
            return ToolResult.error("path outside project")
        except Exception as e:
            return ToolResult.error(str(e), critical=True)

    def preview(self, call: ToolCall) -> str:
        try:
            path = self.safe_path(call.args.get("path", ""))
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                old = f.read()
            new = old.replace(call.args.get("old_str", ""), call.args.get("new_str", ""), 1)
            rel = self.relpath(path)
            return "".join(difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            ))
        except Exception as e:
            return f"[preview error: {e}]"
