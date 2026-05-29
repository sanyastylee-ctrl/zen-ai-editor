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
                rel = self.relpath(path)
                if new_str and new_str in old:
                    return ToolResult(
                        ok=True,
                        title=f"Edit: {rel}",
                        output=f"[ok: {rel} already contains requested text]",
                        meta={"path": rel, "idempotent": True},
                    )
                return ToolResult.error("old_str not found")
            if count > 1:
                return ToolResult.error("old_str is not unique")
            start = old.find(old_str)
            new_str = self._preserve_line_indent(old, start, old_str, new_str)
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

    @staticmethod
    def _preserve_line_indent(text: str, start: int, old_str: str, new_str: str) -> str:
        if start < 0 or "\n" not in new_str:
            return new_str
        line_start = text.rfind("\n", 0, start) + 1
        prefix = text[line_start:start]
        if not prefix or not prefix.isspace() or old_str.startswith(prefix):
            return new_str
        return new_str.replace("\n", "\n" + prefix)

    def preview(self, call: ToolCall) -> str:
        try:
            path = self.safe_path(call.args.get("path", ""))
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                old = f.read()
            old_str = call.args.get("old_str", "")
            new_str = call.args.get("new_str", "")
            start = old.find(old_str)
            new_str = self._preserve_line_indent(old, start, old_str, new_str)
            new = old.replace(old_str, new_str, 1)
            rel = self.relpath(path)
            return "".join(difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            ))
        except Exception as e:
            return f"[preview error: {e}]"
