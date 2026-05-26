from __future__ import annotations

import os
import re
import subprocess

from .base import Tool, ToolCall, ToolResult


class ApplyPatchTool(Tool):
    name = "apply_patch"
    description = "Apply a unified diff to project files."
    mutates_project = True

    PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:a/|b/)?(.+)$")

    def execute(self, call: ToolCall) -> ToolResult:
        patch = call.args.get("patch", "")
        if not patch.strip():
            return ToolResult.error("missing patch")
        try:
            self._validate_patch_paths(patch)
            proc = subprocess.run(
                ["git", "apply", "--whitespace=nowarn"],
                cwd=self.project_root,
                input=patch,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                shell=False,
            )
            output = proc.stdout or ""
            return ToolResult(
                ok=proc.returncode == 0,
                title="Apply patch",
                output=output.strip() or f"[exit {proc.returncode}]",
                meta={"exit_code": proc.returncode},
                critical=proc.returncode != 0,
            )
        except ValueError:
            return ToolResult.error("path outside project")
        except Exception as e:
            return ToolResult.error(str(e), critical=True)

    def preview(self, call: ToolCall) -> str:
        return call.args.get("patch", "")

    def _validate_patch_paths(self, patch: str) -> None:
        for line in patch.splitlines():
            match = self.PATH_RE.match(line)
            if not match:
                continue
            path = match.group(1).strip()
            if path == "/dev/null":
                continue
            # Strip optional timestamps after filenames in classic unified diffs.
            path = path.split("\t", 1)[0]
            if path.startswith('"') and path.endswith('"'):
                path = path[1:-1]
            self.safe_path(path, allow_missing=True)
