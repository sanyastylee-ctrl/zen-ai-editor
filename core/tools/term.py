from __future__ import annotations

import os
import shlex
import subprocess

from .base import Tool, ToolCall, ToolResult


class RunTerminalTool(Tool):
    name = "run_terminal"
    description = "Run an approved command in the project directory."
    runs_command = True

    ALLOWED = {
        "python", "py", "pytest", "pip",
        "node", "npm", "npx", "yarn", "pnpm",
        "git",
    }
    BLOCKED_TOKENS = {
        "rm", "del", "erase", "rmdir", "remove-item", "rd",
        "format", "shutdown", "restart-computer",
    }
    CONTROL_CHARS = {"|", "&", ";", ">", "<"}

    def execute(self, call: ToolCall) -> ToolResult:
        command = call.args.get("command", "").strip()
        if not command:
            return ToolResult.error("missing command")
        if any(ch in command for ch in self.CONTROL_CHARS):
            return ToolResult.error("shell control operators are blocked")
        try:
            args = shlex.split(command, posix=os.name != "nt")
        except ValueError as e:
            return ToolResult.error(str(e))
        if not args:
            return ToolResult.error("empty command")

        exe = args[0].lower()
        if exe not in self.ALLOWED:
            return ToolResult.error(f"command not whitelisted: {args[0]}")
        lowered = {a.lower() for a in args}
        if lowered & self.BLOCKED_TOKENS:
            return ToolResult.error("destructive command token is blocked", critical=True)

        try:
            proc = subprocess.run(
                args,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=int(call.args.get("timeout", "120") or "120"),
                shell=False,
            )
            output = proc.stdout or ""
            if len(output) > 60000:
                output = output[:60000] + "\n[output truncated]"
            return ToolResult(
                ok=proc.returncode == 0,
                title="Terminal",
                output=f"$ {command}\n{output}\n[exit {proc.returncode}]",
                meta={"command": command, "exit_code": proc.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error("command timed out", critical=True)
        except Exception as e:
            return ToolResult.error(str(e), critical=True)
