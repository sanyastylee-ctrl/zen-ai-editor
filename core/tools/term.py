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
        "powershell", "powershell.exe", "pwsh", "pwsh.exe",
    }
    BLOCKED_TOKENS = {
        "rm", "del", "erase", "rmdir", "remove-item", "rd",
        "format", "shutdown", "restart-computer",
    }
    CONTROL_CHARS = {"|", "&", ";", ">", "<"}
    INSTALL_COMMANDS = {
        ("pip", "install"),
        ("python", "-m", "pip", "install"),
        ("py", "-m", "pip", "install"),
        ("npm", "install"),
        ("npm", "i"),
        ("yarn", "add"),
        ("pnpm", "add"),
        ("npx",),
    }
    GIT_NEEDS_CONFIRMATION = {"reset", "clean", "checkout", "switch", "restore", "rebase"}

    def classify_command(self, command: str) -> tuple[str, str]:
        command = command.strip()
        if not command:
            return "blocked", "missing command"
        if any(ch in command for ch in self.CONTROL_CHARS):
            return "blocked", "shell control operators are blocked"
        try:
            args = shlex.split(command, posix=os.name != "nt")
        except ValueError as e:
            return "blocked", str(e)
        if not args:
            return "blocked", "empty command"

        lowered_args = [arg.lower() for arg in args]
        exe = lowered_args[0]
        if exe not in self.ALLOWED:
            return "blocked", f"command not whitelisted: {args[0]}"
        if set(lowered_args) & self.BLOCKED_TOKENS:
            return "blocked", "destructive command token is blocked"
        if exe in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
            return "needs_confirmation", "PowerShell commands require confirmation"
        if exe == "git" and len(lowered_args) > 1 and lowered_args[1] in self.GIT_NEEDS_CONFIRMATION:
            return "needs_confirmation", f"git {lowered_args[1]} requires confirmation"
        for prefix in self.INSTALL_COMMANDS:
            if tuple(lowered_args[:len(prefix)]) == prefix:
                return "needs_confirmation", "install commands require confirmation"
        return "safe", "safe command"

    def execute(self, call: ToolCall) -> ToolResult:
        command = call.args.get("command", "").strip()
        safety, reason = self.classify_command(command)
        if safety == "blocked":
            return ToolResult.error(reason)
        try:
            args = shlex.split(command, posix=os.name != "nt")
        except ValueError as e:
            return ToolResult.error(str(e))

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
