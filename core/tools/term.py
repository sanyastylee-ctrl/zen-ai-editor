from __future__ import annotations

import os
import re
import shlex
import subprocess
import time

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

    @staticmethod
    def _split_command(command: str) -> list[str]:
        repaired = RunTerminalTool._split_unquoted_windows_exe(command)
        if repaired:
            return repaired
        try:
            parts = shlex.split(command, posix=False)
        except ValueError:
            # Retry POSIX parsing for malformed-but-common quoted snippets so
            # callers get a deterministic parser error below if this also fails.
            parts = shlex.split(command, posix=True)
        return [RunTerminalTool._strip_outer_quotes(part) for part in parts]

    @staticmethod
    def _split_unquoted_windows_exe(command: str) -> list[str]:
        text = str(command or "").strip()
        if not text or text[0] in {"'", '"'}:
            return []
        match = re.match(r"^(?P<exe>[A-Za-z]:\\.*?\.exe)(?:\s+(?P<rest>.*))?$", text, re.IGNORECASE)
        if not match:
            return []
        exe = match.group("exe")
        rest = match.group("rest") or ""
        try:
            tail = shlex.split(rest, posix=False) if rest else []
        except ValueError:
            tail = shlex.split(rest, posix=True) if rest else []
        return [RunTerminalTool._strip_outer_quotes(exe)] + [
            RunTerminalTool._strip_outer_quotes(part)
            for part in tail
        ]

    @staticmethod
    def _strip_outer_quotes(arg: str) -> str:
        text = str(arg or "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            return text[1:-1]
        return text

    @staticmethod
    def _exe_name(arg: str) -> str:
        base = os.path.basename((arg or "").strip().strip('"')).lower()
        return base or (arg or "").lower()

    @classmethod
    def _normalized_args(cls, args: list[str]) -> list[str]:
        if not args:
            return []
        exe = cls._exe_name(args[0])
        if exe == "python.exe":
            exe = "python"
        return [exe] + [str(arg).lower() for arg in args[1:]]

    @staticmethod
    def _is_python_inline_code(args: list[str], index: int) -> bool:
        if index <= 0:
            return False
        exe = RunTerminalTool._exe_name(args[0])
        if exe == "python.exe":
            exe = "python"
        return exe in {"python", "py"} and index > 1 and args[index - 1] == "-c"

    def _has_blocked_shell_operator(self, args: list[str]) -> bool:
        for idx, arg in enumerate(args):
            if self._is_python_inline_code(args, idx):
                continue
            if arg in self.CONTROL_CHARS:
                return True
            if any(op in arg for op in ("&&", "||", ">", "<", "|")):
                return True
            if ";" in arg:
                return True
        return False

    def classify_command(self, command: str) -> tuple[str, str]:
        command = command.strip()
        if not command:
            return "blocked", "missing command"
        try:
            args = self._split_command(command)
        except ValueError as e:
            return "blocked", str(e)
        if not args:
            return "blocked", "empty command"
        if self._has_blocked_shell_operator(args):
            return "blocked", "shell control operators are blocked"

        lowered_args = self._normalized_args(args)
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
            return ToolResult(
                ok=False,
                title="Terminal blocked",
                output=f"[error: {reason}]",
                meta={"command": command, "cwd": self.project_root, "blocked": True, "reason": reason},
            )
        try:
            args = self._split_command(command)
        except ValueError as e:
            return ToolResult(
                ok=False,
                title="Terminal malformed",
                output=f"[error: {e}]",
                meta={"command": command, "cwd": self.project_root, "malformed": True, "reason": str(e)},
            )

        timeout = int(call.args.get("timeout", "120") or "120")
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                args,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
            )
            duration = time.perf_counter() - started
            output = proc.stdout or ""
            if len(output) > 60000:
                output = output[:60000] + "\n[output truncated]"
            return ToolResult(
                ok=proc.returncode == 0,
                title="Terminal",
                output=f"$ {command}\n{output}\n[exit {proc.returncode}]",
                meta={
                    "command": command,
                    "cwd": self.project_root,
                    "exit_code": proc.returncode,
                    "timeout": timeout,
                    "duration_sec": round(duration, 3),
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False,
                output="[error: command timed out]",
                critical=True,
                meta={
                    "command": command,
                    "cwd": self.project_root,
                    "exit_code": None,
                    "timeout": timeout,
                    "duration_sec": round(time.perf_counter() - started, 3),
                },
            )
        except Exception as e:
            return ToolResult(
                ok=False,
                title="Terminal execution failed",
                output=f"[error: {e}]",
                critical=True,
                meta={"command": command, "cwd": self.project_root, "execution_failure": True, "reason": str(e)},
            )
