"""
Tool-calling agent loop for coder profiles.

The local llama.cpp completion API is model-dependent, so tools are requested
with a small XML dialect in plain text and parsed after each model turn.
"""

from __future__ import annotations

import html
import os
import re
import threading
import time

from PyQt6.QtCore import QThread, pyqtSignal

from core.chat_templates import detect_template, format_prompt
from core.diagnostics import acceleration_warning, write_log
from core.model_manager import LLAMA_AVAILABLE, ModelManager
from core.paths import resolve_model_path
from core.profiles import AIProfile, ChatTemplate
from core.token_budget import TokenBudget
from core.tools import ToolCall, ToolResult, default_tools


MAX_AGENT_STEPS = 5
MAX_TOOL_CALLS = 10
MAX_GENERATION_SECONDS = 180
MAX_CONTEXT_CHARS = 240_000
MAX_CRITICAL_ERRORS = 3

TOOL_BLOCK_RE = re.compile(
    r"<tool\s+name=[\"'](?P<name>[^\"']+)[\"']\s*>(?P<body>.*?)</tool>",
    re.IGNORECASE | re.DOTALL,
)
ARG_RE = re.compile(
    r"<(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)>(?P<value>.*?)</(?P=key)>",
    re.IGNORECASE | re.DOTALL,
)
LEGACY_FILE_RE = re.compile(
    r"\[(?:CREATE_FILE|FILE):\s*(?P<path>[^\]\n]+?)\s*\]\s*"
    r"(?P<content>.*?)\[/(?:CREATE_FILE|FILE)\]",
    re.IGNORECASE | re.DOTALL,
)
LEGACY_RUN_RE = re.compile(r"\[RUN:\s*(?P<command>[^\]\n]+?)\s*\]", re.IGNORECASE)
LEGACY_START_RE = re.compile(r"\[(?:CREATE_FILE|FILE|RUN):", re.IGNORECASE)
TOOL_START_RE = re.compile(r"<tool\b", re.IGNORECASE)
FENCED_CONTENT_RE = re.compile(
    r"^\s*```[a-zA-Z0-9_+\-]*\s*\n(?P<content>.*?)\n```\s*$",
    re.DOTALL,
)
INLINE_CODE_RE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")

REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "write_file": ("path", "content"),
    "create_file": ("path", "content"),
    "edit_file": ("path", "old_str", "new_str"),
    "apply_patch": ("patch",),
    "patch_file": ("patch",),
    "run_terminal": ("command",),
    "run_command": ("command",),
    "read_file": ("path",),
    "search_files": ("query",),
}


AGENT_TOOL_INSTRUCTIONS = """
You can operate on the current project by emitting XML tool calls.

Use only the XML tool protocol below. This overrides any earlier [CREATE_FILE]
or [RUN:] instructions saved in an older profile. Do not wrap tool calls in
markdown fences.

Available tools:

<tool name="read_file">
<path>core/profiles.py</path>
</tool>

<tool name="list_files">
<path>core/</path>
<max_depth>3</max_depth>
</tool>

<tool name="search_files">
<query>ProfileManager</query>
<path>.</path>
</tool>

<tool name="write_file">
<path>core/new_module.py</path>
<content>
def hello():
    return "hello"
</content>
</tool>

To create a file and write it in one operation:

<tool name="write_file">
<path>hello.py</path>
<content>print("hi")
</content>
</tool>

<tool name="edit_file">
<path>hello.py</path>
<old_str>print("hi")</old_str>
<new_str>print("hello world")</new_str>
</tool>

<tool name="run_terminal">
<command>python hello.py</command>
</tool>

<tool name="apply_patch">
<patch>
--- a/hello.py
+++ b/hello.py
@@
-print("hi")
+print("hello world")
</patch>
</tool>

Rules:
1. First inspect files before changing behavior.
2. To create or replace a file, call write_file with BOTH path and complete content.
3. To change an existing file, read it first, then use edit_file or apply_patch.
4. To run a safe verification command, use run_terminal.
5. Use tools only when you need project facts or verification.
6. After a tool result, continue from the new evidence.
7. When you are done, answer normally without any <tool> block.
8. Tool paths must be project-relative.
"""


def _content_value(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    fenced = FENCED_CONTENT_RE.match(value)
    return fenced.group("content") if fenced else value


def _inline_code_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in INLINE_CODE_RE.finditer(text)]


def _inside_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _find_dangling_tool_start(text: str, patterns=None):
    spans = _inline_code_spans(text)
    patterns = patterns or (TOOL_START_RE, LEGACY_START_RE)
    matches = []
    for pattern in patterns:
        matches.extend(pattern.finditer(text))
    matches.sort(key=lambda match: match.start())
    for match in matches:
        if not _inside_spans(match.start(), spans):
            return match
    return None


def parse_tool_response(text: str) -> tuple[list[ToolCall], list[str]]:
    calls: list[ToolCall] = []
    errors: list[str] = []
    matched_spans: list[tuple[int, int]] = []
    for match in TOOL_BLOCK_RE.finditer(text):
        matched_spans.append(match.span())
        body = match.group("body")
        args: dict[str, str] = {}
        for arg_match in ARG_RE.finditer(body):
            key = arg_match.group("key").strip().lower()
            raw_value = arg_match.group("value")
            value = _content_value(raw_value) if key in {"content", "old_str", "new_str", "patch"} else html.unescape(raw_value.strip())
            args[key] = value
        call = ToolCall(
            name=match.group("name").strip().lower(),
            args=args,
            raw=match.group(0),
        )
        missing = [key for key in REQUIRED_ARGS.get(call.name, ()) if key not in args]
        if missing:
            errors.append(f"{call.name}: missing <{'>, <'.join(missing)}>")
        else:
            calls.append(call)

    residual = text
    for start, end in reversed(matched_spans):
        residual = residual[:start] + residual[end:]
    if _find_dangling_tool_start(residual, (TOOL_START_RE,)):
        errors.append("truncated or malformed XML <tool> block")

    # Backward compatibility for saved Agent-Coder prompts from before XML tools.
    for match in LEGACY_FILE_RE.finditer(residual):
        calls.append(
            ToolCall(
                name="write_file",
                args={
                    "path": match.group("path").strip(),
                    "content": _content_value(match.group("content")),
                },
                raw=match.group(0),
            )
        )
    legacy_without_files = LEGACY_FILE_RE.sub("", residual)
    for match in LEGACY_RUN_RE.finditer(legacy_without_files):
        calls.append(
            ToolCall(
                name="run_terminal",
                args={"command": html.unescape(match.group("command").strip())},
                raw=match.group(0),
            )
        )
    legacy_remainder = LEGACY_RUN_RE.sub("", legacy_without_files)
    if _find_dangling_tool_start(legacy_remainder, (LEGACY_START_RE,)):
        errors.append("truncated or malformed legacy agent action")
    return calls, errors


def parse_tools(text: str) -> list[ToolCall]:
    calls, _ = parse_tool_response(text)
    return calls


def strip_tool_blocks(text: str) -> str:
    visible = TOOL_BLOCK_RE.sub("", text)
    visible = LEGACY_FILE_RE.sub("", visible)
    visible = LEGACY_RUN_RE.sub("", visible)
    dangling = _find_dangling_tool_start(visible)
    if dangling:
        visible = visible[:dangling.start()]
    if re.fullmatch(r"\s*```(?:xml|tool)?\s*```\s*", visible, re.IGNORECASE):
        return ""
    return visible.strip()


class AgentWorker(QThread):
    chunk_received = pyqtSignal(str)
    status = pyqtSignal(str)
    tool_started = pyqtSignal(dict)
    tool_finished = pyqtSignal(dict)
    confirmation_requested = pyqtSignal(dict)
    model_loading = pyqtSignal(str)
    model_loaded = pyqtSignal(str, bool, str)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        profile: AIProfile,
        user_message: str,
        code_context: str = "",
        history: list[tuple[str, str]] | None = None,
        project_root: str | None = None,
        terminal_history: list[str] | None = None,
        session_summary: str = "",
        confirmation_policy: str = "confirm_changes",
        max_agent_steps: int = MAX_AGENT_STEPS,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_generation_seconds: int = MAX_GENERATION_SECONDS,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.profile = profile
        self.user_message = user_message
        self.code_context = code_context
        self.history = history or []
        self.project_root = os.path.realpath(project_root or os.getcwd())
        self.terminal_history = terminal_history or []
        self.session_summary = session_summary
        self.confirmation_policy = confirmation_policy
        self.max_agent_steps = max(1, int(max_agent_steps or MAX_AGENT_STEPS))
        self.max_tool_calls = max(1, int(max_tool_calls or MAX_TOOL_CALLS))
        self.max_generation_seconds = max(1, int(max_generation_seconds or MAX_GENERATION_SECONDS))
        self.max_context_chars = max(1000, int(max_context_chars or MAX_CONTEXT_CHARS))
        self._stop = False
        self._tool_calls_used = 0
        self._generation_interrupted_reason = ""

        self._tools = default_tools(self.project_root)
        self._cb_start = None
        self._cb_finish = None
        self._confirm_events: dict[str, threading.Event] = {}
        self._confirm_results: dict[str, bool] = {}

    def stop(self) -> None:
        self._stop = True
        for event in self._confirm_events.values():
            event.set()

    def resolve_confirmation(self, call_id: str, accepted: bool) -> None:
        self._confirm_results[call_id] = accepted
        event = self._confirm_events.get(call_id)
        if event:
            event.set()

    def run(self) -> None:
        if not LLAMA_AVAILABLE:
            self.chunk_received.emit("\n[Ошибка: llama-cpp-python не установлен]\n")
            self.finished_signal.emit()
            return
        if not self.profile.model_file:
            self.chunk_received.emit("\n[Не выбран файл модели в настройках]\n")
            self.finished_signal.emit()
            return

        mm = ModelManager.instance()
        model_path = resolve_model_path(self.profile.model_file)

        def _on_start(path: str) -> None:
            if path == model_path:
                self.model_loading.emit(path)

        def _on_finish(path: str, ok: bool, err) -> None:
            if path == model_path:
                self.model_loaded.emit(path, ok, err or "")

        self._cb_start = _on_start
        self._cb_finish = _on_finish
        mm.on_load_start(_on_start)
        mm.on_load_finish(_on_finish)

        try:
            warning = acceleration_warning(self.profile, model_path)
            if warning:
                self.status.emit("⚠ CPU backend")
                self.chunk_received.emit(f"\n[Предупреждение: {warning}]\n")

            model = mm.get_model(
                path=model_path,
                n_ctx=self.profile.n_ctx,
                n_gpu_layers=self.profile.n_gpu_layers,
            )
            transcript = self._initial_transcript()
            critical_errors = 0

            for iteration in range(1, self.max_agent_steps + 1):
                if self._stop:
                    self.chunk_received.emit("\n[остановлено]")
                    break
                if self._over_context_budget(transcript):
                    self.chunk_received.emit("\n[Агент остановлен: превышен бюджет контекста]\n")
                    break

                self.status.emit(f"Агент: итерация {iteration}/{self.max_agent_steps}")
                response = self._complete(model, transcript)

                visible = strip_tool_blocks(response)
                if visible:
                    self.chunk_received.emit(visible + "\n")

                calls, parser_errors = parse_tool_response(response)
                for error in parser_errors:
                    self._report_parser_error(error)
                    transcript.append(f"Tool parser error:\n[error: {error}]")

                if self._generation_interrupted_reason:
                    write_log(f"[agent_generation_stopped] {self._generation_interrupted_reason}")
                    if calls:
                        self.chunk_received.emit(
                            "\n[Действия tool не выполнены: ответ модели был прерван]\n"
                        )
                    self.chunk_received.emit(
                        f"\n[Агент остановлен: {self._generation_interrupted_reason}; частичный ответ сохранён]\n"
                    )
                    break

                if not calls:
                    if parser_errors:
                        continue
                    break

                transcript.append(f"Assistant tool request:\n{response}")
                for call in calls:
                    if self._tool_calls_used >= self.max_tool_calls:
                        self.chunk_received.emit("\n[Агент остановлен: достигнут лимит tool calls]\n")
                        return
                    result = self._execute_tool(call)
                    self._tool_calls_used += 1
                    transcript.append(
                        f"Tool result for {call.name}:\n{result.output}"
                    )
                    tool = self._tools.get(call.name)
                    if result.ok and tool is not None and tool.mutates_project:
                        test_result = self._run_auto_tests_after_change()
                        if test_result is not None:
                            transcript.append(
                                f"Automatic test result:\n{test_result.output}"
                            )
                    if result.critical:
                        critical_errors += 1
                    else:
                        critical_errors = 0
                    if critical_errors >= MAX_CRITICAL_ERRORS:
                        self.chunk_received.emit(
                            "\n[Агент остановлен: 3 критические ошибки tool подряд]\n"
                        )
                        return
            else:
                self.chunk_received.emit("\n[Агент остановлен: достигнут лимит итераций]\n")
        except Exception as e:
            self.chunk_received.emit(f"\n[Ошибка агента: {e}]\n")
        finally:
            if self._cb_start:
                mm.off_load_start(self._cb_start)
            if self._cb_finish:
                mm.off_load_finish(self._cb_finish)
            self.finished_signal.emit()

    def _initial_transcript(self) -> list[str]:
        items = [f"User task:\n{self.user_message}"]
        if self.code_context.strip():
            items.append(f"Current editor/project context:\n{self.code_context}")
        if self.terminal_history:
            items.append("Recent terminal output:\n" + "\n\n".join(self.terminal_history[-5:]))
        if self.session_summary.strip():
            items.append("Session changes summary:\n" + self.session_summary.strip())
        return items

    def _complete(self, model, transcript: list[str]) -> str:
        self._generation_interrupted_reason = ""
        system = self._agent_system_prompt()
        user = "\n\n---\n\n".join(transcript)

        template = self.profile.chat_template
        if template == ChatTemplate.AUTO:
            template = detect_template(self.profile.model_file)
        prompt = format_prompt(template, system, user, self.history[-8:])

        stream = model(
            prompt,
            max_tokens=self.profile.max_tokens,
            temperature=self.profile.temperature,
            top_p=self.profile.top_p,
            top_k=self.profile.top_k,
            repeat_penalty=self.profile.repeat_penalty,
            stop=self.profile.stop_sequences or self._default_stops(),
            stream=True,
        )

        chunks: list[str] = []
        started = time.monotonic()
        for chunk in stream:
            if self._stop:
                self._generation_interrupted_reason = "остановлено пользователем"
                break
            if time.monotonic() - started > self.max_generation_seconds:
                self._stop = True
                self._generation_interrupted_reason = "превышен лимит времени генерации"
                break
            text = chunk["choices"][0]["text"]
            if text:
                chunks.append(text)
        return "".join(chunks)

    def _over_context_budget(self, transcript: list[str]) -> bool:
        system = self._agent_system_prompt()
        text = system + "\n\n" + "\n\n".join(transcript)
        if len(text) > self.max_context_chars:
            return True
        return TokenBudget.estimate(text) > (self.profile.n_ctx - self.profile.max_tokens)

    def _agent_system_prompt(self) -> str:
        base = self.profile.system_prompt or ""
        return f"{base}\n\n{AGENT_TOOL_INSTRUCTIONS}\n\n{self._project_tree_context()}"

    def _project_tree_context(self) -> str:
        lines = ["Current project tree (depth 3):"]
        skip = {".git", "__pycache__", ".venv", "venv", "node_modules", ".zen_ai"}

        def visit(path: str, prefix: str = "", depth: int = 0) -> None:
            if depth > 3:
                return
            try:
                entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
            except OSError:
                return
            entries = [
                e for e in entries
                if not e.name.startswith(".") and (not e.is_dir() or e.name not in skip)
            ]
            for i, entry in enumerate(entries):
                last = i == len(entries) - 1
                lines.append(prefix + ("`-- " if last else "|-- ") + entry.name)
                if entry.is_dir():
                    visit(entry.path, prefix + ("    " if last else "|   "), depth + 1)

        visit(self.project_root)
        return "\n".join(lines)

    def _execute_tool(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        payload = {"id": call.id, "name": call.name, "args": call.args}
        if tool is None:
            result = ToolResult.error(f"unknown tool: {call.name}", critical=True)
            self.tool_finished.emit({**payload, "ok": False, "output": result.output})
            write_log(f"[agent_tool_error] {call.name}: {result.output}")
            return result

        self.tool_started.emit(payload)
        if self.confirmation_policy == "read_only" and (tool.mutates_project or tool.runs_command):
            result = ToolResult.error("tool blocked by read-only confirmation policy")
            self.tool_finished.emit({**payload, "ok": False, "output": result.output})
            write_log(f"[agent_tool_error] {call.name}: {result.output}")
            return result

        if self._needs_confirmation(tool):
            preview = tool.preview(call)
            event = threading.Event()
            self._confirm_events[call.id] = event
            self.confirmation_requested.emit({**payload, "preview": preview})
            event.wait()
            self._confirm_events.pop(call.id, None)
            accepted = self._confirm_results.pop(call.id, False)
            if self._stop:
                result = ToolResult.error("stopped")
                self.tool_finished.emit({**payload, "ok": False, "output": result.output})
                write_log(f"[agent_tool_error] {call.name}: {result.output}")
                return result
            if not accepted:
                result = ToolResult(ok=False, output="[user rejected]")
                self.tool_finished.emit({**payload, "ok": False, "output": result.output})
                write_log(f"[agent_tool_rejected] {call.name}")
                return result

        result = tool.execute(call)
        self.tool_finished.emit(
            {
                **payload,
                "ok": result.ok,
                "output": result.output,
                "title": result.title,
                "meta": result.meta,
            }
        )
        if not result.ok:
            write_log(f"[agent_tool_error] {call.name}: {result.output}")
        return result

    def _report_parser_error(self, error: str) -> None:
        call = ToolCall(name="parser_error", args={"error": error}, raw="")
        payload = {"id": call.id, "name": call.name, "args": call.args}
        result = ToolResult.error(error)
        self.tool_started.emit(payload)
        self.tool_finished.emit({**payload, "ok": False, "output": result.output, "title": "Tool parser"})
        self.chunk_received.emit(f"\n[Ошибка разбора tool call: {error}]\n")
        write_log(f"[agent_parser_error] {error}")

    def _run_auto_tests_after_change(self) -> ToolResult | None:
        command = self._detect_test_command()
        if not command:
            return None
        call = ToolCall(
            name="run_terminal",
            args={"command": command, "timeout": "180"},
            raw=f"<tool name=\"run_terminal\"><command>{command}</command></tool>",
        )
        self.status.emit(f"Агент: проверка ({command})")
        return self._execute_tool(call)

    def _detect_test_command(self) -> str:
        if (
            os.path.exists(os.path.join(self.project_root, "pytest.ini"))
            or os.path.exists(os.path.join(self.project_root, "pyproject.toml"))
            or os.path.isdir(os.path.join(self.project_root, "tests"))
        ):
            return "python -m pytest"
        package_json = os.path.join(self.project_root, "package.json")
        if os.path.exists(package_json):
            try:
                with open(package_json, "r", encoding="utf-8") as f:
                    text = f.read()
                if "\"test\"" in text:
                    return "npm test"
            except OSError:
                pass
        return ""

    def _needs_confirmation(self, tool) -> bool:
        if self.confirmation_policy == "confirm_all":
            return True
        if self.confirmation_policy == "auto_confirm":
            return bool(tool.runs_command)
        if self.confirmation_policy == "confirm_changes":
            return bool(tool.mutates_project or tool.runs_command)
        return False

    @staticmethod
    def _default_stops() -> list[str]:
        return [
            "<|im_end|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "<end_of_turn>",
        ]
