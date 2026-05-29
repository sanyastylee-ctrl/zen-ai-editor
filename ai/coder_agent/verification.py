from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandGoal:
    raw: str
    normalized: str
    mode: str = "exact"  # exact literal command or semantic verb goal
    source: str = "user"

    def matches(self, command: str) -> bool:
        command_key = normalize_command(command)
        if self.mode == "semantic":
            return command_key == self.normalized or command_key.startswith(self.normalized + " ")
        return command_key == self.normalized

    def example(self) -> str:
        if self.mode != "semantic":
            return self.raw
        if self.normalized.endswith(" add"):
            return self.raw + ' "first task"'
        if self.normalized.endswith(" done"):
            return self.raw + " 1"
        return self.raw


def normalize_command(value: str) -> str:
    normalized = re.sub(r"[\"']", "", value or "").replace("\\", "/").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"^(py|py\.exe|python\.exe)\s+", "python ", normalized)
    normalized = normalized.replace("python ./", "python ")
    return normalized


def extract_command_goals(text: str) -> list[CommandGoal]:
    value = text or ""
    normalized_text = value.lower()
    goals: list[CommandGoal] = []

    for match in re.finditer(
        r"(?im)^\s*(?:[-*]\s*)?`?(?P<command>(?:python|py)(?:\.exe)?\s+"
        r"-m\s+[\w.]+\s+[^\r\n`]+?)`?\s*$",
        value,
        re.IGNORECASE,
    ):
        command = match.group("command").strip()
        command = re.sub(r"^(py|py\.exe|python\.exe)\s+", "python ", command, flags=re.IGNORECASE)
        command = re.sub(r"\s+", " ", command.replace("\\", "/"))
        goal = CommandGoal(raw=command, normalized=normalize_command(command), mode="exact", source="literal")
        if goal.normalized not in {item.normalized for item in goals}:
            goals.append(goal)

    for match in re.finditer(
        r"(?im)^\s*(?:[-*]\s*)?`?(?P<command>(?:python|py)(?:\.exe)?\s+"
        r"-m\s+(?:pytest|unittest)(?:\s+[^\r\n`]*)?)`?\s*$",
        value,
        re.IGNORECASE,
    ):
        command = match.group("command").strip()
        command = re.sub(r"^(py|py\.exe|python\.exe)\s+", "python ", command, flags=re.IGNORECASE)
        command = re.sub(r"\s+", " ", command.replace("\\", "/"))
        goal = CommandGoal(raw=command, normalized=normalize_command(command), mode="exact", source="literal")
        if goal.normalized not in {item.normalized for item in goals}:
            goals.append(goal)

    for match in re.finditer(
        r"(?im)^\s*(?:[-*]\s*)?`?(?P<command>(?:python|py)(?:\.exe)?\s+"
        r"\.?(?P<script>[\w./\\-]+\.py)\s+(?P<verb>add|list|done|clear)"
        r"(?:[ \t]+[^\r\n`]+)?)`?\s*$",
        value,
        re.IGNORECASE,
    ):
        script = match.group("script").replace("\\", "/").lstrip("./")
        command = match.group("command").strip()
        command = re.sub(r"^(py|py\.exe|python\.exe)\s+", "python ", command, flags=re.IGNORECASE)
        command = command.replace("\\", "/").replace("python ./", "python ")
        command = re.sub(r"\s+", " ", command)
        raw = re.sub(
            r"^python\s+[\w./-]+\.py\s+",
            f"python {script} ",
            command,
            count=1,
            flags=re.IGNORECASE,
        )
        goal = CommandGoal(raw=raw, normalized=normalize_command(raw), mode="exact", source="literal")
        if goal.normalized not in {item.normalized for item in goals}:
            goals.append(goal)

    has_cli_words = all(word in normalized_text for word in ("add", "list", "done", "clear"))
    looks_like_notes_cli = any(
        marker in normalized_text
        for marker in ("task notes", "todo", "notes cli", "notes.json", "замет", "заметки")
    )
    mentions_main = "main.py" in normalized_text or "python-проект" in normalized_text or "python project" in normalized_text
    if has_cli_words and looks_like_notes_cli and mentions_main:
        for raw in (
            "python main.py add",
            "python main.py list",
            "python main.py done",
            "python main.py clear",
        ):
            goal = CommandGoal(raw=raw, normalized=normalize_command(raw), mode="semantic", source="inferred")
            already_covered = any(
                item.normalized == goal.normalized
                or item.normalized.startswith(goal.normalized + " ")
                for item in goals
            )
            if not already_covered:
                goals.append(goal)
    looks_like_calculator_divide = (
        "divide" in normalized_text
        and ("calculator" in normalized_text or "калькулятор" in normalized_text)
        and ("cli" in normalized_text or "app.cli" in normalized_text)
    )
    if looks_like_calculator_divide:
        for raw in (
            "python -m app.cli add 2 3",
            "python -m app.cli subtract 5 2",
            "python -m app.cli multiply 4 3",
            "python -m app.cli divide 10 2",
            "python -m app.cli divide 10 0",
        ):
            goal = CommandGoal(raw=raw, normalized=normalize_command(raw), mode="exact", source="inferred")
            if goal.normalized not in {item.normalized for item in goals}:
                goals.append(goal)
    return goals


@dataclass(frozen=True)
class TracebackInfo:
    error_type: str
    relevant_files: list[str]

    def summary(self) -> str:
        files = ", ".join(self.relevant_files) if self.relevant_files else "unknown location"
        return f"{self.error_type} in {files}"


def parse_traceback(output: str, project_root: str) -> TracebackInfo | None:
    """Extracts error type and relevant project files from a Python traceback."""
    if not output or "Traceback (most recent call last)" not in output:
        return None

    lines = output.splitlines()
    error_type = "Unknown Error"
    
    # Ищем тип ошибки (обычно последняя непустая строка без отступов)
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("File ") and ":" in line:
            error_type = line.split(":")[0].strip()
            break

    # Ищем затронутые файлы
    files = []
    for line in lines:
        match = re.search(r'File "([^"]+)"', line)
        if match:
            files.append(match.group(1))

    # Оставляем только файлы, которые лежат внутри проекта (игнорируем системные либы)
    relevant = []
    project_root_normalized = project_root.replace("\\", "/").lower()
    
    for f in files:
        f_norm = f.replace("\\", "/").lower()
        if project_root_normalized in f_norm or not os.path.isabs(f):
            # Сохраняем относительный путь для чистоты
            rel = os.path.relpath(f, project_root) if os.path.isabs(f) else f
            if rel not in relevant:
                relevant.append(rel)

    return TracebackInfo(error_type=error_type, relevant_files=relevant)
