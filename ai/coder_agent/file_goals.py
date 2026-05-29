from __future__ import annotations

import ast
import os
import re
from dataclasses import asdict
from pathlib import Path

from .state import FileGoal, FileGoalStatus


FILE_PATH_RE = re.compile(
    r"(?<![\w/.-])([\w.-]+(?:[/\\][\w.-]+)*\."
    r"(?:py|md|txt|json|toml|yaml|yml|html|css|js|ts|tsx|jsx|ini|cfg))",
    re.IGNORECASE,
)

PLACEHOLDER_MARKERS = (
    "rest of the code",
    "remaining code",
    "same as before",
    "omitted",
    "TODO: implement",
    "TODO implement",
    "implementation here",
    "placeholder",
    "здесь будет",
    "остальной код",
    "оставшийся код",
    "реализация позже",
    "дописать позже",
    "pass  # TODO",
    "raise NotImplementedError",
    "NotImplementedError",
)


def normalize_goal_path(path: str) -> str:
    return path.strip().replace("\\", "/").lstrip("./")


def extract_file_goals(text: str) -> list[FileGoal]:
    """Extract concrete file goals from a user task or checkpoint text."""
    seen: set[str] = set()
    goals: list[FileGoal] = []
    for match in FILE_PATH_RE.finditer(text or ""):
        path = normalize_goal_path(match.group(1))
        if not path or path in seen or _looks_like_noise_path(path):
            continue
        seen.add(path)
        goals.append(
            FileGoal(
                path=path,
                purpose=_infer_purpose(path),
                expected_symbols=_infer_expected_symbols(path, text or ""),
            )
        )
    _apply_simple_dependencies(goals)
    return goals


def serialize_file_goals(goals: list[FileGoal]) -> list[dict]:
    items: list[dict] = []
    for goal in goals:
        data = asdict(goal)
        data["status"] = goal.status.value
        items.append(data)
    return items


def deserialize_file_goals(items: list[dict] | None) -> list[FileGoal]:
    goals: list[FileGoal] = []
    for item in items or []:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        data = dict(item)
        try:
            data["status"] = FileGoalStatus(str(data.get("status") or FileGoalStatus.PLANNED.value))
        except ValueError:
            data["status"] = FileGoalStatus.PLANNED
        goals.append(FileGoal(**data))
    return goals


def detect_lazy_placeholders(content: str, *, path: str = "", purpose: str = "") -> list[str]:
    if _empty_init_allowed(path, content):
        return []
    markers: list[str] = []
    lower = content.lower()
    for marker in PLACEHOLDER_MARKERS:
        if marker.lower() in lower:
            if "notimplementederror" in marker.lower() and _abstract_context_allowed(content, purpose):
                continue
            markers.append(marker)
    if re.search(r"(?m)^\s*(?:\.\.\.|…)\s*(?:#.*)?$", content):
        markers.append("...")
    return sorted(set(markers))


def verify_file_goal(goal: FileGoal, project_root: str) -> FileGoal:
    """Update and return a FileGoal based on filesystem evidence."""
    path = normalize_goal_path(goal.path)
    abs_path = Path(project_root) / path
    goal.path = path
    goal.failure_reason = ""
    if goal.must_exist and not abs_path.exists():
        goal.status = FileGoalStatus.FAILED
        goal.failure_reason = f"Missing required file: {path}"
        return goal
    if not abs_path.exists():
        return goal
    try:
        content = abs_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        goal.status = FileGoalStatus.FAILED
        goal.failure_reason = f"Cannot read {path}: {exc}"
        return goal
    goal.evidence.append(f"exists chars={len(content)}")
    if goal.must_be_non_empty and not content.strip() and not _empty_init_allowed(path, content):
        goal.status = FileGoalStatus.FAILED
        goal.failure_reason = f"Empty required file: {path}"
        return goal
    if goal.must_not_contain_placeholders:
        markers = detect_lazy_placeholders(content, path=path, purpose=goal.purpose)
        if markers:
            goal.status = FileGoalStatus.FAILED
            goal.failure_reason = f"Placeholder code detected in {path}: {', '.join(markers[:5])}"
            return goal
    if goal.must_compile_if_python and path.endswith(".py"):
        try:
            tree = ast.parse(content or "\n", filename=path)
        except SyntaxError as exc:
            goal.status = FileGoalStatus.FAILED
            goal.failure_reason = f"Python syntax error in {path}: line {exc.lineno}: {exc.msg}"
            return goal
        missing = _missing_expected_symbols(tree, goal.expected_symbols)
        if missing:
            goal.status = FileGoalStatus.FAILED
            goal.failure_reason = f"Missing expected symbols in {path}: {', '.join(missing)}"
            return goal
    goal.status = FileGoalStatus.DONE
    goal.evidence.append("verified")
    return goal


def _looks_like_noise_path(path: str) -> bool:
    lower = path.lower()
    if lower.startswith((".venv/", "venv/", "build/", "dist/", "__pycache__/")):
        return True
    return lower.endswith((".pyc", ".pyo", ".gguf"))


def _infer_purpose(path: str) -> str:
    name = os.path.basename(path).lower()
    if name == "readme.md":
        return "documentation"
    if name == "main.py":
        return "entrypoint"
    if "model" in name:
        return "business logic/model"
    if "controller" in name:
        return "controller/glue"
    if "view" in name:
        return "view/interface"
    if "test" in name:
        return "tests"
    return "project file"


def _infer_expected_symbols(path: str, text: str) -> list[str]:
    lower_path = path.lower()
    lower_text = text.lower()
    symbols: list[str] = []
    if lower_path.endswith("calculator.py") and "divide" in lower_text:
        symbols.append("divide")
    if lower_path.endswith("store.py"):
        for name in ("add", "list", "done", "clear"):
            if name in lower_text:
                symbols.append(name)
    return symbols


def _apply_simple_dependencies(goals: list[FileGoal]) -> None:
    by_path = {goal.path: f"filegoal-{index + 1}" for index, goal in enumerate(goals)}
    for index, goal in enumerate(goals):
        goal_id = by_path[goal.path]
        if not goal.dependency_ids:
            goal.dependency_ids = []
        if goal.path.endswith("app/controller.py") and "app/model.py" in by_path:
            goal.dependency_ids.append(by_path["app/model.py"])
        if goal.path.endswith("app/view.py") and "app/controller.py" in by_path:
            goal.dependency_ids.append(by_path["app/controller.py"])
        if goal.path.lower() == "readme.md":
            goal.dependency_ids.extend(
                dep_id for path, dep_id in by_path.items()
                if path != goal.path and path.endswith((".py", ".json"))
            )
        goal.evidence.append(f"id={goal_id}")


def _missing_expected_symbols(tree: ast.AST, expected: list[str]) -> list[str]:
    if not expected:
        return []
    found = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    return [name for name in expected if name not in found]


def _empty_init_allowed(path: str, content: str) -> bool:
    return os.path.basename(path).lower() == "__init__.py" and not content.strip()


def _abstract_context_allowed(content: str, purpose: str) -> bool:
    lower = (content + "\n" + purpose).lower()
    return any(token in lower for token in ("abstractmethod", "abc.", "protocol", "interface"))
