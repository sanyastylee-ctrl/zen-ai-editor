from __future__ import annotations

import json
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QCoreApplication

from ai.agent import AgentWorker
from core.model_manager import ModelManager
from core.profiles import AIProfile, ChatTemplate, ProfileKind


MODEL_FILE = "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
PLACEHOLDER_MARKERS = [
    "...",
    "TODO implement",
    "TODO: implement",
    "ostальной код",
    "остальной код",
    "здесь будет",
    "NotImplementedError",
]


def make_profile(max_tokens: int = 2048, n_ctx: int = 10240) -> AIProfile:
    return AIProfile(
        id="real-coder-smoke",
        name="Coder real smoke",
        kind=ProfileKind.CODER,
        model_file=MODEL_FILE,
        chat_template=ChatTemplate.CHATML,
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.1,
        max_tokens=max_tokens,
        agent_mode=True,
        auto_continue_enabled=True,
        max_auto_continues_per_task=5,
        max_total_task_minutes=30,
        max_no_progress_retries=2,
    )


def run_cmd(cwd: Path, command: str, timeout: int = 30) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration": round(time.time() - started, 3),
    }


class SmokeRun:
    def __init__(
        self,
        name: str,
        prompt: str,
        project_setup: Callable[[Path], None] | None = None,
        max_agent_steps: int = 10,
        max_tool_calls: int = 120,
        max_generation_seconds: int = 420,
    ) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.name = name
        self.project = Path(tempfile.gettempdir()) / f"ZenAI_real_model_{name}_{stamp}"
        if self.project.exists():
            shutil.rmtree(self.project)
        self.project.mkdir(parents=True)
        if project_setup:
            project_setup(self.project)
        self.prompt = prompt
        self.max_agent_steps = max_agent_steps
        self.max_tool_calls = max_tool_calls
        self.max_generation_seconds = max_generation_seconds
        self.events: list[dict[str, Any]] = []
        self.chunks: list[str] = []
        self.states: list[dict[str, Any]] = []
        self.finished: list[dict[str, Any]] = []
        self.blocked: list[dict[str, Any]] = []
        self.auto_continues = 0
        self.last_worker: AgentWorker | None = None

    def run(self) -> None:
        continuation: dict[str, Any] | None = None
        user_message = self.prompt
        for idx in range(6):
            worker = AgentWorker(
                profile=make_profile(),
                user_message=user_message,
                history=[],
                project_root=str(self.project),
                confirmation_policy="auto_confirm",
                max_agent_steps=self.max_agent_steps,
                max_tool_calls=self.max_tool_calls,
                max_generation_seconds=self.max_generation_seconds,
                continuation_state=continuation,
            )
            self.last_worker = worker
            worker.chunk_received.connect(lambda text: self.chunks.append(str(text)))
            worker.tool_started.connect(lambda ev: self.events.append({"event": "tool_started", **dict(ev)}))
            worker.tool_finished.connect(lambda ev: self.events.append({"event": "tool_finished", **dict(ev)}))
            worker.agent_state_updated.connect(lambda ev: self.states.append(dict(ev)))
            worker.agent_finished.connect(lambda ev: self.finished.append(dict(ev)))
            worker.agent_blocked.connect(lambda ev: self.blocked.append(dict(ev)))
            worker.agent_auto_continue.connect(lambda ev: self.events.append({"event": "auto_continue", **dict(ev)}))
            worker.model_loading.connect(lambda path: self.events.append({"event": "model_loading", "path": path}))
            worker.model_loaded.connect(
                lambda path, ok, err: self.events.append(
                    {"event": "model_loaded", "path": path, "ok": bool(ok), "error": str(err)}
                )
            )
            worker.run()
            if worker.auto_continue_requested and worker.continuation_state:
                self.auto_continues += 1
                continuation = worker.continuation_state
                user_message = "продолжай"
                continue
            break

    def snapshot(self) -> dict[str, Any]:
        worker = self.last_worker
        file_goals = []
        command_goals_done = []
        final_allowed = None
        blocked_reason = ""
        if worker is not None:
            file_goals = [
                {
                    "path": goal.path,
                    "status": getattr(goal.status, "value", str(goal.status)),
                    "failure_reason": goal.failure_reason,
                }
                for goal in getattr(worker, "_file_goals", [])
            ]
            command_goals_done = sorted(getattr(worker, "_command_goals_done", set()))
            try:
                final_allowed = worker._final_evaluation().allowed
            except Exception as exc:
                final_allowed = f"error: {exc}"
            blocked_reason = getattr(worker, "blocked_reason", "") or getattr(worker, "stop_reason", "")
        return {
            "name": self.name,
            "project": str(self.project),
            "auto_continues": self.auto_continues,
            "events": self.events,
            "states_tail": self.states[-5:],
            "finished_tail": self.finished[-3:],
            "blocked_tail": self.blocked[-3:],
            "chunks_tail": "".join(self.chunks)[-4000:],
            "file_goals": file_goals,
            "command_goals_done": command_goals_done,
            "final_allowed": final_allowed,
            "blocked_reason": blocked_reason,
            "files": sorted(
                str(path.relative_to(self.project)).replace("\\", "/")
                for path in self.project.rglob("*")
                if path.is_file()
            ),
        }

    def cleanup_if_passed(self, passed: bool) -> None:
        if passed and self.project.exists():
            shutil.rmtree(self.project)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def setup_existing_project(root: Path) -> None:
    write_text(root / "app" / "__init__.py", "")
    write_text(
        root / "app" / "calculator.py",
        "def add(a, b):\n"
        "    return a + b\n\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n\n\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
    )
    write_text(
        root / "app" / "cli.py",
        "import argparse\n\n"
        "from app.calculator import add, subtract, multiply\n\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description=\"Small calculator CLI\")\n"
        "    parser.add_argument(\"operation\", choices=[\"add\", \"subtract\", \"multiply\"])\n"
        "    parser.add_argument(\"a\", type=float)\n"
        "    parser.add_argument(\"b\", type=float)\n"
        "    args = parser.parse_args()\n\n"
        "    if args.operation == \"add\":\n"
        "        result = add(args.a, args.b)\n"
        "    elif args.operation == \"subtract\":\n"
        "        result = subtract(args.a, args.b)\n"
        "    else:\n"
        "        result = multiply(args.a, args.b)\n\n"
        "    print(result)\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    main()\n",
    )
    write_text(
        root / "tests" / "test_calculator.py",
        "from app.calculator import add, subtract, multiply\n\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n\n"
        "def test_subtract():\n"
        "    assert subtract(5, 2) == 3\n\n\n"
        "def test_multiply():\n"
        "    assert multiply(4, 3) == 12\n",
    )
    write_text(
        root / "README.md",
        "# Calculator CLI\n\n"
        "Commands:\n\n"
        "```bash\n"
        "python -m app.cli add 2 3\n"
        "python -m app.cli subtract 5 2\n"
        "python -m app.cli multiply 4 3\n"
        "```\n",
    )


def setup_traceback_project(root: Path) -> None:
    write_text(root / "main.py", "from calculator import divide\n\nprint(divide(10, 0))\n")
    write_text(root / "calculator.py", "def divide(a, b):\n    return a / b\n")


def has_placeholders(root: Path) -> list[str]:
    hits = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in PLACEHOLDER_MARKERS:
            if marker in text:
                hits.append(f"{path.relative_to(root)}:{marker}")
    return hits


def validate_cli_todo(run: SmokeRun) -> tuple[bool, dict[str, Any]]:
    results = [
        run_cmd(run.project, 'python main.py add "first task"'),
        run_cmd(run.project, "python main.py list"),
        run_cmd(run.project, "python main.py done 1"),
        run_cmd(run.project, "python main.py clear"),
    ]
    notes = run.project / "notes.json"
    detail = {
        "external_commands": results,
        "notes_exists": notes.exists(),
        "notes_content": notes.read_text(encoding="utf-8", errors="ignore") if notes.exists() else "",
        "command_goals_done": run.snapshot()["command_goals_done"],
    }
    passed = (
        (run.project / "main.py").exists()
        and all(item["exit_code"] == 0 for item in results)
        and "first task" in results[1]["stdout"]
        and notes.exists()
        and {"python main.py add \"first task\"", "python main.py list", "python main.py done 1", "python main.py clear"}.issubset(
            set(detail["command_goals_done"])
        )
    )
    return passed, detail


def validate_multifile(run: SmokeRun) -> tuple[bool, dict[str, Any]]:
    expected = ["main.py", "app/model.py", "app/controller.py", "app/view.py", "README.md"]
    compiles = [run_cmd(run.project, f"python -m py_compile {path}") for path in expected if path.endswith(".py")]
    launch = run_cmd(run.project, "python main.py", timeout=15)
    placeholders = has_placeholders(run.project)
    detail = {"expected": expected, "compiles": compiles, "launch": launch, "placeholders": placeholders}
    passed = (
        all((run.project / path).exists() and (run.project / path).stat().st_size > 0 for path in expected)
        and all(item["exit_code"] == 0 for item in compiles)
        and launch["exit_code"] == 0
        and not placeholders
    )
    return passed, detail


def validate_existing_patch(run: SmokeRun) -> tuple[bool, dict[str, Any]]:
    test_command = (
        "python -m pytest"
        if importlib.util.find_spec("pytest") is not None
        else (
            "python -c \"exec('from tests import test_calculator as t\\n"
            "for name in (\\'test_add\\', \\'test_subtract\\', \\'test_multiply\\', \\'test_divide\\', \\'test_divide_by_zero\\'):\\n"
            "    getattr(t, name)()\\n"
            "print(\\'calculator tests passed\\')')\""
        )
    )
    commands = [
        test_command,
        "python -m app.cli add 2 3",
        "python -m app.cli subtract 5 2",
        "python -m app.cli multiply 4 3",
        "python -m app.cli divide 10 2",
        "python -m app.cli divide 10 0",
    ]
    results = [run_cmd(run.project, command) for command in commands]
    calc = (run.project / "app" / "calculator.py").read_text(encoding="utf-8", errors="ignore")
    cli = (run.project / "app" / "cli.py").read_text(encoding="utf-8", errors="ignore")
    detail = {
        "external_commands": results,
        "calculator_contains_divide": "def divide" in calc,
        "cli_contains_divide": "divide" in cli,
        "agent_ran_tests": any(
            ev.get("event") == "tool_finished"
            and ev.get("name") == "run_terminal"
            and "test_calculator" in json.dumps(ev.get("args", {}), ensure_ascii=False)
            and ev.get("ok") is True
            for ev in run.events
        ),
        "tool_events": [
            {k: ev.get(k) for k in ("event", "name", "path", "command", "ok")}
            for ev in run.events
            if ev.get("event") in {"tool_started", "tool_finished"}
        ],
    }
    passed = (
        all(item["exit_code"] == 0 for item in results)
        and ("5" in results[4]["stdout"] or "5.0" in results[4]["stdout"])
        and "Traceback" not in results[5]["stderr"]
        and detail["calculator_contains_divide"]
        and detail["cli_contains_divide"]
        and detail["agent_ran_tests"]
    )
    return passed, detail


def validate_traceback(run: SmokeRun) -> tuple[bool, dict[str, Any]]:
    rerun = run_cmd(run.project, "python main.py")
    calc = (run.project / "calculator.py").read_text(encoding="utf-8", errors="ignore")
    main = (run.project / "main.py").read_text(encoding="utf-8", errors="ignore")
    detail = {"rerun": rerun, "calculator": calc, "main": main}
    passed = rerun["exit_code"] == 0 and "ZeroDivisionError" not in (rerun["stdout"] + rerun["stderr"])
    return passed, detail


def run_one(name: str, prompt: str, setup: Callable[[Path], None] | None, validator, **kwargs) -> dict[str, Any]:
    smoke = SmokeRun(name, prompt, setup, **kwargs)
    smoke.run()
    passed, detail = validator(smoke)
    snapshot = smoke.snapshot()
    snapshot["validation"] = detail
    snapshot["passed"] = bool(passed)
    smoke.cleanup_if_passed(bool(passed))
    snapshot["project_deleted"] = bool(passed and not Path(snapshot["project"]).exists())
    return snapshot


def main() -> int:
    QCoreApplication.instance() or QCoreApplication([])
    ModelManager.instance().set_max_loaded(1)
    scenarios = [
        (
            "cli_todo",
            'Создай CLI todo app. Команды должны работать:\n'
            'python main.py add "first task"\n'
            "python main.py list\n"
            "python main.py done 1\n"
            "python main.py clear\n\n"
            "Храни задачи в notes.json. После создания сам проверь все команды через терминал.",
            None,
            validate_cli_todo,
            {},
        ),
        (
            "multifile_filegoals",
            "Создай небольшой multi-file Python проект:\n\n"
            "* main.py\n* app/model.py\n* app/controller.py\n* app/view.py\n* README.md\n\n"
            "Требования:\n"
            "* main.py запускает приложение;\n"
            "* model.py содержит бизнес-логику;\n"
            "* controller.py связывает model/view;\n"
            "* view.py содержит простой CLI или tkinter interface;\n"
            "* README.md объясняет запуск;\n"
            "* не используй заглушки;\n"
            "* после создания проверь запуск через терминал.",
            None,
            validate_multifile,
            {},
        ),
        (
            "existing_project_patch",
            "В существующий calculator CLI добавь операцию divide. Нужно:\n"
            "* добавить функцию divide(a, b);\n"
            "* обработать деление на ноль понятной ошибкой;\n"
            "* добавить CLI operation divide;\n"
            "* добавить тесты;\n"
            "* обновить README;\n"
            "* запустить проверки.\n"
            "Не переписывай существующие файлы целиком, внеси точечные изменения.",
            setup_existing_project,
            validate_existing_patch,
            {},
        ),
        (
            "traceback_repair",
            "Запусти проект, найди ошибку по traceback и исправь так, чтобы программа больше не падала. "
            "После исправления снова запусти команду.",
            setup_traceback_project,
            validate_traceback,
            {},
        ),
        (
            "auto_continue",
            "Создай небольшой multi-file Python проект:\n\n"
            "* main.py\n* app/model.py\n* app/controller.py\n* app/view.py\n* README.md\n\n"
            "Требования:\n"
            "* main.py запускает приложение;\n"
            "* model.py содержит бизнес-логику;\n"
            "* controller.py связывает model/view;\n"
            "* view.py содержит простой CLI или tkinter interface;\n"
            "* README.md объясняет запуск;\n"
            "* не используй заглушки;\n"
            "* после создания проверь запуск через терминал.",
            None,
            validate_multifile,
            {"max_agent_steps": 2, "max_tool_calls": 12, "max_generation_seconds": 360},
        ),
    ]
    report: list[dict[str, Any]] = []
    for name, prompt, setup, validator, kwargs in scenarios:
        result = run_one(name, prompt, setup, validator, **kwargs)
        report.append(result)
        out_path = ROOT / ".agent_smoke" / f"{name}_result.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"name": name, "passed": result["passed"], "project": result["project"]}, ensure_ascii=False), flush=True)
        if not result["passed"]:
            break
    summary = {"results": report, "all_passed": all(item["passed"] for item in report)}
    (ROOT / ".agent_smoke" / "real_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
