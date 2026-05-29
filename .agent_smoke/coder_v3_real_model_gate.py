from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ai.agent import AgentWorker
from core.profiles import AIProfile, ChatTemplate, ProfileKind


MODEL_FILE = "qwen2.5-coder-14b-instruct-q4_k_m.gguf"


def profile() -> AIProfile:
    return AIProfile(
        id="coder-real-smoke",
        name="Coder real smoke",
        kind=ProfileKind.CODER,
        model_file=MODEL_FILE,
        chat_template=ChatTemplate.CHATML,
        n_ctx=10240,
        max_tokens=2048,
        temperature=0.18,
        top_p=0.9,
        repeat_penalty=1.1,
        n_gpu_layers=-1,
        agent_mode=True,
    )


def run_worker(task: str, root: Path, *, continuation_state=None, max_agent_steps=18, max_context_chars=240_000):
    worker = AgentWorker(
        profile(),
        task,
        project_root=str(root),
        confirmation_policy="auto_confirm",
        max_agent_steps=max_agent_steps,
        max_tool_calls=120,
        max_generation_seconds=300,
        max_context_chars=max_context_chars,
        continuation_state=continuation_state,
    )
    chunks: list[str] = []
    tools: list[dict] = []
    progress: list[dict] = []
    worker.chunk_received.connect(chunks.append)
    worker.tool_finished.connect(tools.append)
    worker.agent_state_updated.connect(progress.append)
    started = time.time()
    worker.run()
    return {
        "chunks": chunks,
        "tools": tools,
        "progress": progress,
        "worker": worker,
        "seconds": round(time.time() - started, 1),
    }


def command_events(result: dict) -> list[dict]:
    events = []
    for item in result["tools"]:
        if item.get("name") in {"run_terminal", "run_command"}:
            meta = item.get("meta") or {}
            events.append(
                {
                    "command": meta.get("command") or "",
                    "ok": bool(item.get("ok")),
                    "exit_code": meta.get("exit_code"),
                    "output_preview": str(item.get("output") or "")[:1000],
                }
            )
    return events


def file_text(root: Path, rel: str) -> str:
    path = root / rel
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def smoke_cli_todo() -> dict:
    root = Path(tempfile.mkdtemp(prefix="ZenAI_real_cli_todo_"))
    task = (
        "Создай CLI todo app. Команды должны работать:\n"
        "python main.py add \"first task\"\n"
        "python main.py list\n"
        "python main.py done 1\n"
        "python main.py clear\n\n"
        "Храни задачи в notes.json. После создания сам проверь все команды через терминал."
    )
    result = run_worker(task, root, max_agent_steps=24)
    commands = command_events(result)
    expected = [
        'python main.py add "first task"',
        "python main.py list",
        "python main.py done 1",
        "python main.py clear",
    ]
    outputs = "\n".join(event["output_preview"] for event in commands)
    notes = file_text(root, "notes.json")
    passed = (
        (root / "main.py").exists()
        and (root / "notes.json").exists()
        and all(any(event["command"] == cmd and event["ok"] for event in commands) for cmd in expected)
        and "first task" in outputs
        and result["worker"].run_state_v3.final_allowed
        and result["worker"].continuation_state is None
    )
    return finish_result(
        "cli_todo",
        root,
        result,
        passed,
        extra={"commands": commands, "notes_json": notes[:1000], "expected_commands": expected},
    )


def smoke_multifile() -> dict:
    root = Path(tempfile.mkdtemp(prefix="ZenAI_real_filegoals_"))
    task = (
        "Создай небольшой multi-file Python проект:\n"
        "* main.py\n* app/model.py\n* app/controller.py\n* app/view.py\n* README.md\n\n"
        "Требования:\n"
        "* main.py запускает приложение;\n"
        "* model.py содержит бизнес-логику;\n"
        "* controller.py связывает model/view;\n"
        "* view.py содержит простой CLI или tkinter interface;\n"
        "* README.md объясняет запуск;\n"
        "* не используй заглушки;\n"
        "* после создания проверь запуск через терминал."
    )
    result = run_worker(task, root, max_agent_steps=26)
    required = ["main.py", "app/model.py", "app/controller.py", "app/view.py", "README.md"]
    placeholders = ["...", "TODO implement", "остальной код", "NotImplementedError", "здесь будет"]
    files = {rel: file_text(root, rel) for rel in required}
    commands = command_events(result)
    passed = (
        all(files[rel].strip() for rel in required)
        and not any(marker in text for text in files.values() for marker in placeholders)
        and any(event["command"].startswith("python main.py") and event["ok"] for event in commands)
        and result["worker"].run_state_v3.final_allowed
        and all(getattr(goal.status, "value", goal.status) == "done" for goal in result["worker"]._file_goals)
    )
    return finish_result(
        "multifile_filegoals",
        root,
        result,
        passed,
        extra={"files": {k: len(v) for k, v in files.items()}, "commands": commands, "file_goals": file_goals(result)},
    )


def prepare_existing_project(root: Path) -> None:
    (root / "app").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def subtract(a, b):\n    return a - b\n\n"
        "def multiply(a, b):\n    return a * b\n",
        encoding="utf-8",
    )
    (root / "app" / "cli.py").write_text(
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
        encoding="utf-8",
    )
    (root / "tests" / "test_calculator.py").write_text(
        "from app.calculator import add, subtract, multiply\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_subtract():\n    assert subtract(5, 2) == 3\n\n\n"
        "def test_multiply():\n    assert multiply(4, 3) == 12\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Calculator CLI\n\nCommands:\n\n"
        "```bash\npython -m app.cli add 2 3\npython -m app.cli subtract 5 2\npython -m app.cli multiply 4 3\n```\n",
        encoding="utf-8",
    )


def smoke_existing_patch() -> dict:
    root = Path(tempfile.mkdtemp(prefix="ZenAI_real_existing_patch_"))
    prepare_existing_project(root)
    task = (
        "В существующий calculator CLI добавь операцию divide. Нужно:\n"
        "* добавить функцию divide(a, b);\n"
        "* обработать деление на ноль понятной ошибкой;\n"
        "* добавить CLI operation divide;\n"
        "* добавить тесты;\n"
        "* обновить README;\n"
        "* запустить проверки.\n"
        "Не переписывай существующие файлы целиком, внеси точечные изменения."
    )
    result = run_worker(task, root, max_agent_steps=28)
    commands = command_events(result)
    calc = file_text(root, "app/calculator.py")
    cli = file_text(root, "app/cli.py")
    tests = file_text(root, "tests/test_calculator.py")
    readme = file_text(root, "README.md")
    passed = (
        "divide" in calc
        and "divide" in cli
        and "divide" in tests
        and "divide" in readme.lower()
        and any("python -m app.cli divide 10 2" in event["command"] and event["ok"] for event in commands)
        and any("python -m app.cli divide 10 0" in event["command"] for event in commands)
        and result["worker"].run_state_v3.final_allowed
    )
    return finish_result("existing_project_patch", root, result, passed, extra={"commands": commands})


def smoke_traceback_repair() -> dict:
    root = Path(tempfile.mkdtemp(prefix="ZenAI_real_traceback_"))
    (root / "main.py").write_text("from calculator import divide\n\nprint(divide(10, 0))\n", encoding="utf-8")
    (root / "calculator.py").write_text("def divide(a, b):\n    return a / b\n", encoding="utf-8")
    task = "Запусти проект, найди ошибку по traceback и исправь так, чтобы программа больше не падала. После исправления снова запусти команду."
    result = run_worker(task, root, max_agent_steps=22)
    commands = command_events(result)
    calc = file_text(root, "calculator.py")
    passed = (
        any("ZeroDivisionError" in event["output_preview"] for event in commands)
        and any(event["command"] == "python main.py" and event["ok"] for event in commands)
        and "ZeroDivisionError" not in commands[-1]["output_preview"] if commands else False
    )
    passed = passed and result["worker"].run_state_v3.final_allowed and ("b == 0" in calc or "ZeroDivisionError" in calc or "zero" in calc.lower())
    return finish_result("traceback_repair", root, result, passed, extra={"commands": commands})


def smoke_auto_continue() -> dict:
    root = Path(tempfile.mkdtemp(prefix="ZenAI_real_autocontinue_"))
    task = "Создай main.py и app/model.py. main.py должен запускаться. После создания проверь python main.py."
    continuation = None
    runs = []
    for index in range(1, 5):
        result = run_worker(
            task if index == 1 else "продолжай",
            root,
            continuation_state=continuation,
            max_agent_steps=1,
            max_context_chars=240_000,
        )
        runs.append(result)
        continuation = result["worker"].continuation_state
        if continuation is None:
            break
    commands = [event for result in runs for event in command_events(result)]
    passed = (
        (root / "main.py").exists()
        and (root / "app" / "model.py").exists()
        and any(event["command"].startswith("python main.py") and event["ok"] for event in commands)
        and continuation is None
        and len(runs) > 1
    )
    merged = runs[-1]
    merged["tools"] = [item for result in runs for item in result["tools"]]
    merged["chunks"] = [item for result in runs for item in result["chunks"]]
    return finish_result("auto_continue", root, merged, passed, extra={"runs": len(runs), "commands": commands})


def file_goals(result: dict) -> list[dict]:
    return [
        {"path": goal.path, "status": getattr(goal.status, "value", str(goal.status)), "reason": goal.failure_reason}
        for goal in result["worker"]._file_goals
    ]


def finish_result(name: str, root: Path, result: dict, passed: bool, *, extra: dict) -> dict:
    files = sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file())
    data = {
        "name": name,
        "passed": bool(passed),
        "temp_project": str(root),
        "seconds": result["seconds"],
        "files": files,
        "tool_names": [item.get("name") for item in result["tools"]],
        "tool_count": len(result["tools"]),
        "final_allowed": bool(result["worker"].run_state_v3.final_allowed),
        "blocked_reason": result["worker"].run_state_v3.blocked_reason,
        "continuation_state": result["worker"].continuation_state is not None,
        "chunks_tail": "".join(result["chunks"])[-2000:],
        **extra,
    }
    if passed:
        shutil.rmtree(root, ignore_errors=True)
        data["temp_project_deleted"] = not root.exists()
    else:
        data["temp_project_deleted"] = False
    return data


def main() -> int:
    QApplication.instance() or QApplication([])
    selected = sys.argv[1:] or ["cli_todo", "multifile", "existing", "traceback", "auto_continue"]
    funcs = {
        "cli_todo": smoke_cli_todo,
        "multifile": smoke_multifile,
        "existing": smoke_existing_patch,
        "traceback": smoke_traceback_repair,
        "auto_continue": smoke_auto_continue,
    }
    results = []
    for name in selected:
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        print(json.dumps({"event": "smoke_start", "name": name, "started": started}, ensure_ascii=False), flush=True)
        try:
            results.append(funcs[name]())
        except Exception as exc:
            results.append({"name": name, "passed": False, "error": repr(exc)})
        print(json.dumps(results[-1], ensure_ascii=False, indent=2), flush=True)
        if not results[-1].get("passed"):
            break
    print(json.dumps({"summary": results}, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(item.get("passed") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
