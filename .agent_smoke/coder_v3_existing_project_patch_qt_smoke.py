from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QStyleFactory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SMOKE_APPDATA_ROOT = Path(tempfile.gettempdir()) / "ZenAI_existing_patch_gate_appdata"
os.environ["APPDATA"] = str(SMOKE_APPDATA_ROOT)

from ui.main_window import ZenEditor  # noqa: E402


APPDATA = Path(os.environ["APPDATA"]) / "ZenAI"
SETTINGS = APPDATA / "settings"
SESSIONS = APPDATA / "sessions"
LOG = APPDATA / "logs" / "zenai.log"

TASK = (
    "В существующий calculator CLI добавь операцию divide. Нужно:\n"
    "- добавить функцию divide(a, b);\n"
    "- обработать деление на ноль понятной ошибкой;\n"
    "- добавить CLI operation divide;\n"
    "- добавить тесты;\n"
    "- обновить README;\n"
    "- запустить проверки.\n"
    "Не переписывай существующие файлы целиком, внеси точечные изменения."
)


class ScriptedModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def __call__(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        text = self.outputs.pop(0) if self.outputs else "Готово: изменения внесены и проверены."
        return iter([{"choices": [{"text": text}]}])


class FakeManager:
    def __init__(self, model: ScriptedModel) -> None:
        self.model = model

    def on_load_start(self, cb): pass
    def on_load_finish(self, cb): pass
    def off_load_start(self, cb): pass
    def off_load_finish(self, cb): pass
    def get_model(self, **kwargs): return self.model


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_project(project: Path) -> None:
    (project / "app").mkdir(parents=True)
    (project / "app" / "__init__.py").write_text("", encoding="utf-8")
    (project / "app" / "calculator.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n\n\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
        encoding="utf-8",
    )
    (project / "app" / "cli.py").write_text(
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
    (project / "tests").mkdir()
    (project / "tests" / "test_calculator.py").write_text(
        "from app.calculator import add, subtract, multiply\n\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n\n"
        "def test_subtract():\n"
        "    assert subtract(5, 2) == 3\n\n\n"
        "def test_multiply():\n"
        "    assert multiply(4, 3) == 12\n",
        encoding="utf-8",
    )
    (project / "README.md").write_text(
        "# Calculator CLI\n\n"
        "Commands:\n\n"
        "```bash\n"
        "python -m app.cli add 2 3\n"
        "python -m app.cli subtract 5 2\n"
        "python -m app.cli multiply 4 3\n"
        "```\n",
        encoding="utf-8",
    )


def prepare_state(project: Path) -> dict[Path, str | None]:
    backups: dict[Path, str | None] = {}
    files = [
        SETTINGS / "profiles.json",
        SESSIONS / "chat_state.json",
        SESSIONS / "recent_projects.json",
    ]
    for path in files:
        backups[path] = path.read_text(encoding="utf-8") if path.exists() else None

    profiles_path = SETTINGS / "profiles.json"
    profiles = read_json(profiles_path, {})
    coder_id = ""
    for profile in profiles.get("profiles", []):
        if profile.get("kind") == "coder":
            profile["agent_mode"] = True
            profile["n_gpu_layers"] = -1
            coder_id = profile.get("id", "")
            break
    if not coder_id:
        coder_id = "smoke-coder"
        profiles.setdefault("profiles", []).append(
            {
                "id": coder_id,
                "name": "Кодер",
                "kind": "coder",
                "icon": "ti-code",
                "model_file": "fake.gguf",
                "chat_template": "chatml",
                "n_ctx": 8192,
                "n_gpu_layers": -1,
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 20,
                "repeat_penalty": 1.1,
                "max_tokens": 1024,
                "stop_sequences": [],
                "system_prompt": "",
                "agent_mode": True,
                "persona": {},
                "enable_vision_assist": False,
                "vision_model_file": "",
                "mmproj_file": "",
                "vision_handler": "",
                "max_visual_context_chars": 4000,
                "vision_first_policy": "auto",
                "search_enabled": True,
                "max_search_results": 5,
                "max_pages_to_read": 3,
                "require_sources_for_fresh_info": True,
                "answer_style": "detailed",
            }
        )
    profiles.setdefault("active", {})["coder"] = coder_id
    write_json(profiles_path, profiles)

    chat_state = read_json(SESSIONS / "chat_state.json", {})
    chat_state["last_profile_id"] = coder_id
    chat_state.setdefault("sessions_by_profile", {})
    write_json(SESSIONS / "chat_state.json", chat_state)
    write_json(
        SESSIONS / "recent_projects.json",
        {"version": 1, "last_opened": str(project), "recent": [str(project)]},
    )
    return backups


def restore_state(backups: dict[Path, str | None]) -> None:
    for path, text in backups.items():
        if text is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def tail_log(start_size: int) -> str:
    if not LOG.exists():
        return ""
    with LOG.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(min(start_size, LOG.stat().st_size))
        return f.read()


def scripted_outputs() -> list[str]:
    return [
        """
План:
1. Осмотрю существующие calculator CLI файлы.
2. Точечно добавлю divide в код, CLI, тесты и README.
3. Запущу регрессионные CLI проверки.
<tool name="list_files"><path>.</path><max_depth>3</max_depth></tool>
<tool name="read_file"><path>app/calculator.py</path></tool>
<tool name="read_file"><path>app/cli.py</path></tool>
<tool name="read_file"><path>tests/test_calculator.py</path></tool>
<tool name="read_file"><path>README.md</path></tool>
""",
        """
<tool name="edit_file">
<path>app/calculator.py</path>
<old_str>def multiply(a, b):
    return a * b</old_str>
<new_str>def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b</new_str>
</tool>
<tool name="edit_file">
<path>app/cli.py</path>
<old_str>from app.calculator import add, subtract, multiply</old_str>
<new_str>from app.calculator import add, subtract, multiply, divide</new_str>
</tool>
<tool name="edit_file">
<path>app/cli.py</path>
<old_str>parser.add_argument("operation", choices=["add", "subtract", "multiply"])</old_str>
<new_str>parser.add_argument("operation", choices=["add", "subtract", "multiply", "divide"])</new_str>
</tool>
<tool name="edit_file">
<path>app/cli.py</path>
<old_str>    elif args.operation == "subtract":
        result = subtract(args.a, args.b)
    else:
        result = multiply(args.a, args.b)</old_str>
<new_str>    elif args.operation == "subtract":
        result = subtract(args.a, args.b)
    elif args.operation == "multiply":
        result = multiply(args.a, args.b)
    else:
        result = divide(args.a, args.b)</new_str>
</tool>
<tool name="edit_file">
<path>tests/test_calculator.py</path>
<old_str>from app.calculator import add, subtract, multiply</old_str>
<new_str>from app.calculator import add, subtract, multiply, divide</new_str>
</tool>
<tool name="edit_file">
<path>tests/test_calculator.py</path>
<old_str>def test_multiply():
    assert multiply(4, 3) == 12</old_str>
<new_str>def test_multiply():
    assert multiply(4, 3) == 12


import unittest


class TestDivide(unittest.TestCase):
    def test_divide(self):
        self.assertEqual(divide(10, 2), 5)

    def test_divide_by_zero(self):
        self.assertIn("zero", divide(10, 0).lower())


def test_divide():
    assert divide(10, 2) == 5


def test_divide_by_zero():
    assert "zero" in divide(10, 0).lower()</new_str>
</tool>
<tool name="edit_file">
<path>README.md</path>
<old_str>python -m app.cli multiply 4 3</old_str>
<new_str>python -m app.cli multiply 4 3
python -m app.cli divide 10 2
python -m app.cli divide 10 0</new_str>
</tool>
""",
        """
<tool name="run_terminal"><command>python -m unittest discover -s tests</command></tool>
<tool name="run_terminal"><command>python -m app.cli add 2 3</command></tool>
<tool name="run_terminal"><command>python -m app.cli subtract 5 2</command></tool>
<tool name="run_terminal"><command>python -m app.cli multiply 4 3</command></tool>
<tool name="run_terminal"><command>python -m app.cli divide 10 2</command></tool>
<tool name="run_terminal"><command>python -m app.cli divide 10 0</command></tool>
""",
        "Готово: добавил divide, обновил CLI, тесты и README, проверки прошли.",
    ]


def main() -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    project = Path(tempfile.gettempdir()) / f"ZenAI_existing_patch_gate_{stamp}"
    prepare_project(project)
    start_size = LOG.stat().st_size if LOG.exists() else 0
    backups = prepare_state(project)
    result: dict = {"tool_events": []}
    connected_workers: set[int] = set()
    model = ScriptedModel(scripted_outputs())

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = ZenEditor()
    window.show()

    def send() -> None:
        window.chat_input.setText(TASK)
        window.send_message()

    QTimer.singleShot(1000, send)
    deadline = time.time() + 180

    def maybe_connect_worker() -> None:
        worker = getattr(window, "worker", None)
        if worker is None or id(worker) in connected_workers:
            return
        connected_workers.add(id(worker))

        def on_tool(payload: dict) -> None:
            result["tool_events"].append(payload)

        worker.tool_finished.connect(on_tool)

    def poll() -> None:
        maybe_connect_worker()
        text = tail_log(start_size)
        tool_events = result.get("tool_events", [])
        names = [str(item.get("name", "")) for item in tool_events]
        terminal_outputs = [
            str(item.get("output", ""))
            for item in tool_events
            if item.get("name") == "run_terminal"
        ]
        calc = (project / "app" / "calculator.py").read_text(encoding="utf-8", errors="ignore")
        cli = (project / "app" / "cli.py").read_text(encoding="utf-8", errors="ignore")
        tests = (project / "tests" / "test_calculator.py").read_text(encoding="utf-8", errors="ignore")
        readme = (project / "README.md").read_text(encoding="utf-8", errors="ignore")
        divide_added = all(
            marker in text_value
            for marker, text_value in [
                ("def divide", calc),
                ('"divide"', cli),
                ("test_divide", tests),
                ("divide 10 2", readme),
            ]
        )
        command_checks = {
            "tests": any("python -m unittest discover -s tests" in output and "[exit 0]" in output for output in terminal_outputs),
            "add": any("python -m app.cli add 2 3" in output and "[exit 0]" in output for output in terminal_outputs),
            "subtract": any("python -m app.cli subtract 5 2" in output and "[exit 0]" in output for output in terminal_outputs),
            "multiply": any("python -m app.cli multiply 4 3" in output and "[exit 0]" in output for output in terminal_outputs),
            "divide_ok": any("python -m app.cli divide 10 2" in output and "5.0" in output and "[exit 0]" in output for output in terminal_outputs),
            "divide_zero": any("python -m app.cli divide 10 0" in output and "zero" in output.lower() and "[exit 0]" in output for output in terminal_outputs),
        }
        no_full_rewrite = "write_file" not in names
        read_relevant = {"app/calculator.py", "app/cli.py", "tests/test_calculator.py", "README.md"}.issubset(
            {
                str((item.get("args") or {}).get("path", ""))
                for item in tool_events
                if item.get("name") == "read_file"
            }
        )
        final_allowed = "[coder_evaluator]" in text and "allowed=true" in text.lower()
        finished = "[agent_finished]" in text and "completed" in text
        done = (
            divide_added
            and all(command_checks.values())
            and no_full_rewrite
            and read_relevant
            and final_allowed
            and finished
        )
        blocked = "[agent_blocked]" in text
        if done or blocked or time.time() > deadline:
            result.update({
                "done": done,
                "blocked": blocked,
                "timeout": time.time() > deadline,
                "command_checks": command_checks,
                "no_full_rewrite": no_full_rewrite,
                "read_relevant": read_relevant,
                "divide_added": divide_added,
                "final_allowed": final_allowed,
                "finished": finished,
                "calculator_py": calc,
                "cli_py": cli,
                "tests_py": tests,
                "readme": readme,
                "prompts": len(model.prompts),
                "relevant_logs": [
                    line for line in text.splitlines()
                    if "[ui_send_" in line
                    or "[agent_tool_" in line
                    or "[agent_command_goal" in line
                    or "[coder_verification]" in line
                    or "[coder_evaluator]" in line
                    or "[agent_finished]" in line
                    or "[coder_guard_triggered]" in line
                ][-120:],
            })
            window.close()
            app.quit()
            return
        QTimer.singleShot(1000, poll)

    patches = [
        mock.patch("ai.agent.LLAMA_AVAILABLE", True),
        mock.patch("ai.agent.resolve_model_path", return_value=str(project / "fake.gguf")),
        mock.patch("ai.agent.ModelManager.instance", return_value=FakeManager(model)),
        mock.patch("ai.agent.acceleration_warning", return_value=""),
    ]
    for patcher in patches:
        patcher.start()
    try:
        QTimer.singleShot(1000, poll)
        exit_code = app.exec()
    finally:
        for patcher in reversed(patches):
            patcher.stop()

    os.chdir(str(ROOT))
    restore_state(backups)
    deleted = False
    try:
        if project.exists() and "ZenAI_existing_patch_gate_" in str(project):
            shutil.rmtree(project)
            deleted = not project.exists()
    except Exception as exc:
        result["delete_error"] = str(exc)
    result["temp_project"] = str(project)
    result["temp_project_deleted"] = deleted
    try:
        if SMOKE_APPDATA_ROOT.exists() and "ZenAI_existing_patch_gate_appdata" in str(SMOKE_APPDATA_ROOT):
            shutil.rmtree(SMOKE_APPDATA_ROOT)
            result["temp_appdata_deleted"] = not SMOKE_APPDATA_ROOT.exists()
    except Exception as exc:
        result["temp_appdata_delete_error"] = str(exc)
    result["app_exit_code"] = exit_code
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result.get("done") and deleted else 1


if __name__ == "__main__":
    raise SystemExit(main())
