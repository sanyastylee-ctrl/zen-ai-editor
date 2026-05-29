from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QStyleFactory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.main_window import ZenEditor  # noqa: E402


APPDATA = Path(os.environ["APPDATA"]) / "ZenAI"
SETTINGS = APPDATA / "settings"
SESSIONS = APPDATA / "sessions"
LOG = APPDATA / "logs" / "zenai.log"

TASK = (
    "Запусти проект, найди ошибку по traceback и исправь так, чтобы программа "
    "больше не падала. После исправления снова запусти команду."
)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_project(project: Path) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / "main.py").write_text(
        "from calculator import divide\n\nprint(divide(10, 0))\n",
        encoding="utf-8",
    )
    (project / "calculator.py").write_text(
        "def divide(a, b):\n    return a / b\n",
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
        raise RuntimeError("coder profile not found")
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


def main() -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    project = Path(tempfile.gettempdir()) / f"ZenAI_traceback_gate_{stamp}"
    prepare_project(project)
    start_size = LOG.stat().st_size if LOG.exists() else 0
    backups = prepare_state(project)
    result: dict = {"tool_events": []}
    connected_workers: set[int] = set()

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = ZenEditor()
    window.show()

    def send() -> None:
        window.chat_input.setText(TASK)
        window.send_message()

    QTimer.singleShot(1500, send)
    deadline = time.time() + 720

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
        terminal_outputs = [
            str(item.get("output", ""))
            for item in tool_events
            if item.get("name") == "run_terminal"
        ]
        traceback_seen = any("ZeroDivisionError" in output for output in terminal_outputs) or "ZeroDivisionError" in text
        rerun_ok = any("python main.py" in output and "[exit 0]" in output for output in terminal_outputs)
        calculator_text = (project / "calculator.py").read_text(encoding="utf-8", errors="ignore")
        changed = "return a / b" not in calculator_text or "b == 0" in calculator_text
        repair_logged = "[coder_repair]" in text and "traceback_error" in text
        final_allowed = "[coder_evaluator]" in text and "allowed=true" in text.lower()
        finished = "[agent_finished]" in text and "completed" in text
        done = traceback_seen and repair_logged and changed and rerun_ok and final_allowed and finished
        blocked = "[agent_blocked]" in text or "repair_no_progress_blocker" in text
        if done or blocked or time.time() > deadline:
            result.update({
                "done": done,
                "blocked": blocked,
                "timeout": time.time() > deadline,
                "main_py": (project / "main.py").read_text(encoding="utf-8", errors="ignore"),
                "calculator_py": calculator_text,
                "traceback_seen": traceback_seen,
                "repair_logged": repair_logged,
                "rerun_ok": rerun_ok,
                "final_allowed": final_allowed,
                "finished": finished,
                "relevant_logs": [
                    line for line in text.splitlines()
                    if "[ui_send_" in line
                    or "[agent_tool_" in line
                    or "[agent_command_goal" in line
                    or "[coder_repair]" in line
                    or "[coder_evaluator]" in line
                    or "[agent_finished]" in line
                    or "ZeroDivisionError" in line
                ][-100:],
            })
            window.close()
            app.quit()
            return
        QTimer.singleShot(5000, poll)

    QTimer.singleShot(5000, poll)
    exit_code = app.exec()

    os.chdir(str(ROOT))
    restore_state(backups)
    deleted = False
    try:
        if project.exists() and "ZenAI_traceback_gate_" in str(project):
            shutil.rmtree(project)
            deleted = not project.exists()
    except Exception as exc:
        result["delete_error"] = str(exc)
    result["temp_project"] = str(project)
    result["temp_project_deleted"] = deleted
    result["app_exit_code"] = exit_code
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result.get("done") and deleted else 1


if __name__ == "__main__":
    raise SystemExit(main())
