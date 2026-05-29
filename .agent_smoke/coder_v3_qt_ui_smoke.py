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

TASK = """Создай CLI todo app. Команды должны работать:
python main.py add "first task"
python main.py list
python main.py done 1
python main.py clear
Храни задачи в notes.json. После создания сам проверь все команды через терминал."""


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    project = Path(tempfile.gettempdir()) / f"ZenAI_coder_v3_qt_smoke_{stamp}"
    project.mkdir(parents=True, exist_ok=True)
    start_size = LOG.stat().st_size if LOG.exists() else 0
    backups = prepare_state(project)
    result: dict = {}

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = ZenEditor()
    window.show()

    def send() -> None:
        window.chat_input.setText(TASK)
        window.send_message()

    QTimer.singleShot(1500, send)

    deadline = time.time() + 720

    def poll() -> None:
        text = tail_log(start_size)
        commands = [
            'python main.py add "first task"',
            "python main.py list",
            "python main.py done 1",
            "python main.py clear",
        ]
        done = (
            (project / "main.py").exists()
            and (project / "notes.json").exists()
            and all(command in text for command in commands)
            and "[agent_finished]" in text
            and "completed" in text
        )
        blocked = "[agent_blocked]" in text or "repair_no_progress_blocker" in text
        if done or blocked or time.time() > deadline:
            result.update({
                "done": done,
                "blocked": blocked,
                "main_exists": (project / "main.py").exists(),
                "notes_exists": (project / "notes.json").exists(),
                "notes_json": (project / "notes.json").read_text(encoding="utf-8", errors="ignore") if (project / "notes.json").exists() else "",
                "relevant_logs": [
                    line for line in text.splitlines()
                    if "[ui_send_" in line
                    or "[agent_command_goal_done]" in line
                    or "[coder_verification]" in line
                    or "[coder_repair]" in line
                    or "[coder_evaluator]" in line
                    or "[agent_finished]" in line
                ][-80:],
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
        if project.exists() and "ZenAI_coder_v3_qt_smoke_" in str(project):
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
