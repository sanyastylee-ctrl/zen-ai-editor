from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import win32clipboard
from pywinauto import Desktop, keyboard, mouse


ROOT = Path(__file__).resolve().parents[1]
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


def set_clipboard(text: str) -> None:
    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(text)
    finally:
        win32clipboard.CloseClipboard()


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
    active = profiles.setdefault("active", {})
    active["coder"] = coder_id
    write_json(profiles_path, profiles)

    chat_state_path = SESSIONS / "chat_state.json"
    chat_state = read_json(chat_state_path, {})
    chat_state["last_profile_id"] = coder_id
    chat_state.setdefault("sessions_by_profile", {})
    write_json(chat_state_path, chat_state)

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


def wait_for_window(proc: subprocess.Popen, timeout: int = 120):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout = ""
            stderr = ""
            try:
                stdout, stderr = proc.communicate(timeout=1)
            except Exception:
                pass
            raise RuntimeError(
                f"ZenAI exited before opening a window: code={proc.returncode}\n"
                f"stdout={stdout}\nstderr={stderr}"
            )
        try:
            candidates = []
            for backend in ("uia", "win32"):
                desktop = Desktop(backend=backend)
                for window in desktop.windows(visible_only=False, enabled_only=False):
                    try:
                        pid = (
                            window.element_info.process_id
                            if backend == "uia"
                            else window.process_id()
                        )
                        if pid == proc.pid:
                            candidates.append(window)
                    except Exception:
                        continue
            if candidates:
                window = max(candidates, key=lambda item: item.rectangle().width() * item.rectangle().height())
                window.set_focus()
                return window
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"ZenAI window not found: {last_error}")


def send_chat_message(window) -> None:
    debug = {
        "window_title": window.window_text(),
        "window_process_id": getattr(window.element_info, "process_id", None),
        "edit_candidates": [],
        "controls": [],
    }
    focused = False
    try:
        edits = window.descendants(control_type="Edit")
    except Exception:
        edits = []
    # The chat input is the visible editable field nearest to the bottom.
    candidates = []
    for edit in edits:
        try:
            rect = edit.rectangle()
            debug["edit_candidates"].append({
                "text": edit.window_text(),
                "name": edit.element_info.name,
                "rect": [rect.left, rect.top, rect.right, rect.bottom],
            })
            if rect.width() > 120 and rect.height() > 16:
                candidates.append((rect.bottom, edit))
        except Exception:
            continue
    if candidates:
        _, edit = sorted(candidates, key=lambda item: item[0])[-1]
        try:
            edit.set_focus()
            focused = True
        except Exception:
            try:
                rect = edit.rectangle()
                mouse.click(button="left", coords=(rect.left + 12, rect.top + rect.height() // 2))
                focused = True
            except Exception:
                focused = False
    if not focused:
        rect = window.rectangle()
        x = rect.left + max(180, int(rect.width() * 0.25))
        y = rect.bottom - 34
        mouse.click(button="left", coords=(x, y))
    try:
        for ctrl in window.descendants()[:80]:
            try:
                rect = ctrl.rectangle()
                debug["controls"].append({
                    "type": ctrl.element_info.control_type,
                    "name": ctrl.element_info.name,
                    "text": ctrl.window_text(),
                    "rect": [rect.left, rect.top, rect.right, rect.bottom],
                })
            except Exception:
                continue
    except Exception:
        pass
    globals()["_LAST_SEND_DEBUG"] = debug
    time.sleep(0.5)
    set_clipboard(TASK)
    keyboard.send_keys("^v")
    time.sleep(0.3)
    keyboard.send_keys("{ENTER}")


def tail_log(start_size: int) -> str:
    if not LOG.exists():
        return ""
    with LOG.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(min(start_size, LOG.stat().st_size))
        return f.read()


def wait_for_agent_result(project: Path, start_size: int, timeout: int = 720) -> dict:
    deadline = time.time() + timeout
    last_text = ""
    while time.time() < deadline:
        last_text = tail_log(start_size)
        if "[ui_send_start]" not in last_text and time.time() > deadline - timeout + 45:
            break
        main_exists = (project / "main.py").exists()
        notes_exists = (project / "notes.json").exists()
        completed = "[agent_finished]" in last_text and "completed" in last_text
        commands = [
            'python main.py add "first task"',
            "python main.py list",
            "python main.py done 1",
            "python main.py clear",
        ]
        commands_seen = all(command in last_text for command in commands)
        if main_exists and notes_exists and completed and commands_seen:
            break
        if "[agent_blocked]" in last_text or "repair_no_progress_blocker" in last_text:
            break
        time.sleep(5)

    content = (project / "notes.json").read_text(encoding="utf-8", errors="ignore") if (project / "notes.json").exists() else ""
    command_lines = [
        line for line in last_text.splitlines()
        if "[agent_command_goal_done]" in line or "[coder_verification]" in line or "[coder_repair]" in line
    ][-40:]
    return {
        "main_exists": (project / "main.py").exists(),
        "notes_exists": (project / "notes.json").exists(),
        "notes_json": content,
        "send_reached_pipeline": "[ui_send_start]" in last_text,
        "send_debug": globals().get("_LAST_SEND_DEBUG", {}),
        "command_lines": command_lines,
        "last_log_lines": last_text.splitlines()[-80:],
    }


def main() -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    project = Path(tempfile.gettempdir()) / f"ZenAI_coder_v3_ui_smoke_{stamp}"
    project.mkdir(parents=True, exist_ok=True)
    start_size = LOG.stat().st_size if LOG.exists() else 0
    backups = prepare_state(project)
    proc = None
    result = {}
    try:
        proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        window = wait_for_window(proc)
        time.sleep(4)
        send_chat_message(window)
        result = wait_for_agent_result(project, start_size)
        return 0
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        if proc:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        restore_state(backups)
        deleted = False
        try:
            if project.exists() and "ZenAI_coder_v3_ui_smoke_" in str(project):
                shutil.rmtree(project)
                deleted = not project.exists()
        except Exception as exc:
            result["delete_error"] = str(exc)
        result["temp_project"] = str(project)
        result["temp_project_deleted"] = deleted
        print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
