from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from core.companion import is_companion_echo_response, normalize_companion_reply_for_repeat
from core.model_manager import ModelManager
from core.profiles import DEFAULT_COMPANION_MODEL_FILE, ProfileKind
from ui.main_window import ZenEditor


def wait_for_turn(app: QApplication, window: ZenEditor, timeout_s: int = 120) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        app.processEvents()
        worker = getattr(window, "worker", None)
        if worker is None:
            app.processEvents()
            return
        time.sleep(0.02)
    raise TimeoutError("UI companion turn timed out")


def main() -> int:
    ModelManager.instance().set_max_loaded(1)
    app = QApplication.instance() or QApplication([])
    window = ZenEditor()
    profile = window.pm.get_active(ProfileKind.COMPANION)
    if profile is None:
        raise RuntimeError("No active companion profile")
    profile.model_file = DEFAULT_COMPANION_MODEL_FILE
    profile.n_gpu_layers = -1
    profile.max_tokens = 96
    profile.n_ctx = 8192
    profile.persona = dict(profile.persona or {})
    profile.persona["memory_enabled"] = "false"
    window.profile_switcher.set_active(profile.id)
    window.pm.active[ProfileKind.COMPANION] = profile.id
    window._histories[profile.id] = []
    window._persist_chat_session = lambda profile_id: None
    window._capture_companion_memory = lambda profile_id, user_msg: None

    short_messages = [
        "Привет",
        "Тут?",
        "Скажи новую фразу про кофе.",
        "А теперь совсем другую про дождь.",
        "Что ты сейчас делаешь?",
    ]
    seeds = [
        "Привет",
        "Тут?",
        "Продолжай",
        "Что думаешь?",
        "Коротко ответь про музыку",
        "Теперь про тёплый свет",
        "Маркер диалога",
    ]
    long_messages = [f"ui-msg-{i:03d}: {seeds[(i - 1) % len(seeds)]}" for i in range(1, 61)]
    results: list[dict] = []

    for phase, messages in [("short", short_messages), ("long", long_messages)]:
        for message in messages:
            before = len(window._histories.get(profile.id, []))
            window.chat_input.setText(message)
            window.send_message()
            wait_for_turn(app, window)
            history = window._histories.get(profile.id, [])
            if history and history[-1][0] == message:
                saved_user, response = history[-1]
                saved = saved_user == message
            else:
                response = ""
                saved = False
            is_echo, similarity = is_companion_echo_response(message, response)
            norm = normalize_companion_reply_for_repeat(response)
            replay = bool(norm and norm in [normalize_companion_reply_for_repeat(item[1]) for item in history[:-1][-5:]])
            valid = saved and bool(response.strip()) and not is_echo and not replay
            result = {
                "phase": phase,
                "user": message,
                "response": response,
                "saved": saved,
                "valid": valid,
                "echo": is_echo,
                "echo_similarity": round(similarity, 3),
                "replay": replay,
            }
            results.append(result)
            if phase == "short" or not valid or message.endswith("001: Привет") or message.startswith("ui-msg-045") or message.startswith("ui-msg-060"):
                print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "model": DEFAULT_COMPANION_MODEL_FILE,
        "turns": len(results),
        "valid_turns": sum(1 for item in results if item["valid"]),
        "all_passed": all(item["valid"] for item in results),
        "failures": [item for item in results if not item["valid"]],
        "short": results[: len(short_messages)],
        "long_tail": results[-10:],
    }
    out = ROOT / ".agent_smoke" / "lera_ui_pipeline_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    window.close()
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
