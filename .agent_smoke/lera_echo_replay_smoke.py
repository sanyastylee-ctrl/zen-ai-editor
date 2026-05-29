from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.worker import InferenceWorker
from core.companion import is_companion_echo_response, normalize_companion_reply_for_repeat
from core.model_manager import ModelManager
from core.profiles import (
    AIProfile,
    ChatTemplate,
    DEFAULT_COMPANION_MODEL_FILE,
    DEFAULT_COMPANION_PROMPT,
    ProfileKind,
)


def companion_profile() -> AIProfile:
    return AIProfile(
        id="lera-real-smoke",
        name="Лера",
        kind=ProfileKind.COMPANION,
        model_file=DEFAULT_COMPANION_MODEL_FILE,
        chat_template=ChatTemplate.CHATML,
        n_ctx=8192,
        n_gpu_layers=-1,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repeat_penalty=1.09,
        max_tokens=96,
        system_prompt=DEFAULT_COMPANION_PROMPT,
        persona={
            "character_name": "Лера",
            "age": "23",
            "appearance": "светлые волосы до плеч, серо-голубые глаза",
            "personality": "тёплая, живая, любопытная, немного игривая",
            "speaking_style": "живой разговорный, без формальностей",
            "background": "любит уютные вечера и творческие разговоры",
            "current_mood": "спокойное",
            "relationship_to_user": "близкая подруга / девушка",
            "user_name": "",
            "companion_mode": "chat",
            "memory_enabled": "false",
        },
    )


def run_turn(profile: AIProfile, user: str, history: list[tuple[str, str]]) -> dict:
    worker = InferenceWorker(
        profile,
        user,
        history=list(history),
        max_generation_seconds=90,
    )
    chunks: list[str] = []
    loads: list[dict] = []
    worker.chunk_received.connect(chunks.append)
    worker.model_loading.connect(lambda path: loads.append({"event": "model_loading", "path": path}))
    worker.model_loaded.connect(lambda path, ok, error: loads.append({"event": "model_loaded", "path": path, "ok": ok, "error": error}))
    worker.run()
    response = "".join(chunks).strip()
    is_echo, echo_similarity = is_companion_echo_response(user, response)
    response_norm = normalize_companion_reply_for_repeat(response)
    prior_norms = [normalize_companion_reply_for_repeat(item[1]) for item in history[-5:]]
    replay = bool(response_norm and response_norm in prior_norms)
    valid = bool(getattr(worker, "companion_response_valid", True)) and response and not is_echo and not replay
    if valid:
        history.append((user, response))
    return {
        "user": user,
        "response": response,
        "valid": valid,
        "worker_valid": bool(getattr(worker, "companion_response_valid", True)),
        "block_reason": getattr(worker, "companion_block_reason", ""),
        "echo": is_echo,
        "echo_similarity": round(echo_similarity, 3),
        "replay": replay,
        "loads": loads,
    }


def main() -> int:
    ModelManager.instance().set_max_loaded(1)
    profile = companion_profile()
    history: list[tuple[str, str]] = []
    short_messages = [
        "Привет",
        "Тут?",
        "Скажи новую фразу про кофе.",
        "А теперь совсем другую про дождь.",
        "Что ты сейчас делаешь?",
    ]
    long_messages = []
    seeds = [
        "Привет",
        "Тут?",
        "Продолжай",
        "Что думаешь?",
        "Коротко ответь про музыку",
        "Теперь про тёплый свет",
        "Маркер диалога",
    ]
    for i in range(1, 61):
        long_messages.append(f"msg-{i:03d}: {seeds[(i - 1) % len(seeds)]}")

    results: list[dict] = []
    for message in short_messages:
        result = run_turn(profile, message, history)
        result["phase"] = "short"
        results.append(result)
        print(json.dumps({k: result[k] for k in ("phase", "user", "response", "valid", "echo", "replay", "block_reason")}, ensure_ascii=False), flush=True)

    for message in long_messages:
        result = run_turn(profile, message, history)
        result["phase"] = "long"
        results.append(result)
        if not result["valid"] or int(message[4:7]) in {1, 20, 45, 60}:
            print(json.dumps({k: result[k] for k in ("phase", "user", "response", "valid", "echo", "replay", "block_reason")}, ensure_ascii=False), flush=True)

    summary = {
        "model": DEFAULT_COMPANION_MODEL_FILE,
        "turns": len(results),
        "valid_turns": sum(1 for item in results if item["valid"]),
        "all_passed": all(item["valid"] for item in results),
        "failures": [item for item in results if not item["valid"]],
        "short": results[: len(short_messages)],
        "long_tail": results[-10:],
    }
    out = ROOT / ".agent_smoke" / "lera_echo_replay_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
