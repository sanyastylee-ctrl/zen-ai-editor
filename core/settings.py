import os
import json
from pathlib import Path

from .app_data import SETTINGS_DIR, migrate_legacy_file

_ZEN_CONFIG_DIR = str(SETTINGS_DIR)
_SETTINGS_PATH = str(SETTINGS_DIR / "settings.json")
_LEGACY_SETTINGS_PATH = Path.home() / ".zen_ai" / "settings.json"

class PersistentSettings:
    DEFAULT = {
        'coder_model': '', 'assistant_model': '',
        'system_prompt': 'Ты — полезный ИИ-ассистент. Отвечай по существу.',
        'coder_system_prompt': 'Ты — опытный программист. Пиши чистый рабочий код. Отвечай кратко.',
        'temperature': 0.7, 'max_tokens': 2048, 'n_ctx': 4096,
        'diff_before_apply': True, 'use_rag': False,
        'agent_confirmation_policy': 'confirm_changes',
        'comfyui_enabled': False, 'comfyui_url': 'http://127.0.0.1:8188',
        'comfyui_steps': 20, 'comfyui_cfg': 7.0,
    }

    @staticmethod
    def load() -> dict:
        base = dict(PersistentSettings.DEFAULT)
        try:
            migrate_legacy_file(_LEGACY_SETTINGS_PATH, Path(_SETTINGS_PATH))
            if os.path.exists(_SETTINGS_PATH):
                with open(_SETTINGS_PATH, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                base.update(saved)
        except Exception:
            pass
        return base

    @staticmethod
    def save(settings: dict):
        try:
            os.makedirs(_ZEN_CONFIG_DIR, exist_ok=True)
            tmp = _SETTINGS_PATH + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            os.replace(tmp, _SETTINGS_PATH)
        except Exception:
            pass
