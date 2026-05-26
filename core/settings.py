import os
import json

_ZEN_CONFIG_DIR  = os.path.join(os.path.expanduser("~"), ".zen_ai")
_SETTINGS_PATH   = os.path.join(_ZEN_CONFIG_DIR, "settings.json")

class PersistentSettings:
    DEFAULT = {
        'coder_model': '', 'assistant_model': '',
        'system_prompt': 'Ты — полезный ИИ-ассистент. Отвечай по существу.',
        'coder_system_prompt': 'Ты — опытный программист. Пиши чистый рабочий код. Отвечай кратко.',
        'temperature': 0.7, 'max_tokens': 2048, 'n_ctx': 4096,
        'diff_before_apply': True, 'use_rag': False,
        'comfyui_enabled': False, 'comfyui_url': 'http://127.0.0.1:8188',
        'comfyui_steps': 20, 'comfyui_cfg': 7.0,
    }

    @staticmethod
    def load() -> dict:
        base = dict(PersistentSettings.DEFAULT)
        try:
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
            with open(_SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception:
            pass