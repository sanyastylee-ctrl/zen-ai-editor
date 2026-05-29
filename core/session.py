import os
import re
from pathlib import Path

from .app_data import SESSIONS_DIR, atomic_write_json, read_json

_RECOVERY_DIR = str(SESSIONS_DIR / "recovery")
_SESSION_FILE = str(SESSIONS_DIR / "editor_session.json")

class SessionManager:
    @staticmethod
    def save(editor_tabs):
        try:
            os.makedirs(_RECOVERY_DIR, exist_ok=True)
            tabs_info = []
            for i, td in enumerate(editor_tabs._tabs):
                fp   = td['file_path']
                text = editor_tabs._get_text_for_editor(td['editor'])
                rec  = None
                if td['modified'] or not fp:
                    safe_name = re.sub(r'[^\w.]', '_', os.path.basename(fp)) if fp else 'untitled'
                    rec = os.path.join(_RECOVERY_DIR, f"tab{i}_{safe_name}.bak")
                    with open(rec, 'w', encoding='utf-8') as f:
                        f.write(text)
                tabs_info.append({
                    'file_path': fp,
                    'recovery':  rec,
                    'modified':  td['modified'],
                })
            session = {
                'tabs':        tabs_info,
                'current_tab': editor_tabs.currentIndex(),
            }
            atomic_write_json(Path(_SESSION_FILE), session)
        except Exception:
            pass

    @staticmethod
    def load() -> dict | None:
        data = read_json(Path(_SESSION_FILE), None)
        return data if isinstance(data, dict) else None

    @staticmethod
    def clear():
        try:
            if os.path.exists(_SESSION_FILE):
                os.remove(_SESSION_FILE)
            if os.path.isdir(_RECOVERY_DIR):
                for bak in os.listdir(_RECOVERY_DIR):
                    if bak.endswith('.bak'):
                        os.remove(os.path.join(_RECOVERY_DIR, bak))
        except Exception:
            pass
