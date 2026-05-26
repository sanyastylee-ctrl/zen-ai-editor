import os
import json
import re

_RECOVERY_DIR  = os.path.join(os.getcwd(), '.zen_ai', 'recovery')
_SESSION_FILE  = os.path.join(os.getcwd(), '.zen_ai', 'session.json')

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
            os.makedirs(os.path.dirname(_SESSION_FILE), exist_ok=True)
            with open(_SESSION_FILE, 'w', encoding='utf-8') as f:
                json.dump(session, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def load() -> dict | None:
        try:
            if os.path.exists(_SESSION_FILE):
                with open(_SESSION_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return None

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