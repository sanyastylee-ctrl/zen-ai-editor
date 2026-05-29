"""
Application-owned persistent storage paths.

User data is kept outside opened projects so project trees stay source-only and
the same storage works for both source runs and packaged Windows builds.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any


_appdata = os.getenv("APPDATA")
APP_DIR = Path(_appdata) / "ZenAI" if _appdata else Path.home() / "AppData" / "Roaming" / "ZenAI"
CHATS_DIR = APP_DIR / "chats"
SESSIONS_DIR = APP_DIR / "sessions"
MEMORY_DIR = APP_DIR / "memory"
SETTINGS_DIR = APP_DIR / "settings"
LOGS_DIR = APP_DIR / "logs"

for _directory in (APP_DIR, CHATS_DIR, SESSIONS_DIR, MEMORY_DIR, SETTINGS_DIR, LOGS_DIR):
    try:
        _directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Storage must never prevent the application from starting.
        pass


def atomic_write_json(path: Path, data: Any) -> bool:
    """Write JSON atomically so an interrupted close cannot corrupt user data."""
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def migrate_legacy_file(legacy_path: Path, target_path: Path) -> None:
    """Copy an existing legacy data file once, without overwriting new data."""
    if target_path.exists() or not legacy_path.exists():
        return
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_path, target_path)
    except OSError:
        pass


def project_memory_dir(project_root: str) -> Path:
    """Stable per-project cache location without writing into the project."""
    root = os.path.realpath(project_root)
    key = hashlib.sha256(os.path.normcase(root).encode("utf-8")).hexdigest()[:16]
    directory = MEMORY_DIR / "projects" / key
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return directory
