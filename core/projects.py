"""
Управление "текущим проектом" редактора.

Текущий проект — это папка, которую видит:
- Дерево файлов в сайдбаре
- Терминал (cwd)
- RAG-индексатор
- Контекст кодера (_get_project_tree)

История хранится в AppData/ZenAI/sessions/recent_projects.json — последние N открытых,
LRU. Используется для меню "Недавние".

ProjectManager — singleton (как ModelManager).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .app_data import SESSIONS_DIR, migrate_legacy_file


CONFIG_DIR = SESSIONS_DIR
RECENT_FILE = CONFIG_DIR / "recent_projects.json"
LEGACY_RECENT_FILE = Path.home() / ".zen_ai" / "recent_projects.json"
MAX_RECENT = 8


class ProjectManager:
    """Singleton: текущий проект + история."""

    _instance: "ProjectManager | None" = None

    @classmethod
    def instance(cls) -> "ProjectManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._current: str = os.getcwd()
        self._recent: list[str] = []
        migrate_legacy_file(LEGACY_RECENT_FILE, RECENT_FILE)
        self._load()

    # ---------- public ----------

    @property
    def current(self) -> str:
        return self._current

    @property
    def current_name(self) -> str:
        """Имя последней папки в пути ('my-project' из '/home/u/code/my-project')."""
        return os.path.basename(os.path.normpath(self._current)) or self._current

    def recent(self) -> list[str]:
        """Возвращает список последних проектов (без текущего, существующие на диске)."""
        return [p for p in self._recent if p != self._current and os.path.isdir(p)]

    def open(self, path: str) -> bool:
        """
        Переключиться на проект. Возвращает True если действительно сменили,
        False если та же папка или путь не существует.
        """
        if not path:
            return False
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            return False
        if abs_path == self._current:
            return False

        self._current = abs_path
        # перемещаем в начало истории
        if abs_path in self._recent:
            self._recent.remove(abs_path)
        self._recent.insert(0, abs_path)
        # обрезаем
        self._recent = self._recent[:MAX_RECENT]
        self._save()
        # cwd — для терминала и относительных путей
        os.chdir(abs_path)
        return True

    def remove_from_recent(self, path: str) -> None:
        if path in self._recent:
            self._recent.remove(path)
            self._save()

    # ---------- io ----------

    def _load(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            if RECENT_FILE.exists():
                with open(RECENT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._recent = [p for p in data.get("recent", []) if isinstance(p, str)]
                last = data.get("last_opened")
                if last and os.path.isdir(last):
                    self._current = last
                    os.chdir(last)
        except Exception:
            # битый конфиг — игнорируем, начнём с чистого
            self._recent = []

    def _save(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            tmp = RECENT_FILE.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({
                    "version": 1,
                    "last_opened": self._current,
                    "recent": self._recent,
                }, f, ensure_ascii=False, indent=2)
            tmp.replace(RECENT_FILE)
        except Exception:
            pass
