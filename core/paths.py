"""
Пути к моделям и проектным файлам.
"""

from __future__ import annotations

import sys
from pathlib import Path


def get_app_root() -> Path:
    """Return the application root, independent from opened project/cwd."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_models_dir() -> Path:
    if getattr(sys, "frozen", False):
        # Packaged builds keep large GGUF files external, next to the app.
        return get_app_root() / "models"
    return get_app_root() / "models"


def resource_path(relative_path: str) -> Path:
    """Resolve bundled application assets for source and PyInstaller runs."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).resolve().parent.parent / relative_path


def resolve_model_path(model_file: str) -> str:
    """
    Превращает имя файла (qwen2.5-coder-14b.gguf) в полный путь.
    Если передан абсолютный путь — возвращаем как есть.
    """
    if not model_file:
        return ""
    p = Path(model_file)
    if p.is_absolute():
        return str(p)
    return str(get_models_dir() / model_file)


def list_available_models() -> list[str]:
    """Возвращает имена .gguf файлов в папке моделей."""
    models_dir = get_models_dir()
    if not models_dir.exists():
        return []
    return sorted([
        f.name for f in models_dir.iterdir()
        if f.is_file() and f.suffix == ".gguf"
    ])
