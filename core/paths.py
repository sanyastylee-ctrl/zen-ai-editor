"""
Пути к моделям и проектным файлам.
"""

from __future__ import annotations

import os
from pathlib import Path


# Папка с моделями относительно cwd проекта.
# Можно переопределить через переменную окружения ZEN_AI_MODELS_DIR.
def get_models_dir() -> Path:
    custom = os.environ.get("ZEN_AI_MODELS_DIR")
    if custom:
        return Path(custom).expanduser()
    return Path.cwd() / "models"


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
        models_dir.mkdir(parents=True, exist_ok=True)
        return []
    return sorted([
        f.name for f in models_dir.iterdir()
        if f.is_file() and f.suffix == ".gguf"
    ])
