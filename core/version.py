from __future__ import annotations

import os
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


APP_NAME = "ZenAI"


def _version_metadata_path() -> Path | None:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "version.json"
    return None


def _load_version_metadata() -> dict:
    path = _version_metadata_path()
    if not path or not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


_META = _load_version_metadata()


APP_VERSION = str(_META.get("version") or os.getenv("ZENAI_VERSION", "0.1.0"))
APP_CHANNEL = str(_META.get("channel") or os.getenv("ZENAI_CHANNEL", "dev"))
BUILD_ID = str(_META.get("build_id") or os.getenv("ZENAI_BUILD_ID", ""))
GIT_COMMIT = str(_META.get("git_commit") or os.getenv("ZENAI_GIT_COMMIT", ""))
BUILD_DATE = str(_META.get("build_date") or os.getenv("ZENAI_BUILD_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d")))

