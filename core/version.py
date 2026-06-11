from __future__ import annotations

import os
from datetime import datetime, timezone


APP_NAME = "ZenAI"
APP_VERSION = os.getenv("ZENAI_VERSION", "0.1.0")
APP_CHANNEL = os.getenv("ZENAI_CHANNEL", "dev")
BUILD_ID = os.getenv("ZENAI_BUILD_ID", "")
GIT_COMMIT = os.getenv("ZENAI_GIT_COMMIT", "")
BUILD_DATE = os.getenv("ZENAI_BUILD_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

