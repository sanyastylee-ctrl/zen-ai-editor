from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import app_data
from .paths import get_models_dir


GPU_BACKEND_HINTS = ("cuda", "vulkan", "kompute", "clblast", "hip", "rocm", "metal")


def _llama_cpp_lib_dir() -> Path | None:
    try:
        import llama_cpp
    except Exception:
        return None
    return Path(llama_cpp.__file__).resolve().parent / "lib"


def llama_backend_diagnostics() -> dict[str, Any]:
    lib_dir = _llama_cpp_lib_dir()
    dlls: list[str] = []
    if lib_dir and lib_dir.exists():
        dlls = sorted(p.name for p in lib_dir.iterdir() if p.is_file())
    lower = [name.lower() for name in dlls]
    gpu_dlls = [
        name for name in dlls
        if any(hint in name.lower() for hint in GPU_BACKEND_HINTS)
    ]
    return {
        "llama_cpp_lib_dir": str(lib_dir) if lib_dir else "",
        "llama_cpp_lib_exists": bool(lib_dir and lib_dir.exists()),
        "llama_cpp_lib_files": dlls,
        "gpu_backend_files": gpu_dlls,
        "cpu_backend_present": any("cpu" in name for name in lower),
        "gpu_backend_present": bool(gpu_dlls),
    }


def write_log(message: str) -> None:
    try:
        app_data.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(app_data.LOGS_DIR / "zenai.log", "a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
    except OSError:
        pass


def log_runtime_diagnostics(
    *,
    profile,
    model_path: str,
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backend = llama_backend_diagnostics()
    data: dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "frozen": bool(getattr(sys, "frozen", False)),
        "exe": str(Path(sys.executable).resolve()),
        "cwd": os.getcwd(),
        "models_dir": str(get_models_dir()),
        "model_file": getattr(profile, "model_file", ""),
        "model_path": model_path,
        "profile_id": getattr(profile, "id", ""),
        "profile_name": getattr(profile, "name", ""),
        "profile_kind": getattr(getattr(profile, "kind", ""), "value", getattr(profile, "kind", "")),
        "agent_mode": bool(getattr(profile, "agent_mode", False)),
        "n_ctx": getattr(profile, "n_ctx", None),
        "n_threads": getattr(profile, "n_threads", None),
        "n_gpu_layers": getattr(profile, "n_gpu_layers", None),
        **backend,
    }
    if extra:
        data.update(extra)

    lines = ["[runtime_diagnostics]"]
    for key in sorted(data):
        lines.append(f"{key}={data[key]}")
    write_log("\n".join(lines))

    if data.get("n_gpu_layers") == 0:
        write_log("[warning] n_gpu_layers is 0; model will run on CPU.")
    if not data.get("gpu_backend_present"):
        write_log("[warning] llama.cpp GPU backend DLL was not detected; GPU offload is unavailable.")
    return data


def acceleration_warning(profile, model_path: str) -> str:
    data = log_runtime_diagnostics(profile=profile, model_path=model_path, source="preflight")
    if data.get("n_gpu_layers") == 0:
        return "GPU offload disabled: n_gpu_layers=0. Heavy coder/agent models may run very slowly on CPU."
    if not data.get("gpu_backend_present"):
        return "GPU backend DLL not detected in llama-cpp runtime. Heavy coder/agent models will run on CPU."
    return ""
