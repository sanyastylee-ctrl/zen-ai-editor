from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import ssl
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .app_data import APP_DIR
from .version import APP_CHANNEL, APP_NAME, APP_VERSION


UPDATES_DIR = APP_DIR / "updates"
DEFAULT_TIMEOUT_SECONDS = 10
PROTECTED_INSTALL_NAMES = {"models"}
MAX_UPDATE_FILE_COUNT = 20000
MAX_UPDATE_TOTAL_SIZE = 3 * 1024 * 1024 * 1024


class UpdateError(Exception):
    """Clean, user-facing update error."""


@dataclass(frozen=True)
class UpdateManifest:
    app: str
    channel: str
    latest_version: str
    published_at: str
    min_supported_version: str
    update_url: str
    sha256: str
    size: int
    release_notes: list[str] = field(default_factory=list)
    requires_full_update: bool = True
    critical: bool = False


@dataclass(frozen=True)
class UpdateCheckResult:
    available: bool
    current_version: str
    manifest: UpdateManifest | None = None
    message: str = ""


@dataclass(frozen=True)
class UpdaterResult:
    ok: bool
    backup_dir: Path | None
    staging_dir: Path | None
    message: str
    relaunched: bool = False


def updates_dir() -> Path:
    UPDATES_DIR.mkdir(parents=True, exist_ok=True)
    return UPDATES_DIR


def load_current_version() -> str:
    return APP_VERSION


def _clean_note(text: Any) -> str:
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", str(text or ""))
    return clean.strip()[:500]


def compare_versions(current: str, latest: str) -> int:
    def parts(value: str) -> list[Any]:
        out: list[Any] = []
        for piece in re.split(r"[.\-+_]", value.strip()):
            if piece.isdigit():
                out.append(int(piece))
            elif piece:
                out.append(piece.lower())
        return out

    left = parts(current)
    right = parts(latest)
    max_len = max(len(left), len(right))
    left.extend([0] * (max_len - len(left)))
    right.extend([0] * (max_len - len(right)))
    for a, b in zip(left, right):
        if a == b:
            continue
        if isinstance(a, int) and isinstance(b, int):
            return 1 if a > b else -1
        return 1 if str(a) > str(b) else -1
    return 0


def _validate_update_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UpdateError("Update URL must use http or https.")
    if not parsed.netloc:
        raise UpdateError("Update URL is missing a host.")


def _sanitize_diagnostic_text(text: Any, *, limit: int = 240) -> str:
    clean = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or ""))
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:limit]


def _has_exception_type(exc: BaseException, kind: type[BaseException]) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, kind):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _exception_chain_type(exc: BaseException) -> str:
    names: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        names.append(type(current).__name__)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return "->".join(names)


def parse_manifest(data: str | bytes | dict[str, Any], *, current_version: str | None = None, channel: str | None = None) -> UpdateManifest:
    raw: dict[str, Any]
    if isinstance(data, dict):
        raw = data
    else:
        try:
            raw = json.loads(data)
        except json.JSONDecodeError as exc:
            raise UpdateError("Manifest is not valid JSON.") from exc

    current_version = current_version or APP_VERSION
    channel = channel or APP_CHANNEL
    app = str(raw.get("app", "")).strip()
    manifest_channel = str(raw.get("channel", "")).strip()
    latest_version = str(raw.get("latest_version", "")).strip()
    update_url = str(raw.get("update_url", "")).strip()
    sha256 = str(raw.get("sha256", "")).strip().lower()

    if app != APP_NAME:
        raise UpdateError("Manifest is for a different application.")
    if manifest_channel != channel:
        raise UpdateError("Manifest channel does not match current channel.")
    if not latest_version:
        raise UpdateError("Manifest is missing latest_version.")
    if compare_versions(current_version, latest_version) >= 0:
        raise UpdateError("No newer version is available.")
    _validate_update_url(update_url)
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise UpdateError("Manifest sha256 must be a 64-character hex string.")
    try:
        size = int(raw.get("size", 0))
    except (TypeError, ValueError) as exc:
        raise UpdateError("Manifest size must be a positive integer.") from exc
    if size <= 0:
        raise UpdateError("Manifest size must be positive.")

    release_notes = raw.get("release_notes") or []
    if not isinstance(release_notes, list):
        release_notes = [release_notes]

    return UpdateManifest(
        app=app,
        channel=manifest_channel,
        latest_version=latest_version,
        published_at=str(raw.get("published_at", "")).strip(),
        min_supported_version=str(raw.get("min_supported_version", "")).strip(),
        update_url=update_url,
        sha256=sha256,
        size=size,
        release_notes=[note for note in (_clean_note(item) for item in release_notes) if note],
        requires_full_update=bool(raw.get("requires_full_update", True)),
        critical=bool(raw.get("critical", False)),
    )


def fetch_manifest(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    _validate_update_url(url)
    if diagnostics is not None:
        diagnostics.update(
            {
                "manifest_url": url,
                "request_start": True,
                "timeout_seconds": timeout,
            }
        )
    request = urllib.request.Request(url, headers={"User-Agent": "ZenAI-Updater"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(256 * 1024)
            text = body.decode("utf-8", errors="replace")
            if diagnostics is not None:
                diagnostics.update(
                    {
                        "request_done": True,
                        "http_status": getattr(response, "status", None) or response.getcode(),
                        "content_type": response.headers.get("content-type", ""),
                        "response_size": len(body),
                    }
                )
            return text
    except (OSError, urllib.error.URLError) as exc:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "request_done": False,
                    "exception_type": _exception_chain_type(exc),
                    "exception_message": _sanitize_diagnostic_text(exc),
                    "ssl_error": _has_exception_type(exc, ssl.SSLError),
                    "timeout": _has_exception_type(exc, TimeoutError)
                    or _has_exception_type(exc, socket.timeout),
                }
            )
        raise UpdateError("Не удалось проверить обновления.") from exc


def check_for_update(
    manifest_url: str,
    *,
    current_version: str | None = None,
    channel: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> UpdateCheckResult:
    current_version = current_version or APP_VERSION
    channel = channel or APP_CHANNEL
    if diagnostics is not None:
        diagnostics.update(
            {
                "manifest_url": manifest_url,
                "current_version": current_version,
                "current_channel": channel,
                "manifest_parse_ok": False,
            }
        )
    manifest_text = ""
    try:
        manifest_text = fetch_manifest(manifest_url, diagnostics=diagnostics)
        manifest = parse_manifest(manifest_text, current_version=current_version, channel=channel)
    except UpdateError as exc:
        if diagnostics is not None and manifest_text:
            diagnostics.update(
                {
                    "manifest_parse_ok": False,
                    "parse_exception_type": type(exc).__name__,
                    "parse_exception_message": _sanitize_diagnostic_text(exc),
                    "first_120_chars_sanitized": _sanitize_diagnostic_text(manifest_text, limit=120),
                }
            )
        if "No newer version" in str(exc):
            return UpdateCheckResult(False, current_version, message="У вас последняя версия.")
        raise
    if diagnostics is not None:
        diagnostics.update(
            {
                "manifest_parse_ok": True,
                "latest_version": manifest.latest_version,
                "update_url": manifest.update_url,
                "sha256": manifest.sha256,
                "size": manifest.size,
            }
        )
    return UpdateCheckResult(True, current_version, manifest=manifest, message="Доступно обновление.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_hash: str) -> bool:
    return sha256_file(path) == expected_hash.lower()


def download_update(update_url: str, dest: Path, *, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Path:
    _validate_update_url(update_url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    request = urllib.request.Request(update_url, headers={"User-Agent": "ZenAI-Updater"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(dest)
        return dest
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise UpdateError("Не удалось скачать обновление.") from exc


def prepare_update_package(manifest: UpdateManifest, *, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or updates_dir()
    package_name = f"ZenAI-v{manifest.latest_version}-update.zip"
    package_path = cache_dir / package_name
    download_update(manifest.update_url, package_path)
    if not verify_sha256(package_path, manifest.sha256):
        package_path.unlink(missing_ok=True)
        raise UpdateError("Хэш обновления не совпал. Пакет удалён.")
    inspect_update_zip(package_path)
    return package_path


def _normalize_zip_name(name: str) -> str:
    return name.replace("\\", "/").strip()


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def inspect_update_zip(package_path: Path, *, allow_model_update: bool = False) -> list[str]:
    safe_names: list[str] = []
    total_size = 0
    try:
        with zipfile.ZipFile(package_path) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_UPDATE_FILE_COUNT:
                raise UpdateError("Update zip contains too many files.")
            for info in infos:
                name = _normalize_zip_name(info.filename)
                if not name or name.endswith("/"):
                    continue
                if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
                    raise UpdateError("Update zip contains an absolute path.")
                parts = [part for part in name.split("/") if part]
                if any(part == ".." for part in parts):
                    raise UpdateError("Update zip contains path traversal.")
                if _is_symlink(info):
                    raise UpdateError("Update zip contains a symlink.")
                lower_parts = [part.lower() for part in parts]
                if (not allow_model_update) and (
                    (lower_parts and lower_parts[0] == "models") or name.lower().endswith(".gguf")
                ):
                    raise UpdateError("Update zip must not contain models or GGUF files.")
                total_size += int(info.file_size)
                if total_size > MAX_UPDATE_TOTAL_SIZE:
                    raise UpdateError("Update zip is too large.")
                safe_names.append(name)
    except zipfile.BadZipFile as exc:
        raise UpdateError("Update package is not a valid zip.") from exc
    if not any(name.lower().endswith("zenai.exe") for name in safe_names):
        raise UpdateError("Update zip does not contain ZenAI.exe.")
    return safe_names


def _payload_root(staging_dir: Path) -> Path:
    direct = staging_dir / "ZenAI.exe"
    if direct.exists():
        return staging_dir
    children = [child for child in staging_dir.iterdir() if child.is_dir()]
    if len(children) == 1 and (children[0] / "ZenAI.exe").exists():
        return children[0]
    return staging_dir


def safe_extract_zip(package_path: Path, staging_dir: Path) -> Path:
    inspect_update_zip(package_path)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path) as zf:
        for info in zf.infolist():
            name = _normalize_zip_name(info.filename)
            if not name or name.endswith("/"):
                continue
            target = staging_dir / Path(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return _payload_root(staging_dir)


def _copy_install_backup(install_dir: Path, backup_dir: Path) -> None:
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True)
    for item in install_dir.iterdir():
        if item.name.lower() in PROTECTED_INSTALL_NAMES:
            continue
        dest = backup_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, ignore=shutil.ignore_patterns("*.tmp", "updates"))
        else:
            shutil.copy2(item, dest)


def _remove_replaceable_install_files(install_dir: Path) -> None:
    for item in install_dir.iterdir():
        if item.name.lower() in PROTECTED_INSTALL_NAMES:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _copy_payload_to_install(payload_dir: Path, install_dir: Path) -> None:
    for item in payload_dir.iterdir():
        if item.name.lower() in PROTECTED_INSTALL_NAMES:
            continue
        dest = install_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def _rollback_from_backup(install_dir: Path, backup_dir: Path) -> None:
    _remove_replaceable_install_files(install_dir)
    _copy_payload_to_install(backup_dir, install_dir)


def _unique_backup_dir(install_dir: Path) -> Path:
    base = install_dir.parent / f"{install_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = install_dir.parent / f"{base.name}_{counter}"
        counter += 1
    return candidate


def _write_update_log(log_path: Path | None, message: str) -> None:
    if not log_path:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
    except OSError:
        pass


def wait_for_pid_exit(pid: int | None, *, timeout: int = 60) -> None:
    if not pid or pid <= 0:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.5)
    raise UpdateError("Timed out waiting for ZenAI to close.")


def apply_update_package(
    *,
    install_dir: Path,
    package_path: Path,
    expected_sha256: str,
    app_exe: str = "ZenAI.exe",
    pid: int | None = None,
    relaunch: bool = False,
    log_path: Path | None = None,
) -> UpdaterResult:
    install_dir = install_dir.resolve()
    package_path = package_path.resolve()
    backup_dir = _unique_backup_dir(install_dir)
    staging_dir = install_dir.parent / f"{install_dir.name}_update_staging"
    backup_created = False
    try:
        _write_update_log(log_path, "waiting for main app")
        wait_for_pid_exit(pid)
        _write_update_log(log_path, "verifying package")
        if not verify_sha256(package_path, expected_sha256):
            raise UpdateError("Package sha256 mismatch.")
        _write_update_log(log_path, "extracting package")
        payload_dir = safe_extract_zip(package_path, staging_dir)
        _write_update_log(log_path, "creating backup")
        _copy_install_backup(install_dir, backup_dir)
        backup_created = True
        _write_update_log(log_path, "replacing files")
        _remove_replaceable_install_files(install_dir)
        if os.getenv("ZENAI_UPDATER_TEST_FAIL_AFTER_REMOVE") == "1":
            raise UpdateError("Simulated updater failure after removing replaceable files.")
        _copy_payload_to_install(payload_dir, install_dir)
        relaunched = False
        if relaunch:
            exe_path = install_dir / app_exe
            if not exe_path.exists():
                raise UpdateError("Updated executable is missing.")
            subprocess.Popen([str(exe_path)], cwd=str(install_dir))
            relaunched = True
        _write_update_log(log_path, "update complete")
        return UpdaterResult(True, backup_dir, staging_dir, "Update installed.", relaunched=relaunched)
    except Exception as exc:
        _write_update_log(log_path, f"update failed: {exc}")
        if backup_created and backup_dir.exists():
            try:
                _rollback_from_backup(install_dir, backup_dir)
                _write_update_log(log_path, "rollback complete")
            except Exception as rollback_exc:
                _write_update_log(log_path, f"rollback failed: {rollback_exc}")
                raise
        if isinstance(exc, UpdateError):
            raise
        raise UpdateError("Update failed and was rolled back.") from exc
    finally:
        try:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
        except OSError:
            pass


def build_updater_command(
    *,
    updater_exe: Path,
    install_dir: Path,
    package_path: Path,
    expected_sha256: str,
    app_exe: str = "ZenAI.exe",
    pid: int | None = None,
    relaunch: bool = True,
) -> list[str]:
    cmd = [
        str(updater_exe),
        "--install-dir",
        str(install_dir),
        "--package",
        str(package_path),
        "--expected-sha256",
        expected_sha256,
        "--app-exe",
        app_exe,
    ]
    if pid:
        cmd.extend(["--pid", str(pid)])
    if relaunch:
        cmd.append("--relaunch")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZenAI external updater")
    parser.add_argument("--install-dir", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--app-exe", default="ZenAI.exe")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--relaunch", action="store_true")
    parser.add_argument("--log", default="")
    args = parser.parse_args(argv)
    log_path = Path(args.log) if args.log else updates_dir() / "update.log"
    try:
        apply_update_package(
            install_dir=Path(args.install_dir),
            package_path=Path(args.package),
            expected_sha256=args.expected_sha256,
            app_exe=args.app_exe,
            pid=args.pid or None,
            relaunch=args.relaunch,
            log_path=log_path,
        )
    except UpdateError as exc:
        _write_update_log(log_path, f"fatal: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
