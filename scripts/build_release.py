from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ZenAI portable release artifacts.")
    parser.add_argument("--skip-main", action="store_true", help="Do not rebuild ZenAI.exe")
    parser.add_argument("--skip-updater", action="store_true", help="Do not rebuild ZenAIUpdater.exe")
    parser.add_argument("--zip", default="ZenAI-portable-win64.zip", help="Portable zip path")
    args = parser.parse_args()

    if not args.skip_updater:
        run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "ZenAIUpdater.spec"])
        updater_src = ROOT / "dist" / "ZenAIUpdater" / "ZenAIUpdater.exe"
        updater_dest = ROOT / "dist" / "ZenAI" / "ZenAIUpdater.exe"
        if (ROOT / "dist" / "ZenAI").exists() and updater_src.exists():
            shutil.copy2(updater_src, updater_dest)

    if not args.skip_main:
        run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "ZenAI.spec"])
        updater_src = ROOT / "dist" / "ZenAIUpdater" / "ZenAIUpdater.exe"
        updater_dest = ROOT / "dist" / "ZenAI" / "ZenAIUpdater.exe"
        if updater_src.exists():
            shutil.copy2(updater_src, updater_dest)

    zip_path = ROOT / args.zip
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", ROOT / "dist" / "ZenAI")
    print(zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

