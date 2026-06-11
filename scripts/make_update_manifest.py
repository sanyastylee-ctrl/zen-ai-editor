from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a ZenAI update manifest.")
    parser.add_argument("--zip", required=True, help="Update zip path")
    parser.add_argument("--url", required=True, help="HTTPS update URL")
    parser.add_argument("--version", required=True, help="Latest version")
    parser.add_argument("--channel", default="stable")
    parser.add_argument("--min-supported", default="0.1.0")
    parser.add_argument("--out", default="update-manifest.json")
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()

    zip_path = Path(args.zip)
    manifest = {
        "app": "ZenAI",
        "channel": args.channel,
        "latest_version": args.version,
        "published_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "min_supported_version": args.min_supported,
        "update_url": args.url,
        "sha256": sha256_file(zip_path),
        "size": zip_path.stat().st_size,
        "release_notes": args.note or ["Release notes pending."],
        "requires_full_update": True,
        "critical": False,
    }
    Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

