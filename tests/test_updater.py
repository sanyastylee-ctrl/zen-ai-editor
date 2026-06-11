from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from core import update


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UpdaterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def manifest(self, **overrides):
        data = {
            "app": "ZenAI",
            "channel": "dev",
            "latest_version": "0.1.1",
            "published_at": "2026-06-10T20:00:00Z",
            "min_supported_version": "0.1.0",
            "update_url": "https://example.com/ZenAI-v0.1.1-win64-update.zip",
            "sha256": "a" * 64,
            "size": 123,
            "release_notes": ["Исправлен Кодер"],
            "requires_full_update": True,
            "critical": False,
        }
        data.update(overrides)
        return data

    def make_zip(self, name: str = "update.zip", files: dict[str, bytes | str] | None = None) -> Path:
        files = files or {"ZenAI.exe": "new exe", "_internal/runtime.txt": "runtime"}
        path = self.root / name
        with zipfile.ZipFile(path, "w") as zf:
            for filename, content in files.items():
                payload = content.encode("utf-8") if isinstance(content, str) else content
                zf.writestr(filename, payload)
        return path

    def make_install(self) -> Path:
        install = self.root / "ZenAI"
        install.mkdir()
        (install / "ZenAI.exe").write_text("old exe", encoding="utf-8")
        (install / "_internal").mkdir()
        (install / "_internal" / "runtime.txt").write_text("old runtime", encoding="utf-8")
        (install / "models").mkdir()
        (install / "models" / "local.gguf").write_text("model bytes", encoding="utf-8")
        return install

    def test_valid_manifest_parsed(self):
        manifest = update.parse_manifest(self.manifest(), current_version="0.1.0", channel="dev")
        self.assertEqual(manifest.app, "ZenAI")
        self.assertEqual(manifest.latest_version, "0.1.1")
        self.assertEqual(manifest.release_notes, ["Исправлен Кодер"])

    def test_invalid_app_rejected(self):
        with self.assertRaises(update.UpdateError):
            update.parse_manifest(self.manifest(app="Other"), current_version="0.1.0", channel="dev")

    def test_same_or_older_version_ignored(self):
        with self.assertRaises(update.UpdateError):
            update.parse_manifest(self.manifest(latest_version="0.1.0"), current_version="0.1.0", channel="dev")
        with self.assertRaises(update.UpdateError):
            update.parse_manifest(self.manifest(latest_version="0.0.9"), current_version="0.1.0", channel="dev")

    def test_newer_version_detected(self):
        self.assertLess(update.compare_versions("0.1.0", "0.1.1"), 0)

    def test_invalid_sha_rejected(self):
        with self.assertRaises(update.UpdateError):
            update.parse_manifest(self.manifest(sha256="bad"), current_version="0.1.0", channel="dev")

    def test_unsafe_update_url_rejected(self):
        with self.assertRaises(update.UpdateError):
            update.parse_manifest(self.manifest(update_url="file:///tmp/update.zip"), current_version="0.1.0", channel="dev")

    def test_sha256_success_and_mismatch(self):
        path = self.root / "package.zip"
        path.write_text("payload", encoding="utf-8")
        digest = _sha(path)
        self.assertTrue(update.verify_sha256(path, digest))
        self.assertFalse(update.verify_sha256(path, "0" * 64))

    def test_prepare_hash_mismatch_deletes_package(self):
        def fake_download(_url: str, dest: Path, **_kwargs) -> Path:
            dest.write_text("bad", encoding="utf-8")
            return dest

        with mock.patch("core.update.download_update", side_effect=fake_download):
            manifest = update.UpdateManifest(
                app="ZenAI",
                channel="dev",
                latest_version="0.1.1",
                published_at="",
                min_supported_version="0.1.0",
                update_url="https://example.com/update.zip",
                sha256="0" * 64,
                size=3,
            )
            with self.assertRaises(update.UpdateError):
                update.prepare_update_package(manifest, cache_dir=self.root)
        package = self.root / "ZenAI-v0.1.1-update.zip"
        self.assertFalse(package.exists())

    def test_zip_path_traversal_rejected(self):
        package = self.make_zip(files={"../evil.txt": "bad", "ZenAI.exe": "x"})
        with self.assertRaises(update.UpdateError):
            update.inspect_update_zip(package)

    def test_zip_absolute_path_rejected(self):
        package = self.make_zip(files={"/evil.txt": "bad", "ZenAI.exe": "x"})
        with self.assertRaises(update.UpdateError):
            update.inspect_update_zip(package)

    def test_zip_drive_letter_path_rejected(self):
        package = self.make_zip(files={"C:/evil.txt": "bad", "ZenAI.exe": "x"})
        with self.assertRaises(update.UpdateError):
            update.inspect_update_zip(package)

    def test_zip_models_gguf_rejected(self):
        package = self.make_zip(files={"ZenAI.exe": "x", "models/model.gguf": "model"})
        with self.assertRaises(update.UpdateError):
            update.inspect_update_zip(package)

    def test_valid_zip_accepted(self):
        package = self.make_zip()
        names = update.inspect_update_zip(package)
        self.assertIn("ZenAI.exe", names)

    def test_backup_excludes_models_and_extraction_preserves_models(self):
        install = self.make_install()
        package = self.make_zip(files={"ZenAI.exe": "new exe", "_internal/runtime.txt": "new runtime"})
        result = update.apply_update_package(
            install_dir=install,
            package_path=package,
            expected_sha256=_sha(package),
            relaunch=False,
        )
        self.assertTrue(result.ok)
        self.assertEqual((install / "ZenAI.exe").read_text(encoding="utf-8"), "new exe")
        self.assertEqual((install / "models" / "local.gguf").read_text(encoding="utf-8"), "model bytes")
        self.assertIsNotNone(result.backup_dir)
        assert result.backup_dir is not None
        self.assertFalse((result.backup_dir / "models").exists())

    def test_rollback_restores_previous_files(self):
        install = self.make_install()
        package = self.make_zip(files={"ZenAI.exe": "new exe", "_internal/runtime.txt": "new runtime"})
        original_copy = update._copy_payload_to_install

        def fail_new_payload_once(payload_dir: Path, target_dir: Path) -> None:
            if payload_dir.name.endswith("_update_staging") or payload_dir.parent.name.endswith("_update_staging"):
                raise OSError("copy failed")
            original_copy(payload_dir, target_dir)

        with mock.patch("core.update._copy_payload_to_install", side_effect=fail_new_payload_once):
            with self.assertRaises(update.UpdateError):
                update.apply_update_package(
                    install_dir=install,
                    package_path=package,
                    expected_sha256=_sha(package),
                    relaunch=False,
                )
        self.assertEqual((install / "ZenAI.exe").read_text(encoding="utf-8"), "old exe")
        self.assertEqual((install / "models" / "local.gguf").read_text(encoding="utf-8"), "model bytes")

    def test_bad_hash_after_success_does_not_reuse_old_backup_for_rollback(self):
        install = self.make_install()
        package = self.make_zip(files={"ZenAI.exe": "new exe", "_internal/runtime.txt": "new runtime"})
        update.apply_update_package(
            install_dir=install,
            package_path=package,
            expected_sha256=_sha(package),
            relaunch=False,
        )
        self.assertEqual((install / "ZenAI.exe").read_text(encoding="utf-8"), "new exe")
        with self.assertRaises(update.UpdateError):
            update.apply_update_package(
                install_dir=install,
                package_path=package,
                expected_sha256="0" * 64,
                relaunch=False,
            )
        self.assertEqual((install / "ZenAI.exe").read_text(encoding="utf-8"), "new exe")

    def test_appdata_paths_untouched(self):
        install = self.make_install()
        appdata = self.root / "APPDATA" / "ZenAI"
        chats = appdata / "chats"
        runs = appdata / "agent_runs"
        chats.mkdir(parents=True)
        runs.mkdir()
        (chats / "chat.json").write_text("chat", encoding="utf-8")
        (runs / "run.json").write_text("run", encoding="utf-8")
        package = self.make_zip()
        update.apply_update_package(
            install_dir=install,
            package_path=package,
            expected_sha256=_sha(package),
            relaunch=False,
        )
        self.assertEqual((chats / "chat.json").read_text(encoding="utf-8"), "chat")
        self.assertEqual((runs / "run.json").read_text(encoding="utf-8"), "run")

    def test_updater_command_args_built_with_paths(self):
        cmd = update.build_updater_command(
            updater_exe=Path(r"D:\ZenAI\ZenAIUpdater.exe"),
            install_dir=Path(r"D:\ZenAI"),
            package_path=Path(r"C:\Users\User\AppData\Roaming\ZenAI\updates\update.zip"),
            expected_sha256="a" * 64,
            pid=123,
            relaunch=True,
        )
        self.assertEqual(cmd[0], r"D:\ZenAI\ZenAIUpdater.exe")
        self.assertIn("--install-dir", cmd)
        self.assertIn("--package", cmd)
        self.assertIn("--relaunch", cmd)

    def test_update_check_cleanly_handles_network_failure(self):
        with mock.patch("core.update.fetch_manifest", side_effect=update.UpdateError("Не удалось проверить обновления.")):
            with self.assertRaises(update.UpdateError) as ctx:
                update.check_for_update("https://example.com/manifest.json")
        self.assertNotIn("Traceback", str(ctx.exception))

    def test_update_available_result_includes_version_and_notes(self):
        with mock.patch("core.update.fetch_manifest", return_value=__import__("json").dumps(self.manifest())):
            result = update.check_for_update("https://example.com/manifest.json", current_version="0.1.0", channel="dev")
        self.assertTrue(result.available)
        self.assertEqual(result.manifest.latest_version if result.manifest else "", "0.1.1")
        self.assertEqual(result.manifest.release_notes if result.manifest else [], ["Исправлен Кодер"])


if __name__ == "__main__":
    unittest.main()
