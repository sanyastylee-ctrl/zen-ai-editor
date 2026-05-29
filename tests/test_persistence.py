from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import app_data
from core import chat_store
from core import paths
from core.session import SessionManager
import core.session as session_module


class AppDataTests(unittest.TestCase):
    def test_read_json_handles_corrupt_empty_and_missing_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            missing = root / "missing.json"
            corrupt = root / "corrupt.json"
            empty = root / "empty.json"
            corrupt.write_text("{broken", encoding="utf-8")
            empty.write_text("", encoding="utf-8")

            self.assertEqual(app_data.read_json(missing, {"ok": False}), {"ok": False})
            self.assertEqual(app_data.read_json(corrupt, []), [])
            self.assertEqual(app_data.read_json(empty, "fallback"), "fallback")

    def test_atomic_write_json_and_permission_failure_are_safe(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "data.json"
            self.assertTrue(app_data.atomic_write_json(target, {"value": 1}))
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"value": 1})

            with mock.patch("pathlib.Path.mkdir", side_effect=OSError("denied")):
                self.assertFalse(app_data.atomic_write_json(Path(td) / "denied" / "x.json", {"x": 1}))

    def test_migration_never_overwrites_existing_new_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            legacy = root / "legacy.json"
            target = root / "target.json"
            legacy.write_text('{"source":"legacy"}', encoding="utf-8")
            target.write_text('{"source":"new"}', encoding="utf-8")

            app_data.migrate_legacy_file(legacy, target)

            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"source": "new"})

    def test_project_memory_dir_is_outside_project(self):
        with tempfile.TemporaryDirectory() as td:
            original = app_data.MEMORY_DIR
            try:
                app_data.MEMORY_DIR = Path(td) / "memory"
                project = Path(td) / "project"
                result = app_data.project_memory_dir(str(project))
                self.assertTrue(str(result).startswith(str(app_data.MEMORY_DIR)))
                self.assertNotIn(".zen_ai", str(result))
            finally:
                app_data.MEMORY_DIR = original


class ChatSessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.chats = root / "chats"
        self.sessions = root / "sessions"
        self.chats.mkdir()
        self.sessions.mkdir()
        self._old_chats = chat_store.CHATS_DIR
        self._old_state = chat_store._STATE_FILE
        chat_store.CHATS_DIR = self.chats
        chat_store._STATE_FILE = self.sessions / "chat_state.json"

    def tearDown(self):
        chat_store.CHATS_DIR = self._old_chats
        chat_store._STATE_FILE = self._old_state
        self.tmp.cleanup()

    def test_corrupt_state_and_session_json_do_not_crash(self):
        chat_store._STATE_FILE.write_text("{broken", encoding="utf-8")
        store = chat_store.ChatSessionStore()
        store.set_last_profile_id("profile-a")
        store.save_profile("profile-a", "Coder", [{"role": "user", "text": "hi"}], [])

        session_id = store._state["sessions_by_profile"]["profile-a"]
        (self.chats / f"{session_id}.json").write_text("", encoding="utf-8")

        records, history = chat_store.ChatSessionStore().load_profile("profile-a")

        self.assertEqual(records, [])
        self.assertEqual(history, [])

    def test_save_load_preserves_roles_order_and_history(self):
        store = chat_store.ChatSessionStore()
        records = [
            {"role": "user", "sender": "Ты", "text": "one"},
            {"role": "assistant", "sender": "Ассистент", "text": "two", "profile_kind": "coder"},
        ]
        store.save_profile("coder", "Кодер", records, [("one", "two")])

        loaded_records, history = chat_store.ChatSessionStore().load_profile("coder")

        self.assertEqual([r["role"] for r in loaded_records], ["user", "assistant"])
        self.assertEqual([r["text"] for r in loaded_records], ["one", "two"])
        self.assertEqual(history, [("one", "two")])

    def test_fast_repeated_saves_leave_valid_latest_json(self):
        store = chat_store.ChatSessionStore()
        for i in range(25):
            store.save_profile("coder", "Кодер", [{"role": "assistant", "text": f"answer-{i}"}], [])

        path = next(path for path in self.chats.glob("*.json") if path.name != "index.json")
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["messages"][-1]["content"], "answer-24")

    def test_tool_error_card_is_persisted(self):
        store = chat_store.ChatSessionStore()
        store.save_profile(
            "coder",
            "Кодер",
            [{"role": "tool", "sender": "Tool", "tool_name": "Tool: read_file", "output": "[error]", "ok": False}],
            [],
        )

        path = next(path for path in self.chats.glob("*.json") if path.name != "index.json")
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["messages"][0]["role"], "tool")
        self.assertFalse(data["messages"][0]["ok"])
        self.assertEqual(data["messages"][0]["output"], "[error]")

    def test_write_failure_does_not_raise(self):
        store = chat_store.ChatSessionStore()
        with mock.patch("core.chat_store.atomic_write_json", return_value=False):
            store.save_profile("coder", "Кодер", [{"role": "user", "text": "x"}], [])
            store.set_last_profile_id("coder")

    def test_create_list_save_load_session_by_profile_kind(self):
        store = chat_store.ChatSessionStore()
        session = store.create_session("coder", "Coder test", "C:/tmp/project", profile_id="coder-profile")
        store.save_message(session.id, {"role": "user", "text": "hello"})
        store.save_message(session.id, {"role": "assistant", "text": "world"})

        sessions = chat_store.ChatSessionStore().list_sessions("coder")
        records, history = chat_store.ChatSessionStore().load_session(session.id)

        self.assertEqual([s.title for s in sessions], ["Coder test"])
        self.assertEqual([r["text"] for r in records], ["hello", "world"])
        self.assertEqual(history, [("hello", "world")])

    def test_last_active_session_is_tracked_per_profile_kind(self):
        store = chat_store.ChatSessionStore()
        coder = store.create_session("coder", "Coder chat", profile_id="coder-profile")
        lera = store.create_session("companion", "Lera chat", profile_id="lera-profile")

        fresh = chat_store.ChatSessionStore()

        self.assertEqual(fresh.get_last_active_session("coder", profile_id="coder-profile"), coder.id)
        self.assertEqual(fresh.get_last_active_session("companion", profile_id="lera-profile"), lera.id)

    def test_delete_and_clear_session(self):
        store = chat_store.ChatSessionStore()
        session = store.create_session("researcher", "Research")
        store.save_message(session.id, {"role": "user", "text": "query"})

        store.clear_session(session.id)
        records, _ = store.load_session(session.id)
        self.assertEqual(records, [])

        store.delete_session(session.id)
        self.assertEqual(store.list_sessions("researcher"), [])
        self.assertEqual(store.load_session(session.id), ([], []))

    def test_rename_session_updates_index_without_loading_messages(self):
        store = chat_store.ChatSessionStore()
        session = store.create_session("coder", "Old")
        store.save_message(session.id, {"role": "user", "text": "x" * 1000})

        store.rename_session(session.id, "New")
        sessions = chat_store.ChatSessionStore().list_sessions("coder")

        self.assertEqual(sessions[0].title, "New")

    def test_index_listing_does_not_read_session_files(self):
        store = chat_store.ChatSessionStore()
        store.create_session("coder", "Indexed only")

        original_read_json = chat_store.read_json

        def fail_on_session_file(path, default=None):
            if str(path).endswith(".json") and Path(path).name != "index.json" and Path(path) != chat_store._STATE_FILE:
                raise AssertionError("session file should not be read for list_sessions")
            return original_read_json(path, default)

        with mock.patch("core.chat_store.read_json", side_effect=fail_on_session_file):
            sessions = chat_store.ChatSessionStore().list_sessions("coder")

        self.assertEqual(len(sessions), 1)

    def test_load_session_limits_initial_render_to_last_messages(self):
        store = chat_store.ChatSessionStore()
        session = store.create_session("companion", "Long")
        for i in range(75):
            store.save_message(session.id, {"role": "user", "text": f"user-{i}"})

        records, _ = store.load_session(session.id, message_limit=50)

        self.assertEqual(len(records), 50)
        self.assertEqual(records[0]["text"], "user-25")


class SessionManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._old_recovery = session_module._RECOVERY_DIR
        self._old_session = session_module._SESSION_FILE
        session_module._RECOVERY_DIR = str(root / "recovery")
        session_module._SESSION_FILE = str(root / "editor_session.json")

    def tearDown(self):
        session_module._RECOVERY_DIR = self._old_recovery
        session_module._SESSION_FILE = self._old_session
        self.tmp.cleanup()

    def test_save_load_and_corrupt_session(self):
        class FakeTabs:
            _tabs = [{"file_path": "", "modified": True, "editor": object()}]

            def _get_text_for_editor(self, _editor):
                return "draft"

            def currentIndex(self):
                return 0

        SessionManager.save(FakeTabs())
        loaded = SessionManager.load()
        self.assertIsInstance(loaded, dict)
        self.assertEqual(loaded["current_tab"], 0)

        Path(session_module._SESSION_FILE).write_text("{broken", encoding="utf-8")
        self.assertIsNone(SessionManager.load())

    def test_write_failure_does_not_raise(self):
        class FakeTabs:
            _tabs = []

            def currentIndex(self):
                return 0

        with mock.patch("core.session.atomic_write_json", return_value=False):
            SessionManager.save(FakeTabs())


class PathsTests(unittest.TestCase):
    def test_models_dir_source_and_frozen_modes_ignore_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            app_root = Path(paths.__file__).resolve().parent.parent

            with mock.patch.dict(os.environ, {}, clear=True):
                old_cwd = os.getcwd()
                project = root / "opened-project"
                project.mkdir()
                os.chdir(project)
                try:
                    self.assertEqual(paths.get_models_dir(), app_root / "models")
                    self.assertEqual(paths.resolve_model_path("model.gguf"), str(app_root / "models" / "model.gguf"))
                finally:
                    os.chdir(old_cwd)

            exe = root / "Bundle" / "ZenAI.exe"
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(paths.sys, "frozen", True, create=True):
                    with mock.patch.object(paths.sys, "executable", str(exe)):
                        self.assertEqual(paths.get_models_dir(), exe.parent / "models")

    def test_list_models_does_not_create_project_local_models_dir(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "opened-project"
            project.mkdir()
            old_cwd = os.getcwd()
            os.chdir(project)
            try:
                paths.list_available_models()
            finally:
                os.chdir(old_cwd)

            self.assertFalse((project / "models").exists())

    def test_resource_path_uses_meipass_when_packaged(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(paths.sys, "frozen", True, create=True):
                with mock.patch.object(paths.sys, "_MEIPASS", td, create=True):
                    self.assertEqual(paths.resource_path("assets/icons/user.svg"), Path(td) / "assets/icons/user.svg")


if __name__ == "__main__":
    unittest.main()
