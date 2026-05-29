from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox, QFrame, QPushButton, QLabel
from PyQt6.QtWidgets import QSizePolicy

from core import chat_store
from core.profiles import AIProfile, ProfileKind
from ui.main_window import ZenEditor


def profile(pid: str, kind: ProfileKind, name: str) -> AIProfile:
    return AIProfile(id=pid, name=name, kind=kind, model_file=f"{pid}.gguf")


class ChatSessionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._old_chats = chat_store.CHATS_DIR
        self._old_state = chat_store._STATE_FILE
        chat_store.CHATS_DIR = root / "chats"
        chat_store._STATE_FILE = root / "sessions" / "chat_state.json"
        chat_store.CHATS_DIR.mkdir()
        chat_store._STATE_FILE.parent.mkdir()

    def tearDown(self):
        chat_store.CHATS_DIR = self._old_chats
        chat_store._STATE_FILE = self._old_state
        self.tmp.cleanup()

    def test_coder_lera_researcher_histories_are_isolated(self):
        store = chat_store.ChatSessionStore()
        coder = store.create_session("coder", "Coder", profile_id="coder-profile")
        lera = store.create_session("companion", "Lera", profile_id="lera-profile")
        researcher = store.create_session("researcher", "Research", profile_id="research-profile")
        store.save_message(coder.id, {"role": "user", "text": "code"})
        store.save_message(lera.id, {"role": "user", "text": "hello"})
        store.save_message(researcher.id, {"role": "user", "text": "search"})

        fresh = chat_store.ChatSessionStore()

        self.assertEqual(fresh.load_session(coder.id)[0][0]["text"], "code")
        self.assertEqual(fresh.load_session(lera.id)[0][0]["text"], "hello")
        self.assertEqual(fresh.load_session(researcher.id)[0][0]["text"], "search")

    def test_profile_switch_loads_session_without_model_unload(self):
        coder = profile("coder-profile", ProfileKind.CODER, "Кодер")
        lera = profile("lera-profile", ProfileKind.COMPANION, "Лера")

        class FakePM:
            def get(self, pid):
                return {"coder-profile": coder, "lera-profile": lera}.get(pid)

            def set_active(self, kind, pid):
                self.active = (kind, pid)

        class FakeInput:
            def setPlaceholderText(self, text):
                self.placeholder = text

        class FakeStore:
            def __init__(self):
                self.last_profile = ""

            def set_last_profile_id(self, pid):
                self.last_profile = pid

        window = ZenEditor.__new__(ZenEditor)
        window.pm = FakePM()
        window._chat_store = FakeStore()
        window.chat_input = FakeInput()
        window.workspace_pinned = False
        window._load_active_chat_session = mock.Mock()
        window._hide_workspace = mock.Mock()
        window._update_token_bar = mock.Mock()
        window._quote_log = lambda value, limit=240: str(value)

        class FakeModelManager:
            def loaded(self):
                raise AssertionError("profile switch must not inspect loaded models")

        with mock.patch("ui.main_window.ModelManager.instance", return_value=FakeModelManager()):
            window._on_profile_changed("lera-profile")

        window._load_active_chat_session.assert_called_once_with("lera-profile")
        self.assertEqual(window.chat_input.placeholder, "Написать Лера…")

    def test_chat_controls_are_sidebar_not_input_bar(self):
        window = ZenEditor()
        try:
            self.assertTrue(hasattr(window, "chat_sidebar_list_layout"))
            self.assertTrue(hasattr(window, "chat_header_label"))
            self.assertFalse(hasattr(window, "chat_session_combo"))
            self.assertTrue(hasattr(window, "send_btn"))
            self.assertIsNot(window.new_chat_btn.parentWidget(), window.chat_input.parentWidget())
        finally:
            window.close()

    def test_chat_header_uses_profile_title_and_muted_session_subtitle(self):
        window = ZenEditor()
        try:
            coder = window.pm.get_active(ProfileKind.CODER)
            self.assertIsNotNone(coder)
            window.profile_switcher.set_active(coder.id)
            window._on_profile_changed(coder.id)

            self.assertEqual(window.chat_header_title_label.text(), coder.name)
            self.assertNotIn("/", window.chat_header_title_label.text())
            self.assertTrue(window.chat_header_subtitle_label.text())
            self.assertIn(window._session_title(window._current_session_id(coder.id)), window.chat_header_subtitle_label.text())
        finally:
            window.close()

    def test_input_bar_spacing_and_controls_are_clean(self):
        window = ZenEditor()
        try:
            self.assertTrue(hasattr(window, "chat_input_bar"))
            margins = window.chat_input_bar.layout().contentsMargins()
            self.assertGreaterEqual(margins.left(), 8)
            self.assertGreaterEqual(margins.right(), 8)
            self.assertGreaterEqual(window.chat_input_bar.minimumHeight(), 52)
            self.assertEqual(window.chat_input_bar.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
            self.assertEqual(window.chat_input.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)

            input_buttons = {
                child.objectName()
                for child in window.chat_input_bar.findChildren(QPushButton)
            }
            self.assertIn("composerIconButton", input_buttons)
            self.assertIn("composerSendButton", input_buttons)
            self.assertIn("stop_btn", input_buttons)
            self.assertNotIn("chatNewButton", input_buttons)
            self.assertIsNot(window.new_chat_btn.parentWidget(), window.chat_input_bar)
        finally:
            window.close()

    def test_splitters_are_subtle_and_workspace_header_exists(self):
        window = ZenEditor()
        try:
            self.assertLessEqual(window.main_splitter.handleWidth(), 4)
            self.assertLessEqual(window.work_splitter.handleWidth(), 4)
            headers = window.workspace_panel.findChildren(QFrame, "workspace_header")
            self.assertTrue(headers)
            self.assertTrue(hasattr(window, "workspace_pin_btn"))
            self.assertTrue(hasattr(window, "workspace_hide_btn"))
        finally:
            window.close()

    def test_sidebar_active_row_and_message_width_limits_exist(self):
        window = ZenEditor()
        try:
            rows = window.findChildren(QFrame, "chatSessionRow")
            self.assertTrue(rows)
            self.assertTrue(any(row.property("active") == "true" for row in rows))
            self.assertLessEqual(window.chat_view._column.maximumWidth(), 840)
        finally:
            window.close()

    def test_clicking_sidebar_session_switches_profile_and_session(self):
        window = ZenEditor()
        try:
            coder = window.pm.get_active(ProfileKind.CODER)
            lera = window.pm.get_active(ProfileKind.COMPANION)
            self.assertIsNotNone(coder)
            self.assertIsNotNone(lera)
            session = window._chat_store.create_session("companion", "Lera sidebar", profile_id=lera.id)

            window._open_chat_session(lera.id, session.id)

            self.assertEqual(window.profile_switcher.active_id(), lera.id)
            self.assertEqual(window._current_session_id(lera.id), session.id)
        finally:
            window.close()

    def test_delete_current_session_opens_fallback(self):
        window = ZenEditor()
        try:
            profile = window.pm.get_active(ProfileKind.COMPANION)
            self.assertIsNotNone(profile)
            window.profile_switcher.set_active(profile.id)
            window._on_profile_changed(profile.id)
            window.new_chat()
            old_session = window._current_session_id(profile.id)

            with mock.patch("PyQt6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
                window.delete_current_chat()

            self.assertNotEqual(window._current_session_id(profile.id), old_session)
            self.assertTrue(window._current_session_id(profile.id))
        finally:
            window.close()

    def test_workspace_is_visible_for_coder_profile(self):
        window = ZenEditor()
        try:
            coder = window.pm.get_active(ProfileKind.CODER)
            self.assertIsNotNone(coder)
            window.profile_switcher.set_active(coder.id)
            window._on_profile_changed(coder.id)

            self.assertTrue(window.workspace_visible)
            self.assertFalse(window.workspace_panel.isHidden())
        finally:
            window.close()

    def test_searcher_workspace_shows_sources_panel_empty_state(self):
        window = ZenEditor()
        try:
            researcher = window.pm.get_active(ProfileKind.RESEARCHER)
            self.assertIsNotNone(researcher)
            window.profile_switcher.set_active(researcher.id)
            window._on_profile_changed(researcher.id)

            self.assertTrue(window.workspace_visible)
            self.assertEqual(window.active_workspace_tab, "sources")
            self.assertEqual(window._editor_tabs.tabText(window._workspace_tab_index("sources")), "Sources")
            self.assertIn("Источники появятся", window.sources_empty_label.text())
        finally:
            window.close()

    def test_source_cards_render_title_url_excerpt_and_status(self):
        window = ZenEditor()
        try:
            window._render_research_sources([{
                "id": "S1",
                "title": "PyQt docs",
                "url": "https://example.com/doc",
                "domain": "example.com",
                "excerpt": "Useful excerpt",
                "read_ok": True,
                "status": "read",
                "relevance_score": 0.8,
            }])

            cards = window.sources_container.findChildren(QFrame, "sourceCard")
            self.assertEqual(len(cards), 1)
            text = " ".join(label.text() for label in cards[0].findChildren(QLabel))
            self.assertIn("PyQt docs", text)
            self.assertIn("example.com", text)
            self.assertIn("Useful excerpt", text)
            self.assertIn("read", text)
        finally:
            window.close()

    def test_sources_ready_updates_workspace_for_searcher(self):
        window = ZenEditor()
        try:
            researcher = window.pm.get_active(ProfileKind.RESEARCHER)
            self.assertIsNotNone(researcher)
            window.profile_switcher.set_active(researcher.id)
            window._on_profile_changed(researcher.id)

            window._on_research_sources_ready([{
                "title": "Doc",
                "url": "https://example.com",
                "domain": "example.com",
                "excerpt": "Excerpt",
                "read_ok": True,
                "status": "read",
                "relevance_score": 1.0,
            }])

            self.assertEqual(window.active_workspace_tab, "sources")
            self.assertIn("1 источников", window.workspace_reason_label.text())
            self.assertTrue(window.sources_container.findChildren(QFrame, "sourceCard"))
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
