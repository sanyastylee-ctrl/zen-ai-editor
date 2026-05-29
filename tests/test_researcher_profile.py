from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from core import chat_store
import core.profiles as profiles_module
from core.chat_store import ChatSessionStore
from core.profiles import AIProfile, ChatTemplate, ProfileKind, ProfileManager
from core.research import (
    FakeWebBackend,
    ResearchPipeline,
    ResearchSource,
    compress_page_text,
    needs_web_search,
)
from ui.profile_switcher import ProfileSwitcher


class ResearcherProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_profile_kind_researcher_is_created_and_saved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(profiles_module, "CONFIG_DIR", root), \
                 mock.patch.object(profiles_module, "PROFILES_FILE", root / "profiles.json"), \
                 mock.patch.object(profiles_module, "LEGACY_PROFILES_FILE", root / "legacy.json"):
                pm = ProfileManager()
                pm.load()

                researcher = pm.get_active(ProfileKind.RESEARCHER)
                self.assertIsNotNone(researcher)
                self.assertEqual(researcher.name, "Поисковик")
                self.assertTrue(researcher.search_enabled)

                restored = ProfileManager()
                restored.load()
                self.assertIsNotNone(restored.get_active(ProfileKind.RESEARCHER))

    def test_profile_switcher_shows_coder_lera_researcher_not_vision(self):
        switcher = ProfileSwitcher()
        coder = AIProfile(id="coder", name="Кодер", kind=ProfileKind.CODER)
        lera = AIProfile(id="lera", name="Лера", kind=ProfileKind.COMPANION)
        researcher = AIProfile(id="researcher", name="Поисковик", kind=ProfileKind.RESEARCHER)
        vision = AIProfile(id="vision", name="Глаза", kind=ProfileKind.VISION)
        generic_vision = AIProfile(id="old-eyes", name="Глаза", kind=ProfileKind.GENERIC)

        switcher.set_profiles([coder, lera, researcher, vision, generic_vision], "researcher")

        self.assertEqual(set(switcher._buttons), {"coder", "lera", "researcher"})
        self.assertEqual(switcher.active_id(), "researcher")

    def test_fresh_query_triggers_web_search(self):
        backend = FakeWebBackend([
            ResearchSource(
                title="Python Downloads",
                url="https://www.python.org/downloads/",
                snippet="Latest Python release",
                facts=["Python 3.14 is current"],
            )
        ])

        result = ResearchPipeline(backend).run("Какая сейчас актуальная версия Python?")

        self.assertTrue(result.used_search)
        self.assertEqual(len(backend.search_calls), 1)
        self.assertIn("Источники:", result.answer)
        self.assertIn("https://www.python.org/downloads/", result.answer)

    def test_general_query_can_skip_web_search(self):
        backend = FakeWebBackend()
        result = ResearchPipeline(backend).run("Объясни простыми словами, что такое RAG.")

        self.assertFalse(result.used_search)
        self.assertEqual(backend.search_calls, [])

    def test_unavailable_backend_returns_clear_error_without_fake_citations(self):
        result = ResearchPipeline().run("Какая актуальная цена видеокарты?")

        self.assertFalse(result.used_search)
        self.assertEqual(result.sources, [])
        self.assertIn("not configured", result.answer)
        self.assertNotIn("http", result.answer)

    def test_source_objects_include_required_fields(self):
        source = ResearchSource(title="Doc", url="https://example.com", snippet="Snippet")

        self.assertEqual(source.title, "Doc")
        self.assertEqual(source.url, "https://example.com")
        self.assertEqual(source.snippet, "Snippet")
        self.assertTrue(source.retrieved_at)

    def test_search_result_text_is_compressed_before_prompt(self):
        text = "word " * 2000
        compressed = compress_page_text(text, max_chars=300)

        self.assertLessEqual(len(compressed), 315)
        self.assertIn("truncated", compressed)

    def test_max_pages_to_read_is_respected(self):
        sources = [
            ResearchSource(title=f"Source {i}", url=f"https://example.com/{i}", snippet="s")
            for i in range(5)
        ]
        backend = FakeWebBackend(sources, {source.url: "fact text" for source in sources})

        result = ResearchPipeline(backend).run(
            "Сравни актуальные версии Python",
            max_search_results=5,
            max_pages_to_read=2,
        )

        self.assertTrue(result.used_search)
        self.assertEqual(len(backend.read_calls), 2)
        self.assertEqual(len(result.sources), 2)

    def test_researcher_history_is_separate_from_coder_and_companion(self):
        with tempfile.TemporaryDirectory() as td:
            old_chats = chat_store.CHATS_DIR
            old_state = chat_store._STATE_FILE
            try:
                chat_store.CHATS_DIR = Path(td) / "chats"
                chat_store._STATE_FILE = Path(td) / "sessions" / "chat_state.json"
                chat_store.CHATS_DIR.mkdir()
                chat_store._STATE_FILE.parent.mkdir()
                store = ChatSessionStore()
                store.save_profile("coder", "Кодер", [{"role": "user", "text": "код"}], [("код", "ответ")])
                store.save_profile("researcher", "Поисковик", [{"role": "user", "text": "поиск"}], [("поиск", "источник")])

                _, coder_history = ChatSessionStore().load_profile("coder")
                _, researcher_history = ChatSessionStore().load_profile("researcher")

                self.assertEqual(coder_history, [("код", "ответ")])
                self.assertEqual(researcher_history, [("поиск", "источник")])
            finally:
                chat_store.CHATS_DIR = old_chats
                chat_store._STATE_FILE = old_state

    def test_query_classifier_marks_fresh_but_not_general(self):
        self.assertTrue(needs_web_search("Какая сейчас актуальная версия Python?"))
        self.assertFalse(needs_web_search("Объясни простыми словами RAG"))


if __name__ == "__main__":
    unittest.main()
