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
    DuckDuckGoBackend,
    FetchedPage,
    FakeWebBackend,
    PrivacyFirewall,
    ResearchPipeline,
    ResearchCapability,
    ResearchSource,
    SearchResult,
    UnavailableWebBackend,
    compress_page_text,
    extract_html_text,
    is_fetch_url_allowed,
    needs_web_search,
)
from ai.research import ResearchWorker
from ui.profile_switcher import ProfileSwitcher
from ui.main_window import ZenEditor


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
        ], {"https://www.python.org/downloads/": "Python releases and downloads page."})

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
        result = ResearchPipeline(UnavailableWebBackend()).run("Какая актуальная цена видеокарты?")

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

    def test_pipeline_fetches_top_pages_and_returns_citations(self):
        sources = [
            ResearchSource("One", "https://example.com/1", snippet="first"),
            ResearchSource("Two", "https://example.com/2", snippet="second"),
        ]
        backend = FakeWebBackend(sources, {
            "https://example.com/1": "Important public fact about PyQt splitter handles.",
            "https://example.com/2": "Second public fact about stylesheet handles.",
        })

        result = ResearchPipeline(backend).run("Найди документацию PyQt6 QSplitter handle", max_pages_to_read=2)

        self.assertTrue(result.used_search)
        self.assertEqual(len(result.sources), 2)
        self.assertEqual(result.sanitized_query, "Найди документацию PyQt6 QSplitter handle")
        self.assertEqual(result.backend_name, "fake")
        self.assertEqual(len(result.search_results), 2)
        self.assertEqual(len(result.fetched_pages), 2)
        self.assertEqual(len(result.ranked_sources), 2)
        self.assertTrue(result.extracted_facts)
        self.assertTrue(result.final_answer)
        self.assertTrue(result.citations)
        self.assertIn("Источники:", result.answer)
        self.assertIn("https://example.com/1", result.answer)

    def test_failed_fetch_is_not_cited_as_read_source(self):
        backend = FakeWebBackend([ResearchSource("Broken", "https://example.com/broken", snippet="snippet")], {})

        result = ResearchPipeline(backend).run("Найди актуальную документацию PyQt6", max_pages_to_read=1)

        self.assertEqual(result.error, "no readable sources")
        self.assertEqual(result.sources, [])
        self.assertEqual(len(result.ranked_sources), 1)
        self.assertFalse(result.ranked_sources[0].read_ok)
        self.assertEqual(result.ranked_sources[0].read_status, "failed")
        self.assertEqual(result.citations, [])
        self.assertNotIn("Источники:", result.answer)

    def test_non_http_url_is_skipped_before_fetch(self):
        backend = FakeWebBackend([ResearchSource("Local", "file:///secret.txt", snippet="private")], {
            "file:///secret.txt": "secret",
        })

        result = ResearchPipeline(backend).run("Найди актуальную документацию Python")

        self.assertEqual(backend.fetch_calls, [])
        self.assertEqual(result.sources, [])
        self.assertEqual(result.ranked_sources[0].failure_reason, "non_http_url")

    def test_citations_reference_existing_read_source_ids_only(self):
        backend = FakeWebBackend([
            ResearchSource("Read", "https://example.com/read", snippet="ok"),
            ResearchSource("Failed", "https://example.com/failed", snippet="bad"),
        ], {"https://example.com/read": "Readable public source about Python version."})

        result = ResearchPipeline(backend).run("Найди актуальную версию Python", max_pages_to_read=2)

        ranked_ids = {source.id for source in result.ranked_sources if source.read_ok}
        citation_ids = {citation.id for citation in result.citations}
        self.assertTrue(citation_ids)
        self.assertLessEqual(citation_ids, ranked_ids)
        self.assertTrue(all(source.used_in_answer for source in result.sources))

    def test_synthesis_answer_uses_sanitized_query_not_raw_private_path(self):
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")], {
            "https://example.com": "ZeroDivisionError happens when dividing by zero in Python."
        })
        query = r"Найди ошибку D:\Zen Ai Editor\core\app_data.py ZeroDivisionError division by zero"

        result = ResearchPipeline(backend).run(query)

        self.assertEqual(result.original_user_query, query)
        self.assertEqual(result.sanitized_query, "Python ZeroDivisionError division by zero")
        self.assertNotIn("Zen Ai Editor", result.answer)
        self.assertIn("Источники:", result.answer)

    def test_html_extraction_strips_scripts_and_styles(self):
        title, text = extract_html_text(
            "<html><head><title>Doc</title><style>.x{}</style><script>secret()</script></head>"
            "<body><nav>menu</nav><h1>Hello</h1><p>Useful text.</p></body></html>"
        )

        self.assertEqual(title, "Doc")
        self.assertIn("Useful text", text)
        self.assertNotIn("secret", text)
        self.assertNotIn(".x", text)

    def test_fetch_url_blocks_private_and_binary_schemes(self):
        allowed, _ = is_fetch_url_allowed("https://example.com")
        local_allowed, local_reason = is_fetch_url_allowed("http://127.0.0.1:8000")
        file_allowed, file_reason = is_fetch_url_allowed("file:///tmp/a.txt")

        self.assertTrue(allowed)
        self.assertFalse(local_allowed)
        self.assertEqual(local_reason, "private_or_local_url")
        self.assertFalse(file_allowed)
        self.assertEqual(file_reason, "non_http_url")

    def test_duckduckgo_html_parser_extracts_results(self):
        html = '''
        <div class="result">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc">Example <b>Doc</b></a>
          <a class="result__snippet">Useful snippet</a>
        </div>
        '''

        from core.research import _parse_duckduckgo_html

        results = _parse_duckduckgo_html(html, max_results=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/doc")
        self.assertEqual(results[0].source, "duckduckgo")

    def test_real_backend_object_is_available_without_paid_api(self):
        backend = DuckDuckGoBackend()

        self.assertEqual(backend.name, "duckduckgo")
        self.assertTrue(backend.available())

    def test_research_capability_stage1_is_researcher_only(self):
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")], {"https://example.com": "fact"})
        capability = ResearchCapability(backend)

        disabled = capability.search_for_profile("coder", "Найди актуальную версию Python")
        allowed = capability.search_for_profile("researcher", "Найди актуальную версию Python")

        self.assertEqual(disabled.error, "research capability disabled for profile")
        self.assertTrue(allowed.used_search)

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
        self.assertTrue(needs_web_search("Найди документацию PyQt6 QSplitter"))
        self.assertFalse(needs_web_search("Объясни простыми словами RAG"))

    def test_project_code_is_blocked_from_outbound_search(self):
        backend = FakeWebBackend()
        query = """Найди актуальную причину ошибки:
```python
def private_business_logic(secret):
    return secret + 1
```"""

        result = ResearchPipeline(backend).run(query)

        self.assertEqual(backend.search_calls, [])
        self.assertEqual(result.error, "privacy blocked")
        self.assertIn("code", result.privacy_reasons)

    def test_local_windows_paths_are_stripped_before_search(self):
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")])
        query = r"Какая актуальная ошибка PermissionError в C:\Users\rebko\Private\main.py?"

        result = ResearchPipeline(backend).run(query)

        self.assertTrue(result.used_search)
        sent_query = backend.search_calls[0][0]
        self.assertNotIn(r"C:\Users", sent_query)
        self.assertIn("[local path]", sent_query)

    def test_api_key_like_strings_are_blocked(self):
        backend = FakeWebBackend()
        query = "Какая актуальная ошибка OpenAI API key sk-1234567890abcdefXYZ?"

        result = ResearchPipeline(backend).run(query)

        self.assertEqual(backend.search_calls, [])
        self.assertEqual(result.error, "privacy blocked")
        self.assertIn("secret", result.privacy_reasons)

    def test_chat_history_is_not_sent_as_query(self):
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")])
        query = "User: мой приватный чат\nAssistant: ответ\nКакая актуальная версия Python?"

        result = ResearchPipeline(backend).run(query, confirmed_outbound=True)

        sent_query = backend.search_calls[0][0]
        self.assertTrue(result.used_search)
        self.assertNotIn("User:", sent_query)
        self.assertNotIn("Assistant:", sent_query)
        self.assertNotIn("мой приватный чат", sent_query)

    def test_companion_memory_requires_confirmation_and_is_not_sent(self):
        backend = FakeWebBackend()
        query = "companion_memory: user email rebko@example.com\nНайди актуальные новости Python"

        result = ResearchPipeline(backend).run(query)

        self.assertEqual(backend.search_calls, [])
        self.assertEqual(result.error, "privacy confirmation required")
        self.assertIn("companion_memory", result.privacy_reasons)

    def test_traceback_is_sanitized_to_error_type_and_message(self):
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")])
        query = r"""Найди актуальное решение:
Traceback (most recent call last):
  File "C:\Users\rebko\secret_project\main.py", line 1, in <module>
    print(1 / 0)
ZeroDivisionError: division by zero
"""

        result = ResearchPipeline(backend).run(query)

        self.assertTrue(result.used_search)
        sent_query = backend.search_calls[0][0]
        self.assertEqual(sent_query, "Python ZeroDivisionError division by zero")
        self.assertNotIn("rebko", sent_query)
        self.assertNotIn("secret_project", sent_query)

    def test_safe_generic_query_is_allowed(self):
        backend = FakeWebBackend([ResearchSource("Python", "https://www.python.org/")])

        result = ResearchPipeline(backend).run("Какая сейчас актуальная версия Python?")

        self.assertTrue(result.used_search)
        self.assertEqual(backend.search_calls[0][0], "Какая сейчас актуальная версия Python?")

    def test_sensitive_query_requires_confirmation(self):
        backend = FakeWebBackend()

        result = ResearchPipeline(backend).run("Найди актуальные утечки для user@example.com")

        self.assertEqual(result.error, "privacy confirmation required")
        self.assertEqual(backend.search_calls, [])
        self.assertIn("personal_data", result.privacy_reasons)

    def test_confirmed_sensitive_payload_is_allowed_only_after_confirmation(self):
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")])

        blocked = ResearchPipeline(backend).run("Найди актуальные данные для user@example.com")
        allowed = ResearchPipeline(backend).run(
            "Найди актуальные данные для user@example.com",
            confirmed_outbound=True,
        )

        self.assertEqual(blocked.error, "privacy confirmation required")
        self.assertTrue(allowed.used_search)
        self.assertEqual(backend.search_calls[0][0], "Найди актуальные данные для [personal email]")

    def test_logs_show_sanitized_query_not_private_raw_content(self):
        backend = FakeWebBackend()
        query = r"Какая актуальная ошибка в C:\Users\rebko\app.py с token=supersecret123?"

        with mock.patch("core.research.write_log") as log:
            ResearchPipeline(backend).run(query)

        rendered = "\n".join(str(call.args[0]) for call in log.call_args_list)
        self.assertIn("[research_query_sanitized]", rendered)
        self.assertNotIn(r"C:\Users\rebko", rendered)
        self.assertNotIn("supersecret123", rendered)
        self.assertIn("[secret]", rendered)

    def test_researchworker_emits_confirmation_event_without_search(self):
        profile = AIProfile(id="researcher", name="Поисковик", kind=ProfileKind.RESEARCHER)
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")])
        worker = ResearchWorker(profile, "Найди актуальные данные для user@example.com", backend=backend)
        events: list[dict] = []
        worker.confirmation_required.connect(events.append)

        worker.run()

        self.assertTrue(worker.research_pending_confirmation)
        self.assertEqual(backend.search_calls, [])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["sanitized_query"], "Найди актуальные данные для [personal email]")

    def test_researchworker_confirm_sends_sanitized_query(self):
        profile = AIProfile(id="researcher", name="Поисковик", kind=ProfileKind.RESEARCHER)
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")])
        worker = ResearchWorker(
            profile,
            "Найди актуальные данные для user@example.com",
            backend=backend,
            confirmed_outbound=True,
        )
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)

        worker.run()

        self.assertEqual(backend.search_calls[0][0], "Найди актуальные данные для [personal email]")
        self.assertTrue(any("Поисковый запрос: Найди актуальные данные для [personal email]" in c for c in chunks))

    def test_ui_confirmation_dialog_stores_accept_decision(self):
        class FakeBox:
            class Icon:
                Warning = object()

            class ButtonRole:
                AcceptRole = object()
                RejectRole = object()

            clicked = None
            shown_text = ""

            def __init__(self, parent=None):
                self.send = object()
                self.cancel = object()
                FakeBox.clicked = self.send

            def setWindowTitle(self, title): pass
            def setIcon(self, icon): pass
            def setText(self, text): pass
            def setInformativeText(self, text):
                FakeBox.shown_text = text
            def setDetailedText(self, text): pass
            def addButton(self, text, role):
                return self.send if "Отправить" in text else self.cancel
            def setDefaultButton(self, button): pass
            def exec(self):
                return 0
            def clickedButton(self):
                return FakeBox.clicked

        window = ZenEditor.__new__(ZenEditor)
        window._quote_log = lambda value, limit=240: str(value)
        window._pending_research_confirmation = None
        with mock.patch("ui.main_window.QMessageBox", FakeBox):
            window._on_research_confirmation_required({
                "profile_id": "researcher",
                "raw_query": "raw private user@example.com",
                "sanitized_query": "Найди данные для [personal email]",
                "reasons": ["personal_data"],
            })

        self.assertTrue(window._pending_research_confirmation["accepted"])
        self.assertEqual(window._pending_research_confirmation["sanitized_query"], "Найди данные для [personal email]")
        self.assertIn("Найди данные для [personal email]", FakeBox.shown_text)
        self.assertNotIn("raw private", FakeBox.shown_text)

    def test_ui_confirmation_dialog_stores_cancel_decision(self):
        class FakeBox:
            class Icon:
                Warning = object()

            class ButtonRole:
                AcceptRole = object()
                RejectRole = object()

            def __init__(self, parent=None):
                self.send = object()
                self.cancel = object()

            def setWindowTitle(self, title): pass
            def setIcon(self, icon): pass
            def setText(self, text): pass
            def setInformativeText(self, text): pass
            def setDetailedText(self, text): pass
            def addButton(self, text, role):
                return self.send if "Отправить" in text else self.cancel
            def setDefaultButton(self, button): pass
            def exec(self):
                return 0
            def clickedButton(self):
                return self.cancel

        window = ZenEditor.__new__(ZenEditor)
        window._quote_log = lambda value, limit=240: str(value)
        window._pending_research_confirmation = None
        with mock.patch("ui.main_window.QMessageBox", FakeBox):
            window._on_research_confirmation_required({
                "profile_id": "researcher",
                "raw_query": "raw private user@example.com",
                "sanitized_query": "Найди данные для [personal email]",
                "reasons": ["personal_data"],
            })

        self.assertFalse(window._pending_research_confirmation["accepted"])


if __name__ == "__main__":
    unittest.main()
