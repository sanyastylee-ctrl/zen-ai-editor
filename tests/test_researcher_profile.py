from __future__ import annotations

import json
import os
import re
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
    ResearchBackendStatus,
    ResearchPipeline,
    ResearchCapability,
    ResearchSource,
    SearchResult,
    SearxNGBackend,
    UnavailableWebBackend,
    backend_unavailable_message,
    diagnose_research_backend,
    clean_final_answer_for_user,
    clean_user_visible_text,
    compress_page_text,
    detect_css_or_script_dump,
    detect_js_bootstrap_dump,
    detect_token_list_dump,
    extract_answer_facts,
    extract_key_fact_for_answer,
    format_research_final_answer,
    extract_html_text,
    is_human_readable_excerpt,
    is_fetch_url_allowed,
    fallback_public_search_query,
    needs_web_search,
    select_research_backend,
    render_citations,
)
from ai.research import ResearchWorker
from ui.profile_switcher import ProfileSwitcher
from ui.main_window import ZenEditor


USER_VISIBLE_RESEARCH_GARBAGE = [
    "[truncated]",
    "ascii codec",
    "HTTP Error",
    "binary_content",
    "empty_page",
    "js_only_suspected",
    "not_human_readable",
    "js_bootstrap",
    "window.initData",
    "window.gokuProps",
    "document.documentElement",
    "font-family",
    "Traceback",
    "Connection refused",
    "function(",
    "OptanonWrapper",
    "schema.org",
    "@context",
    "@graph",
    "Client Challenge",
    "A required part of this site",
    "webpack",
    "__NEXT_DATA__",
]


def assert_no_user_visible_research_garbage(testcase: unittest.TestCase, text: object) -> None:
    rendered = str(text or "")
    for token in USER_VISIBLE_RESEARCH_GARBAGE:
        testcase.assertNotIn(token, rendered, f"User-visible research text leaked {token!r}: {rendered[:300]}")


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
        self.assertIn("Python Downloads — python.org", result.answer)
        self.assertNotIn("https://www.python.org/downloads/", result.answer)

    def test_general_query_can_skip_web_search(self):
        backend = FakeWebBackend()
        result = ResearchPipeline(backend).run("Объясни простыми словами, что такое RAG.")

        self.assertFalse(result.used_search)
        self.assertEqual(backend.search_calls, [])

    def test_unavailable_backend_returns_clear_error_without_fake_citations(self):
        result = ResearchPipeline(UnavailableWebBackend()).run("Какая актуальная цена видеокарты?")

        self.assertFalse(result.used_search)
        self.assertEqual(result.sources, [])
        self.assertIn("Поисковый backend сейчас недоступен", result.answer)
        self.assertNotIn("http", result.answer)

    def test_auto_backend_uses_searxng_when_health_check_passes(self):
        with mock.patch.object(SearxNGBackend, "health_check", return_value=(True, "")):
            backend, status = select_research_backend("auto", "http://127.0.0.1:8080")

        self.assertIsInstance(backend, SearxNGBackend)
        self.assertTrue(status.searxng_available)
        self.assertEqual(status.effective_backend, "searxng")

    def test_explicit_searxng_without_url_returns_unavailable_cleanly(self):
        backend, status = select_research_backend("searxng", "")

        self.assertIsInstance(backend, UnavailableWebBackend)
        self.assertEqual(status.recommended_action, "configure_searxng_url")
        self.assertFalse(status.searxng_url_configured)

    def test_explicit_duckduckgo_uses_ddg(self):
        with mock.patch.dict(os.environ, {"ZENAI_RESEARCH_BACKEND": "unavailable"}, clear=False):
            backend, status = select_research_backend("duckduckgo", "")

        self.assertIsInstance(backend, DuckDuckGoBackend)
        self.assertEqual(status.effective_backend, "duckduckgo")

    def test_env_backend_overrides_default_auto_profile(self):
        with mock.patch.dict(os.environ, {"ZENAI_RESEARCH_BACKEND": "unavailable"}, clear=False):
            backend, status = select_research_backend("auto", "")

        self.assertIsInstance(backend, UnavailableWebBackend)
        self.assertEqual(status.selected_backend, "unavailable")

    def test_diagnose_research_backend_reports_clean_recommended_action(self):
        with mock.patch.object(SearxNGBackend, "health_check", return_value=(False, "searxng unavailable")):
            status = diagnose_research_backend("auto", "http://127.0.0.1:8080")

        self.assertIsInstance(status, ResearchBackendStatus)
        self.assertNotIn("traceback", status.recommended_action.lower())
        self.assertTrue(status.configured_backend)

    def test_searxng_valid_fake_json_produces_search_results(self):
        class FakeResponse:
            def __init__(self, payload: str):
                self._payload = payload.encode("utf-8")
                self.status = 200
                self.headers = {"content-type": "application/json; charset=utf-8"}

            def read(self, n: int = -1):
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        payload = {"results": [{"title": "PyQt6 Docs", "url": "https://example.com/pyqt", "content": "QSplitter docs"}]}
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(json.dumps(payload))):
            backend = SearxNGBackend("http://127.0.0.1:8080")
            results = backend.search("PyQt6 QSplitter", max_results=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "searxng")
        self.assertEqual(results[0].title, "PyQt6 Docs")

    def test_duckduckgo_challenge_html_is_detected(self):
        class FakeResponse:
            def __init__(self, payload: str):
                self._payload = payload.encode("utf-8")
                self.status = 200
                self.headers = {"content-type": "text/html; charset=utf-8"}

            def read(self, n: int = -1):
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        challenge_html = "<html><body><form id='challenge-form'>verify you are human</form></body></html>"
        with mock.patch("urllib.request.urlopen", side_effect=[FakeResponse(challenge_html), FakeResponse(challenge_html)]):
            backend = DuckDuckGoBackend()
            with self.assertRaises(RuntimeError) as ctx:
                backend.search("PyQt6 QSplitter", max_results=5)

        self.assertIn("duckduckgo_challenged", str(ctx.exception))

    def test_duckduckgo_no_results_html_is_reported_cleanly(self):
        class FakeResponse:
            def __init__(self, payload: str):
                self._payload = payload.encode("utf-8")
                self.status = 200
                self.headers = {"content-type": "text/html; charset=utf-8"}

            def read(self, n: int = -1):
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        no_results_html = "<html><body><div>No results found.</div></body></html>"
        with mock.patch("urllib.request.urlopen", side_effect=[FakeResponse(no_results_html), FakeResponse(no_results_html)]):
            backend = DuckDuckGoBackend()
            with self.assertRaises(RuntimeError) as ctx:
                backend.search("totally obscure query", max_results=5)

        self.assertIn("duckduckgo_no_results", str(ctx.exception))

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
        self.assertEqual(len([source for source in result.ranked_sources if source.read_ok]), 2)

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
        self.assertIn("One — example.com", result.answer)
        self.assertNotIn("https://example.com/1", result.answer)

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

    def test_tradingview_window_initdata_dump_is_not_read_ok(self):
        dump = (
            "window.initData = {\"symbols\":{\"BINANCE:BTCUSDT\":{\"price\":102000}},"
            "\"cookieDomainList\":[\"tradingview.com\"],\"features\":{\"chart\":true}}; "
            "document.documentElement.className = 'theme-dark'; function boot(){return window.initData;}"
        )
        self.assertTrue(detect_js_bootstrap_dump(dump))
        self.assertFalse(is_human_readable_excerpt(dump, "Bitcoin price"))
        backend = FakeWebBackend([
            ResearchSource("TradingView BTC", "https://example.com/tradingview", snippet="BTCUSDT chart on TradingView"),
        ], {"https://example.com/tradingview": dump})

        result = ResearchPipeline(backend).run("Найди актуальную цену Bitcoin", max_pages_to_read=1)

        self.assertEqual(len(result.ranked_sources), 1)
        self.assertFalse(result.ranked_sources[0].read_ok)
        self.assertEqual(result.ranked_sources[0].read_status, "js_bootstrap")
        self.assertFalse(result.ranked_sources[0].used_in_answer)

    def test_binance_gokuprops_css_dump_is_not_read_ok(self):
        dump = (
            "body { font-family: Arial; color: #111; background: #fff; } "
            "window.gokuProps = {\"page\":\"markets\",\"assets\":[\"BTC\",\"ETH\",\"BNB\"],"
            "\"cookieDomainList\":[\"binance.com\"]}; document.body.dataset.ready = true;"
        )
        self.assertTrue(detect_js_bootstrap_dump(dump))
        self.assertTrue(detect_css_or_script_dump(dump))
        backend = FakeWebBackend([
            ResearchSource("Binance Markets", "https://example.com/binance", snippet="Binance BTC market overview"),
        ], {"https://example.com/binance": dump})

        result = ResearchPipeline(backend).run("Найди актуальную цену BTC на Binance", max_pages_to_read=1)

        self.assertFalse(result.ranked_sources[0].read_ok)
        self.assertIn(result.ranked_sources[0].read_status, {"js_bootstrap", "not_human_readable"})
        self.assertNotIn("window.gokuProps", result.answer)
        self.assertNotIn("font-family", result.answer)

    def test_stackoverflow_optanon_jsonld_dump_is_not_read_ok(self):
        dump = (
            "Error python : [ZeroDivisionError: division by zero] - Stack Overflow "
            "function OptanonWrapper() { } { \"@context\": \"https://schema.org\", "
            "\"@graph\": [{\"@type\": \"WebSite\", \"name\": \"Stack Overflow\"}] }"
        )
        self.assertTrue(detect_js_bootstrap_dump(dump))
        self.assertFalse(is_human_readable_excerpt(dump, "Python ZeroDivisionError division by zero"))
        backend = FakeWebBackend([
            ResearchSource("Stack Overflow", "https://example.com/so", snippet="ZeroDivisionError division by zero"),
            ResearchSource("Readable", "https://example.com/readable", snippet="ZeroDivisionError fix"),
        ], {
            "https://example.com/so": dump,
            "https://example.com/readable": (
                "ZeroDivisionError occurs when Python code divides by zero. "
                "Fix it by validating the divisor before division or handling the exception."
            ),
        })

        result = ResearchPipeline(backend).run("Найди Python ZeroDivisionError division by zero", max_pages_to_read=1)

        self.assertEqual(result.ranked_sources[0].url, "https://example.com/readable")
        noisy = next(source for source in result.ranked_sources if source.url == "https://example.com/so")
        self.assertFalse(noisy.read_ok)
        self.assertEqual(noisy.read_status, "js_bootstrap")
        assert_no_user_visible_research_garbage(self, result.answer)
        self.assertNotIn("OptanonWrapper", result.answer)
        self.assertNotIn("@context", result.answer)

    def test_client_challenge_text_is_not_promoted_to_answer(self):
        challenge = "Client Challenge A required part of this site couldn’t load. Enable JavaScript and cookies."
        backend = FakeWebBackend([
            ResearchSource("Client Challenge", "https://example.com/challenge", snippet="llama-cpp-python package"),
            ResearchSource("Guide", "https://example.com/guide", snippet="llama-cpp-python CUDA Windows guide"),
        ], {
            "https://example.com/challenge": challenge,
            "https://example.com/guide": (
                "llama-cpp-python can be built on Windows with CUDA support by installing CUDA, "
                "Visual Studio Build Tools, and passing CMAKE_ARGS for the CUDA backend."
            ),
        })

        result = ResearchPipeline(backend).run(
            "Найди свежую информацию llama-cpp-python CUDA Windows PyInstaller",
            max_pages_to_read=1,
        )

        self.assertNotIn("A required part of this site", result.answer)
        self.assertNotIn("Client Challenge", result.answer)
        assert_no_user_visible_research_garbage(self, result.answer)
        noisy = next(source for source in result.ranked_sources if source.url == "https://example.com/challenge")
        self.assertFalse(noisy.read_ok)

    def test_coinmarketcap_like_price_sentence_is_read_ok(self):
        text = (
            "Bitcoin price today is updated on the market page. "
            "The page lists BTC market capitalization, trading volume, and recent price movement."
        )
        self.assertFalse(detect_token_list_dump(text))
        self.assertTrue(is_human_readable_excerpt(text, "Bitcoin price"))
        backend = FakeWebBackend([
            ResearchSource("CoinMarketCap Bitcoin", "https://example.com/cmc", snippet="Bitcoin price today"),
        ], {"https://example.com/cmc": text})

        result = ResearchPipeline(backend).run("Найди актуальную цену Bitcoin", max_pages_to_read=1)

        self.assertEqual(len(result.sources), 1)
        self.assertTrue(result.ranked_sources[0].read_ok)
        self.assertEqual(result.ranked_sources[0].read_status, "read")
        self.assertIn("Источники:", result.answer)

    def test_final_answer_excludes_js_config_css_tokens(self):
        noisy = (
            "window.initData = {\"cookieDomainList\":[\"site.test\"]}; "
            "document.documentElement.lang='en'; body { font-family: Arial; } [truncated]"
        )
        readable = (
            "PyQt6 QSplitter handles can be styled with the QSplitter::handle selector. "
            "The handle width can be adjusted in a Qt stylesheet."
        )
        backend = FakeWebBackend([
            ResearchSource("Noisy App", "https://example.com/noisy", snippet="QSplitter handle stylesheet reference"),
            ResearchSource("Readable Doc", "https://example.com/readable", snippet="PyQt6 QSplitter handle stylesheet"),
        ], {
            "https://example.com/noisy": noisy,
            "https://example.com/readable": readable,
        })

        result = ResearchPipeline(backend).run("Найди документацию PyQt6 QSplitter handle stylesheet", max_pages_to_read=1)

        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].url, "https://example.com/readable")
        self.assertNotIn("window.initData", result.answer)
        self.assertNotIn("document.documentElement", result.answer)
        self.assertNotIn("font-family", result.answer)
        self.assertNotIn("[truncated]", result.answer)
        self.assertIn("не удалось прочитать как человеческий текст", result.answer)

    def test_navigation_list_excerpt_is_not_promoted_to_answer_fact(self):
        nav_dump = (
            "...mboBox Widget PyQt - QDoubleSpinBox Widget PyQt - QToolBox Widget PyQt - "
            "QMenuBar, QMenu & QAction Widgets PyQt - QToolTip PyQt - QInputDialog Widget PyQt - "
            "QFileDialog Widget PyQt - QTab Widget PyQt - QSplitter Widget PyQt"
        )
        readable = (
            "PyQt6 QSplitter handles can be styled with the QSplitter::handle selector. "
            "The handle can use stylesheet width, margin, and background properties."
        )
        backend = FakeWebBackend([
            ResearchSource("Navigation Page", "https://example.com/nav", snippet="PyQt6 QSplitter Widget"),
            ResearchSource("Readable Doc", "https://example.com/readable", snippet="PyQt6 QSplitter handle stylesheet"),
        ], {
            "https://example.com/nav": nav_dump,
            "https://example.com/readable": readable,
        })

        result = ResearchPipeline(backend).run("Найди документацию PyQt6 QSplitter stylesheet handle", max_pages_to_read=2)

        self.assertIn("QSplitter::handle", result.answer)
        self.assertNotIn("QDoubleSpinBox", result.answer)
        self.assertNotIn("QInputDialog", result.answer)
        self.assertNotIn("...mboBox", result.answer)

    def test_qsplitter_stylesheet_query_gets_handle_selector_fact_when_sources_are_weak(self):
        weak = "QSplitter provides a handle that users can drag to adjust the size of panes."
        backend = FakeWebBackend([
            ResearchSource("PyQt6: Using QSplitter", "https://example.com/qsplitter", snippet="PyQt6 QSplitter handle"),
        ], {"https://example.com/qsplitter": weak})

        result = ResearchPipeline(backend).run("Найди документацию PyQt6 QSplitter stylesheet handle", max_pages_to_read=1)

        self.assertIn("QSplitter::handle", result.answer)
        assert_no_user_visible_research_garbage(self, result.answer)

    def test_zerodivision_query_gets_clean_stable_fallback_when_read_facts_are_weak(self):
        weak = (
            "Courses Tutorials Interview Prep Python Tutorial Data Types Interview Questions Examples Quizzes "
            "ZeroDivisionError: float division by zero in Python Last Updated."
        )
        backend = FakeWebBackend([
            ResearchSource("ZeroDivisionError", "https://example.com/zero", snippet="Python ZeroDivisionError division by zero"),
        ], {"https://example.com/zero": weak})

        result = ResearchPipeline(backend).run("Найди Python ZeroDivisionError division by zero", max_pages_to_read=1)

        self.assertIn("ZeroDivisionError", result.answer)
        self.assertIn("деление на ноль", result.answer)
        self.assertNotIn("Courses Tutorials", result.answer)
        assert_no_user_visible_research_garbage(self, result.answer)

    def test_noisy_sources_remain_visible_with_not_human_readable_status(self):
        source = ResearchSource(
            "Noisy",
            "https://example.com/noisy",
            snippet="Useful search snippet",
            read_ok=False,
            snippet_only=True,
            read_status="not_human_readable",
            failure_reason="not_human_readable",
            excerpt="body { font-family: Arial; } window.gokuProps = {};",
        )

        card = source.to_card()

        self.assertFalse(card["read_ok"])
        self.assertEqual(card["status"], "not_readable")
        self.assertIn("технический код", card["failure_reason"])
        self.assertIn("технический код", card["excerpt"])
        self.assertNotIn("window.gokuProps", card["excerpt"])
        self.assertIn("window.gokuProps", card["raw_excerpt"])
        for key in ("title", "snippet", "excerpt", "failure_reason", "display_message", "display_reason"):
            assert_no_user_visible_research_garbage(self, card.get(key, ""))

    def test_truncated_never_appears_in_answer_citations_or_source_card(self):
        backend = FakeWebBackend([
            ResearchSource("Readable", "https://example.com/readable", snippet="Bitcoin price today [truncated]"),
        ], {
            "https://example.com/readable": (
                "Bitcoin price today is $61,670.32 according to the public market page. "
                + ("Additional market context. " * 120)
            )
        })

        result = ResearchPipeline(backend).run("Найди актуальную цену Bitcoin", max_pages_to_read=1)
        rendered_cards = [source.to_card() for source in result.ranked_sources]

        self.assertNotIn("[truncated]", result.answer)
        self.assertTrue(result.citations)
        self.assertTrue(all("[truncated]" not in citation.reason for citation in result.citations))
        self.assertTrue(all("[truncated]" not in str(card.get("excerpt", "")) for card in rendered_cards))
        self.assertTrue(all("[truncated]" not in str(card.get("snippet", "")) for card in rendered_cards))
        assert_no_user_visible_research_garbage(self, result.answer)
        assert_no_user_visible_research_garbage(self, result.final_answer)
        assert_no_user_visible_research_garbage(self, render_citations(result.citations))

    def test_price_key_fact_prefers_actual_value_over_converter_sentence(self):
        text = (
            "Use our converter to compare Bitcoin and USD. "
            "Bitcoin price today is $61,670.32, with market data updated on this page. "
            "The calculator can convert BTC to USD."
        )

        fact = extract_key_fact_for_answer(text, "Найди актуальную цену Bitcoin")

        self.assertIn("$61,670.32", fact)
        self.assertNotIn("Use our converter", fact)

    def test_pipeline_answer_contains_price_value_when_excerpt_has_it(self):
        backend = FakeWebBackend([
            ResearchSource("CoinMarketCap Bitcoin", "https://example.com/cmc", snippet="Bitcoin price today"),
        ], {
            "https://example.com/cmc": (
                "Use our converter to compare BTC to USD. "
                "Bitcoin price today is $61,670.32 and the current market page shows recent movement."
            )
        })

        result = ResearchPipeline(backend).run("Найди актуальную цену Bitcoin", max_pages_to_read=1)

        self.assertIn("$61,670.32", result.answer)
        self.assertNotIn("Use our converter", result.answer)

    def test_russian_price_sentence_is_used_for_bitcoin_rate(self):
        backend = FakeWebBackend([
            ResearchSource("CoinMarketCap BTC", "https://example.com/cmc", snippet="Цена Bitcoin сегодня"),
        ], {
            "https://example.com/cmc": (
                "Используйте наш бесплатный конвертер для расчёта курса. "
                "Цена Bitcoin (Биткоин) в реальном времени сегодня составляет ₽4,435,960.93 RUB "
                "с суточным объемом торгов ₽2,622,727,330,151.70 RUB."
            )
        })

        result = ResearchPipeline(backend).run("Найди курс биткоина актуальный", max_pages_to_read=1)

        self.assertIn("₽4,435,960.93 RUB", result.answer)
        self.assertNotIn("бесплатный конвертер", result.answer.lower())
        self.assertTrue(result.citations)
        self.assertIn("₽4,435,960.93 RUB", result.citations[0].reason)
        assert_no_user_visible_research_garbage(self, result.answer)

    def test_btc_price_final_answer_uses_strict_clean_template(self):
        backend = FakeWebBackend([
            ResearchSource("CoinMarketCap Bitcoin", "https://coinmarketcap.com/currencies/bitcoin/", snippet="Bitcoin price today"),
            ResearchSource("ProFinance Bitcoin", "https://www.profinance.ru/charts/btc-rub/", snippet="Bitcoin RUB price"),
            ResearchSource("Crypto Navigation", "https://crypto.example.com/btc", snippet="BTC RUB market"),
        ], {
            "https://coinmarketcap.com/currencies/bitcoin/": (
                "Use our converter to compare BTC to RUB. "
                "Цена Bitcoin (Биткоин) в реальном времени сегодня составляет ₽4,398,766.70 RUB. "
                "Market prices change in real time."
            ),
            "https://www.profinance.ru/charts/btc-rub/": (
                "Используя сайт profinance.ru, Вы соглашаетесь с Политикой обработки персональных данных. "
                "Я согласен. Charts loading... калькулятор валют история стоимости кросс-курс онлайн график. "
                "Цена Bitcoin (Биткоин) в реальном времени сегодня составляет ₽4,398,766.70 RUB."
            ),
            "https://crypto.example.com/btc": (
                "Главная ➜ Криптовалюты ➜ Bitcoin ➜ Калькулятор валют ➜ История стоимости ➜ Онлайн график"
            ),
        })

        result = ResearchPipeline(backend).run("Дай актуальный курс биткоина", max_pages_to_read=2)

        self.assertIn("Коротко:", result.answer)
        self.assertIn("Источники:", result.answer)
        self.assertIn("₽4,398,766.70 RUB", result.answer)
        self.assertRegex(result.answer, r"(coinmarketcap\.com|profinance\.ru)")
        self.assertNotIn("Почему релевантно", result.answer)
        self.assertNotIn("Используя сайт", result.answer)
        self.assertNotIn("Я согласен", result.answer)
        self.assertNotIn("Charts loading", result.answer)
        self.assertNotIn("калькулятор валют", result.answer)
        self.assertNotIn("история стоимости", result.answer)
        self.assertNotIn("➜", result.answer)
        self.assertNotRegex(result.answer, r"https?://")
        self.assertLessEqual(len([line for line in result.answer.splitlines() if line.startswith("* ")]), 3)
        source_lines = [line for line in result.answer.splitlines() if re.match(r"^\d+\. ", line)]
        self.assertTrue(source_lines)
        self.assertLessEqual(len(source_lines), 3)
        assert_no_user_visible_research_garbage(self, result.answer)

    def test_format_research_final_answer_excludes_relevance_dump(self):
        source = ResearchSource(
            "CoinMarketCap Bitcoin",
            "https://coinmarketcap.com/currencies/bitcoin/",
            snippet="Bitcoin price today",
            domain="coinmarketcap.com",
            read_ok=True,
            excerpt="Bitcoin price today is $61,670.32.",
            relevance_score=1.0,
            read_status="read",
        )

        answer = format_research_final_answer(None, "Bitcoin price", [source], [], [])

        self.assertIn("Коротко:", answer)
        self.assertIn("Источники:", answer)
        self.assertIn("$61,670.32", answer)
        self.assertIn("CoinMarketCap Bitcoin — coinmarketcap.com", answer)
        self.assertNotIn("Почему релевантно", answer)
        self.assertNotIn("https://coinmarketcap.com", answer)
        self.assertRegex(answer, r"Коротко:\n\n\* .+\n\nИсточники:\n\n1\. ")

    def test_noisy_failed_sources_not_listed_in_main_sources(self):
        readable = ResearchSource(
            "Readable BTC",
            "https://example.com/read",
            snippet="Bitcoin price",
            domain="example.com",
            read_ok=True,
            excerpt="Bitcoin price today is $61,670.32.",
            relevance_score=1.0,
            read_status="read",
        )
        noisy = ResearchSource(
            "Noisy BTC",
            "https://example.com/noisy",
            snippet="BTC",
            domain="example.com",
            read_ok=False,
            excerpt="window.initData = {}; Charts loading...",
            read_status="js_bootstrap",
            failure_reason="js_bootstrap",
        )

        answer = format_research_final_answer(None, "Bitcoin price", [readable], [], [noisy])

        self.assertIn("Readable BTC — example.com", answer)
        self.assertNotIn("Noisy BTC", answer)
        self.assertIn("Ограничение:", answer)
        self.assertIn("Часть источников не удалось прочитать", answer)
        self.assertNotIn("js_bootstrap", answer)
        self.assertNotIn("Charts loading", answer)

    def test_snippet_only_answer_uses_short_source_mark(self):
        snippet = ResearchSource(
            "BTC snippet",
            "https://example.com/snippet",
            snippet="Bitcoin price today is $61,670.32.",
            domain="example.com",
            read_ok=False,
            snippet_only=True,
            read_status="failed",
            relevance_score=1.0,
        )

        answer = format_research_final_answer(None, "Bitcoin price", [], [snippet], [snippet])

        self.assertIn("$61,670.32", answer)
        self.assertIn("BTC snippet — example.com (по сниппету)", answer)
        self.assertIn("Ответ составлен только по поисковым сниппетам", answer)
        self.assertNotIn("failed", answer)

    def test_technical_docs_query_keeps_legitimate_terms(self):
        backend = FakeWebBackend([
            ResearchSource("Qt Docs QSplitter", "https://doc.qt.io/qt-6/qsplitter.html", snippet="QSplitter stylesheet handle"),
        ], {
            "https://doc.qt.io/qt-6/qsplitter.html": (
                "QSplitter supports styling its splitter handle with the QSplitter::handle selector. "
                "A stylesheet can set handle width, background, and margins."
            )
        })

        result = ResearchPipeline(backend).run("Найди документацию PyQt6 QSplitter stylesheet handle", max_pages_to_read=1)

        self.assertIn("Коротко:", result.answer)
        self.assertIn("Источники:", result.answer)
        self.assertIn("QSplitter", result.answer)
        self.assertIn("handle", result.answer)
        self.assertNotIn("Почему релевантно", result.answer)

    def test_failed_ascii_error_is_internal_not_user_visible(self):
        backend = FakeWebBackend([
            ResearchSource("CoinGecko", "https://example.com/coingecko", snippet="Bitcoin price today"),
        ])

        def broken_fetch(url: str, max_chars: int = 10000):
            raise UnicodeEncodeError("ascii", "Цена Bitcoin", 0, 4, "ordinal not in range(128)")

        backend.fetch = broken_fetch  # type: ignore[method-assign]

        result = ResearchPipeline(backend).run("Найди актуальную цену Bitcoin", max_pages_to_read=1)
        card = result.ranked_sources[0].to_card()

        self.assertIn("ascii", result.ranked_sources[0].failure_reason.lower())
        self.assertNotIn("ascii codec", card["failure_reason"])
        self.assertNotIn("UnicodeEncodeError", card["failure_reason"])
        assert_no_user_visible_research_garbage(self, card["excerpt"])
        assert_no_user_visible_research_garbage(self, card["failure_reason"])
        assert_no_user_visible_research_garbage(self, result.answer)

    def test_http_403_error_is_friendly_in_card(self):
        backend = FakeWebBackend([
            ResearchSource("Investing", "https://example.com/investing", snippet="Bitcoin price today"),
        ])

        def forbidden_fetch(url: str, max_chars: int = 10000):
            return FetchedPage(url=url, ok=False, error="HTTP Error 403: Forbidden")

        backend.fetch = forbidden_fetch  # type: ignore[method-assign]

        result = ResearchPipeline(backend).run("Найди актуальную цену Bitcoin", max_pages_to_read=1)
        card = result.ranked_sources[0].to_card()

        self.assertIn("не удалось безопасно прочитать", card["excerpt"])
        self.assertNotIn("HTTP Error 403", card["excerpt"])
        self.assertNotIn("HTTP Error 403", card["failure_reason"])
        assert_no_user_visible_research_garbage(self, result.answer)

    def test_final_answer_safety_net_removes_dirty_tokens(self):
        dirty = (
            "Коротко: window.initData = {}; document.documentElement.x = 1; "
            "HTTP Error 403: Forbidden [truncated] body { font-family: Arial; }"
        )

        cleaned = clean_final_answer_for_user(dirty)

        assert_no_user_visible_research_garbage(self, cleaned)
        self.assertTrue(cleaned)

    def test_clean_user_visible_text_removes_debug_tokens(self):
        cleaned = clean_user_visible_text(
            "HTTP Error 403: Forbidden window.gokuProps = {}; font-family: Arial; [truncated]"
        )

        assert_no_user_visible_research_garbage(self, cleaned)

    def test_one_readable_source_and_three_noisy_sources_uses_readable_only(self):
        noisy_pages = {
            f"https://example.com/noisy-{i}": (
                f"window.initData = {{\"item\": {i}, \"cookieDomainList\": [\"x.test\"]}}; "
                "document.documentElement.className='x'; function boot(){return window.initData;}"
            )
            for i in range(3)
        }
        backend = FakeWebBackend([
            ResearchSource(f"Noisy {i}", f"https://example.com/noisy-{i}", snippet=f"PyQt noisy snippet {i}")
            for i in range(3)
        ] + [
            ResearchSource("Readable", "https://example.com/readable", snippet="PyQt6 QSplitter handle docs")
        ], {
            **noisy_pages,
            "https://example.com/readable": (
                "PyQt6 QSplitter handle styling uses QSplitter::handle. "
                "Applications can set handle width and background in a stylesheet."
            ),
        })

        result = ResearchPipeline(backend).run("Найди документацию PyQt6 QSplitter handle", max_pages_to_read=1)

        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].title, "Readable")
        self.assertEqual(len([s for s in result.ranked_sources if not s.read_ok]), 3)
        self.assertNotIn("window.initData", result.answer)
        self.assertIn("не удалось прочитать как человеческий текст", result.answer)

    def test_no_readable_sources_with_useful_snippets_returns_snippet_only_answer(self):
        backend = FakeWebBackend([
            ResearchSource("TradingView", "https://example.com/tv", snippet="Bitcoin price today is $61,670.32."),
        ], {
            "https://example.com/tv": (
                "window.initData = {\"BTCUSDT\":102000,\"cookieDomainList\":[\"tradingview.com\"]}; "
                "document.documentElement.className='dark'; function boot(){return window.initData;}"
            )
        })

        result = ResearchPipeline(backend).run("Найди актуальную цену Bitcoin", max_pages_to_read=1)

        self.assertEqual(result.error, "no readable sources")
        self.assertEqual(len(result.sources), 1)
        self.assertTrue(result.sources[0].snippet_only)
        self.assertFalse(result.sources[0].read_ok)
        self.assertIn("по поисковым сниппетам", result.answer.lower())
        self.assertNotIn("window.initData", result.answer)

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

    def test_duckduckgo_lite_parser_extracts_results(self):
        html = '''
        <tr><td>
          <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fopenai.com%2Fnews%2F&amp;rut=x" class='result-link'>OpenAI News</a>
        </td></tr>
        <tr><td class='result-snippet'>Stay up to speed on AI news.</td></tr>
        '''
        from core.research import _parse_duckduckgo_lite_html

        results = _parse_duckduckgo_lite_html(html, max_results=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://openai.com/news/")
        self.assertEqual(results[0].snippet, "Stay up to speed on AI news.")
        self.assertEqual(results[0].source, "duckduckgo_lite")

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
        self.assertIn("обезличенный запрос", result.answer)
        self.assertIn("[personal email]", result.answer)
        self.assertNotIn("user@example.com", result.answer)

    def test_public_fallback_query_keeps_only_public_keywords(self):
        self.assertEqual(
            fallback_public_search_query("Найди свежие новости про OpenAI"),
            "OpenAI news latest",
        )

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

    def test_researchworker_stable_question_answers_without_search_and_without_self_dismissal(self):
        profile = AIProfile(id="researcher", name="Поисковик", kind=ProfileKind.RESEARCHER)
        backend = FakeWebBackend()
        worker = ResearchWorker(profile, "Что такое RAG простыми словами?", backend=backend)
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)

        worker.run()

        answer = "".join(chunks)
        self.assertEqual(backend.search_calls, [])
        self.assertIn("RAG", answer)
        self.assertIn("релевантные документы", answer)
        self.assertIn("Интернет-поиск здесь не нужен", answer)
        self.assertNotIn("ищи сам", answer.lower())
        self.assertNotIn("я не поисковая модель", answer.lower())

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
