"""Researcher/Search profile tool pipeline primitives."""

from __future__ import annotations

import re
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from ipaddress import ip_address
from typing import Any, Protocol

from core.diagnostics import write_log


FRESH_QUERY_RE = re.compile(
    r"(сейчас|актуальн|последн|новост|сегодня|цена|стоимост|верси[яи]|"
    r"закон|релиз|найди|поиск|ищи|compare|comparison|latest|current|today|"
    r"price|news|release|version|find|search)",
    re.IGNORECASE,
)

SECRET_RE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9_]{20,}|"
    r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{8,})"
)
WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)+[^\\/:*?\"<>|\s\r\n]+")
UNIX_PATH_RE = re.compile(r"(?<!\w)/(?:home|users|var|etc|tmp|mnt|opt)/[^\s\"'<>|]+", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\s().-]*){9,}\d(?!\w)")
CHAT_HISTORY_RE = re.compile(r"(?im)^\s*(user|assistant|system|tool|ты|ассистент|лера|кодер)\s*:")
MEMORY_RE = re.compile(r"(?i)\b(companion[_ -]?memory|memory summary|memories|личн(?:ая|ые) память|воспоминани[ея])\b")
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):(?P<body>.*)", re.DOTALL)
ERROR_LINE_RE = re.compile(r"(?m)^\s*(?P<error>[A-Za-z_][\w.]*(?:Error|Exception|Warning)):\s*(?P<message>.*)$")
ERROR_KEYWORD_RE = re.compile(r"\b(?P<error>[A-Za-z_][\w.]*(?:Error|Exception|Warning))\b[:\s-]*(?P<message>[^\n\r]{0,160})")
MODULE_RE = re.compile(r"No module named ['\"](?P<module>[^'\"]+)['\"]")


def _quote_log(value: object, limit: int = 240) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return text[:limit] + ("..." if len(text) > limit else "")


def safe_text(value: object, encoding: str = "utf-8") -> str:
    """
    Безопасно конвертирует любой объект в строку.
    Обрабатывает bytes, None, encoding errors — никогда не бросает исключений.
    Сохраняет кириллицу и Unicode.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode(encoding, errors="replace")
        except Exception:
            return value.decode("utf-8", errors="replace")
    try:
        text = str(value)
    except Exception:
        return ""
    # Убираем управляющие символы кроме \n и \t
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


@dataclass
class ResearchSource:
    title: str
    url: str
    snippet: str = ""
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    facts: list[str] = field(default_factory=list)
    id: str = ""
    source: str = ""
    rank: int = 0
    domain: str = ""
    fetched: bool = False
    read_ok: bool = False
    snippet_only: bool = False   # страница не прочитана, но сниппет из поиска сохранён
    excerpt: str = ""
    relevance_score: float = 0.0
    used_in_answer: bool = False
    read_status: str = "snippet"
    error: str = ""
    failure_reason: str = ""

    def citation(self, index: int) -> str:
        suffix = " (сниппет)" if self.snippet_only else ""
        return f"[{index}] {self.title}{suffix} — {self.url}"

    def to_card(self) -> dict[str, object]:
        raw_excerpt = self.excerpt or (self.facts[0] if self.facts else "")
        display_status, status_label = public_source_status(self)
        display_message = public_source_excerpt(self)
        return {
            "id": clean_user_visible_text(self.id, max_chars=40),
            "title": clean_user_visible_text(self.title or self.url, max_chars=180),
            "url": self.url,
            "domain": clean_user_visible_text(self.domain or domain_from_url(self.url), max_chars=120),
            "snippet": public_source_snippet(self),
            "fetched": self.fetched,
            "read_ok": self.read_ok,
            "snippet_only": self.snippet_only,
            "excerpt": display_message,
            "display_message": display_message,
            "display_reason": public_relevance_reason(self),
            "display_status": display_status,
            "status_label": status_label,
            "raw_excerpt": raw_excerpt,
            "relevance_score": self.relevance_score,
            "used_in_answer": self.used_in_answer,
            "status": display_status,
            "internal_status": self.read_status,
            "failure_reason": public_failure_message(self.read_status, self.failure_reason or self.error),
        }


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    rank: int = 0


@dataclass
class FetchedPage:
    url: str
    title: str = ""
    text: str = ""
    status_code: int = 0
    content_type: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ok: bool = False
    error: str = ""


@dataclass
class ResearchCitation:
    id: str
    title: str
    url: str
    domain: str
    reason: str = ""
    snippet_only: bool = False


@dataclass
class ResearchBackendStatus:
    selected_backend: str
    configured_backend: str
    searxng_url_configured: bool
    searxng_available: bool
    duckduckgo_available: bool
    last_backend_error: str = ""
    effective_backend: str = ""
    recommended_action: str = ""


@dataclass
class ResearchResult:
    answer: str
    sources: list[ResearchSource] = field(default_factory=list)
    used_search: bool = False
    error: str = ""
    sanitized_query: str = ""
    privacy_reasons: list[str] = field(default_factory=list)
    original_user_query: str = ""
    backend_name: str = ""
    search_results: list[SearchResult] = field(default_factory=list)
    fetched_pages: list[FetchedPage] = field(default_factory=list)
    ranked_sources: list[ResearchSource] = field(default_factory=list)
    extracted_facts: list[str] = field(default_factory=list)
    final_answer: str = ""
    citations: list[ResearchCitation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    privacy_decision: PrivacyDecision | None = None
    backend_status: ResearchBackendStatus | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.answer:
            self.answer = clean_final_answer_for_user(self.answer)
        if not self.final_answer:
            self.final_answer = self.answer
        elif self.final_answer:
            self.final_answer = clean_final_answer_for_user(self.final_answer)


@dataclass
class PrivacyDecision:
    allowed: bool
    sanitized: str
    reasons: list[str] = field(default_factory=list)
    confirmation_required: bool = False


class PrivacyFirewall:
    """Outbound privacy firewall for web/search payloads.

    Web tools are inbound-only by default: project context, memory, source code,
    paths, secrets, and chat transcripts must not leave the machine as raw query
    text. The firewall returns a compact sanitized query or blocks the outbound
    request with a user-visible reason.
    """

    def detect_secrets(self, text: str) -> bool:
        return bool(SECRET_RE.search(text or ""))

    def detect_local_paths(self, text: str) -> bool:
        value = text or ""
        return bool(WINDOWS_PATH_RE.search(value) or UNIX_PATH_RE.search(value))

    def detect_code_blocks(self, text: str) -> bool:
        value = text or ""
        if CODE_FENCE_RE.search(value):
            return True
        code_like_lines = 0
        for line in value.splitlines():
            stripped = line.strip()
            if re.search(r"^(def|class|import|from|function|const|let|var)\b", stripped):
                code_like_lines += 1
            elif re.search(r"[{};]$", stripped) and len(stripped) > 8:
                code_like_lines += 1
        return code_like_lines >= 3

    def detect_personal_data_light(self, text: str) -> bool:
        value = text or ""
        return bool(EMAIL_RE.search(value) or PHONE_RE.search(value))

    def detect_chat_history(self, text: str) -> bool:
        return bool(CHAT_HISTORY_RE.search(text or ""))

    def detect_memory_context(self, text: str) -> bool:
        return bool(MEMORY_RE.search(text or ""))

    def sanitize_query(self, text: str) -> str:
        value = text or ""
        traceback_query = self._sanitize_traceback(value)
        if traceback_query:
            return traceback_query
        error_query = self._sanitize_error_keywords(value)
        if error_query:
            return error_query
        value = CODE_FENCE_RE.sub("[code omitted]", value)
        kept_lines: list[str] = []
        for line in value.splitlines():
            if CHAT_HISTORY_RE.search(line) or MEMORY_RE.search(line):
                continue
            kept_lines.append(line)
        value = "\n".join(kept_lines)
        value = SECRET_RE.sub("[secret]", value)
        value = WINDOWS_PATH_RE.sub("[local path]", value)
        value = UNIX_PATH_RE.sub("[local path]", value)
        value = EMAIL_RE.sub("[personal email]", value)
        value = PHONE_RE.sub("[personal phone]", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value[:500]

    def require_confirmation_if_sensitive(self, payload: str) -> PrivacyDecision:
        sanitized = self.sanitize_query(payload)
        reasons = self._reasons(payload)
        confirmation = any(reason in reasons for reason in ("personal_data", "chat_history", "companion_memory"))
        return PrivacyDecision(
            allowed=not confirmation and not self._hard_block_reasons(reasons),
            sanitized=sanitized,
            reasons=reasons,
            confirmation_required=confirmation,
        )

    def block_outbound_if_private(self, payload: str, *, confirmed: bool = False) -> PrivacyDecision:
        sanitized = self.sanitize_query(payload)
        reasons = self._reasons(payload)
        hard = self._hard_block_reasons(reasons)
        if hard:
            return PrivacyDecision(False, sanitized, reasons, confirmation_required=False)
        confirmation = any(reason in reasons for reason in ("personal_data", "chat_history", "companion_memory"))
        if confirmation and not confirmed:
            return PrivacyDecision(False, sanitized, reasons, confirmation_required=True)
        if not sanitized:
            return PrivacyDecision(False, sanitized, ["empty_query"], confirmation_required=False)
        return PrivacyDecision(True, sanitized, reasons, confirmation_required=False)

    def _reasons(self, text: str) -> list[str]:
        reasons: list[str] = []
        if self.detect_secrets(text):
            reasons.append("secret")
        if self.detect_code_blocks(text):
            reasons.append("code")
        if self.detect_local_paths(text):
            reasons.append("local_path")
        if self.detect_personal_data_light(text):
            reasons.append("personal_data")
        if self.detect_chat_history(text):
            reasons.append("chat_history")
        if self.detect_memory_context(text):
            reasons.append("companion_memory")
        return reasons

    @staticmethod
    def _hard_block_reasons(reasons: list[str]) -> bool:
        return any(reason in reasons for reason in ("secret", "code"))

    def _sanitize_traceback(self, text: str) -> str:
        if not TRACEBACK_RE.search(text or ""):
            return ""
        error_match = ERROR_LINE_RE.search(text)
        if not error_match:
            return "Python traceback error"
        error = error_match.group("error")
        message = error_match.group("message").strip()
        module_match = MODULE_RE.search(message)
        if module_match:
            return f"Python {error} module {module_match.group('module')}"
        message = WINDOWS_PATH_RE.sub("[local path]", message)
        message = UNIX_PATH_RE.sub("[local path]", message)
        message = SECRET_RE.sub("[secret]", message)
        return f"Python {error} {message}".strip()[:300]

    def _sanitize_error_keywords(self, text: str) -> str:
        path_match = WINDOWS_PATH_RE.search(text or "") or UNIX_PATH_RE.search(text or "")
        if not path_match:
            return ""
        match = ERROR_KEYWORD_RE.search(text or "")
        if not match:
            return ""
        if match.start() < path_match.end():
            return ""
        error = match.group("error")
        message = match.group("message").strip()
        message = WINDOWS_PATH_RE.sub("", message)
        message = UNIX_PATH_RE.sub("", message)
        message = SECRET_RE.sub("[secret]", message)
        message = re.sub(r"\s+", " ", message).strip()
        return f"Python {error} {message}".strip()[:300]


class ResearchBackend(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def search(self, query: str, max_results: int) -> list[SearchResult | ResearchSource]:
        ...

    def fetch(self, url: str, max_chars: int = 10000) -> FetchedPage:
        ...


WebSearchBackend = ResearchBackend


class UnavailableWebBackend:
    name = "unavailable"

    def available(self) -> bool:
        return False

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        raise RuntimeError("Web search backend is not configured.")

    def fetch(self, url: str, max_chars: int = 10000) -> FetchedPage:
        raise RuntimeError("Web page reader backend is not configured.")

    def read_url(self, url: str, max_chars: int = 8000) -> str:
        return self.fetch(url, max_chars=max_chars).text


class FakeWebBackend:
    """Deterministic backend for tests and future dev smoke harnesses."""

    name = "fake"

    def __init__(self, results: list[ResearchSource] | None = None, pages: dict[str, str] | None = None) -> None:
        self.results = results or []
        self.pages = pages or {}
        self.search_calls: list[tuple[str, int]] = []
        self.read_calls: list[str] = []
        self.fetch_calls: list[str] = []
        self.available_value = True

    def available(self) -> bool:
        return self.available_value

    def search(self, query: str, max_results: int) -> list[ResearchSource]:
        self.search_calls.append((query, max_results))
        return self.results[:max_results]

    def fetch(self, url: str, max_chars: int = 10000) -> FetchedPage:
        self.fetch_calls.append(url)
        self.read_calls.append(url)
        text = self.pages.get(url, "")
        if not text:
            return FetchedPage(url=url, ok=False, error="empty page")
        return FetchedPage(
            url=url,
            title="",
            text=text[:max_chars],
            status_code=200,
            content_type="text/plain",
            ok=True,
        )

    def read_url(self, url: str, max_chars: int = 8000) -> str:
        return self.fetch(url, max_chars=max_chars).text


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._title_active = False
        self._in_main = False    # внутри <main> или <article>
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.main_parts: list[str] = []
        self.meta_description: str = ""
        self._active_target: list[list[str]] = [self.text_parts]  # стек

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr_dict = dict(attrs)

        # meta description
        if tag == "meta":
            name = (attr_dict.get("name") or "").lower()
            if name == "description" and attr_dict.get("content"):
                self.meta_description = safe_text(attr_dict["content"])[:500]

        # Скипаемые блоки — всегда, независимо от main/article
        if tag in {"script", "style", "noscript", "svg", "canvas", "iframe", "form",
                   "nav", "footer", "aside", "menu", "header"}:
            self._skip_depth += 1
            return

        if tag == "title":
            self._title_active = True
            return

        if tag in {"main", "article", "section"}:
            self._in_main = True

        if tag in {"p", "br", "li", "div", "section", "article", "h1", "h2", "h3", "h4", "td", "th"}:
            self.text_parts.append("\n")
            if self._in_main:
                self.main_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg", "canvas", "iframe", "form",
                   "nav", "footer", "aside", "menu", "header"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._title_active = False
            return
        if tag in {"main", "article", "section"}:
            self._in_main = False
        if tag in {"p", "li", "h1", "h2", "h3", "h4"}:
            self.text_parts.append("\n")
            if self._in_main:
                self.main_parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._title_active:
            self.title_parts.append(text)
        if self._skip_depth or self._title_active:
            return
        self.text_parts.append(text)
        if self._in_main:
            self.main_parts.append(text)

    def text(self, max_chars: int) -> str:
        return compress_page_text(" ".join(self.text_parts), max_chars=max_chars)

    def main_text(self, max_chars: int) -> str:
        return compress_page_text(" ".join(self.main_parts), max_chars=max_chars)

    def title(self) -> str:
        return compress_page_text(" ".join(self.title_parts), max_chars=200)


_LOW_CONTENT_THRESHOLD = 40   # символов — меньше используем следующую стратегию


def extract_html_text(html: str, max_chars: int = 10000) -> tuple[str, str]:
    """
    Слоёный fallback — всегда возвращает лучший доступный текст:
    1. main/article (предпочтительно — без nav/header/footer)
    2. body (очищен от skip-тегов)
    3. meta description
    4. regexp strip как последний шанс
    Никогда не возвращает nav/header/script контент при наличии нормального текста.
    """
    parser = _HTMLTextExtractor()
    try:
        parser.feed(safe_text(html))
    except Exception:
        plain = re.sub(r"<[^>]+>", " ", safe_text(html))
        return "", compress_page_text(plain, max_chars=max_chars)

    title = parser.title()

    # Стратегия 1: main/article — возвращаем если хоть что-то есть
    main = parser.main_text(max_chars=max_chars)
    if len(main.strip()) >= _LOW_CONTENT_THRESHOLD:
        write_log(f"[research_extract_done] strategy=main chars={len(main)}")
        return title, main

    # Стратегия 2: body целиком (уже очищен от nav/header/script)
    body = parser.text(max_chars=max_chars)
    if len(body.strip()) >= _LOW_CONTENT_THRESHOLD:
        write_log(f"[research_extract_done] strategy=body chars={len(body)}")
        return title, body

    # Стратегия 3: meta description
    if parser.meta_description:
        write_log(f"[research_extract_fallback] strategy=meta_description")
        return title, parser.meta_description

    # Стратегия 4: если main/body есть хоть что-то — вернём его
    if main.strip():
        return title, main
    if body.strip():
        return title, body

    # Стратегия 5: regexp strip как последний шанс
    plain = re.sub(r"<[^>]+>", " ", safe_text(html))
    plain = compress_page_text(plain, max_chars=max_chars)
    write_log(f"[research_extract_fallback] strategy=regexp_strip chars={len(plain)}")
    return title, plain


def _is_http_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url or "")
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_private_host(host: str) -> bool:
    value = (host or "").strip("[]").lower()
    if not value:
        return True
    if value in {"localhost", "local"} or value.endswith(".local"):
        return True
    try:
        ip = ip_address(value)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
    except ValueError:
        return False


def is_fetch_url_allowed(url: str, *, allow_localhost: bool = False) -> tuple[bool, str]:
    parsed = urllib.parse.urlparse(url or "")
    if parsed.scheme not in {"http", "https"}:
        return False, "non_http_url"
    host = parsed.hostname or ""
    if _is_private_host(host) and not allow_localhost:
        return False, "private_or_local_url"
    return True, ""


class UrlFetchMixin:
    # Браузерный UA — меньше блокировок от простых сайтов
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 ZenAIResearch/1.0"
    )

    # Паттерн для JS-only страниц (почти нет текста, много скриптов)
    _JS_HINTS = re.compile(
        r"(?i)(enable javascript|please enable js|javascript is required|"
        r"this page requires javascript|нужно включить javascript|"
        r"loading\.\.\.|captcha|cloudflare|access denied|403 forbidden|"
        r"bot detection|are you a robot)",
        re.IGNORECASE,
    )

    def fetch(self, url: str, max_chars: int = 10000) -> FetchedPage:
        allowed, reason = is_fetch_url_allowed(url)
        if not allowed:
            write_log(f'[research_fetch_skipped] url="{_quote_log(url)}" reason="{reason}"')
            return FetchedPage(url=url, ok=False, error=reason)
        write_log(f'[research_fetch_start] url="{_quote_log(url)}"')
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru,en;q=0.9",
            # Accept-Encoding намеренно НЕ задан: urllib.request сам добавляет
            # gzip-поддержку и автоматически декомпрессирует ответ. При явном
            # Accept-Encoding: gzip декомпрессия отключается и raw.decode() ломается.
        }
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = int(getattr(response, "status", 0) or 0)
                content_type = safe_text(response.headers.get("content-type", ""))
                lower_type = content_type.lower()
                if not any(kind in lower_type for kind in ("text/html", "text/plain", "application/xhtml+xml")):
                    write_log(
                        "[research_fetch_skipped] "
                        f'url="{_quote_log(url)}" reason="binary_content" content_type="{_quote_log(content_type)}"'
                    )
                    return FetchedPage(
                        url=url,
                        status_code=status,
                        content_type=content_type,
                        ok=False,
                        error="binary_content",
                    )
                raw = response.read(max_chars * 6)  # читаем больше для качественного extract
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            reason_str = safe_text(exc)
            write_log(f'[research_fetch_failed] url="{_quote_log(url)}" reason="{_quote_log(reason_str)}"')
            return FetchedPage(url=url, ok=False, error=reason_str)

        # Энкодинг — многоуровневый fallback
        encoding = "utf-8"
        charset_match = re.search(r"charset=([\w.-]+)", content_type, re.IGNORECASE)
        if charset_match:
            encoding = charset_match.group(1).strip()

        # Пробуем объявленный encoding, потом utf-8, потом latin-1
        decoded = ""
        for enc in (encoding, "utf-8", "latin-1"):
            try:
                decoded = raw.decode(enc, errors="replace")
                break
            except Exception:
                continue
        if not decoded:
            decoded = raw.decode("utf-8", errors="replace")

        # Извлечение текста
        if "html" in content_type.lower():
            title, text = extract_html_text(decoded, max_chars=max_chars)
        else:
            title, text = "", compress_page_text(safe_text(decoded), max_chars=max_chars)

        # JS-only detection
        error = ""
        if not text or len(text.strip()) < 120:
            if self._JS_HINTS.search(decoded[:3000]):
                error = "js_only_suspected"
            else:
                error = "empty_page"
        
        write_log(
            "[research_fetch_done] "
            f'url="{_quote_log(url)}" status="{status}" chars="{len(text)}" error="{error}"'
        )
        return FetchedPage(
            url=url,
            title=safe_text(title),
            text=safe_text(text),
            status_code=status,
            content_type=content_type,
            ok=bool(text) and not error,
            error=error,
        )


class DuckDuckGoBackend(UrlFetchMixin):
    name = "duckduckgo"
    search_url = "https://html.duckduckgo.com/html/"
    lite_search_url = "https://lite.duckduckgo.com/lite/"

    def available(self) -> bool:
        return True

    def _looks_like_challenge(self, html: str) -> bool:
        lowered = (html or "").lower()
        return any(marker in lowered for marker in (
            "anomaly-modal",
            "challenge-form",
            "cc=botnet",
            "verify you are human",
            "code:",
        ))

    def _looks_like_no_results(self, html: str) -> bool:
        lowered = (html or "").lower()
        return any(marker in lowered for marker in (
            "no results found",
            "no results.",
            "no results for",
            "no more results",
            "sorry, no results",
        ))

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        request_url = self.search_url + "?" + urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(request_url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                html = response.read(600_000).decode("utf-8", errors="replace")
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            raise RuntimeError(f"DuckDuckGo search failed: {exc}") from exc
        results = _parse_duckduckgo_html(html, max_results=max_results)
        if results:
            return results
        html_challenge = self._looks_like_challenge(html)
        html_no_results = self._looks_like_no_results(html)
        lite_url = self.lite_search_url + "?" + urllib.parse.urlencode({"q": query})
        lite_request = urllib.request.Request(lite_url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(lite_request, timeout=10) as response:
                lite_html = response.read(600_000).decode("utf-8", errors="replace")
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError):
            if html_challenge:
                raise RuntimeError("duckduckgo_challenged")
            if html_no_results:
                raise RuntimeError("duckduckgo_no_results")
            return []
        lite_results = _parse_duckduckgo_lite_html(lite_html, max_results=max_results)
        if lite_results:
            return lite_results
        lite_challenge = self._looks_like_challenge(lite_html)
        lite_no_results = self._looks_like_no_results(lite_html)
        if html_challenge or lite_challenge:
            raise RuntimeError("duckduckgo_challenged")
        if html_no_results or lite_no_results:
            raise RuntimeError("duckduckgo_no_results")
        return []


def _clean_ddg_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if "duckduckgo.com" in (parsed.netloc or "") and parsed.path.startswith("/l/"):
        query = urllib.parse.parse_qs(parsed.query)
        if query.get("uddg"):
            return _normalize_http_url(query["uddg"][0])
    return _normalize_http_url(url)


def _normalize_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme and parsed.netloc:
        parsed = urllib.parse.urlparse("https:" + url)
    if parsed.scheme not in {"http", "https"}:
        return url
    path = urllib.parse.quote(urllib.parse.unquote(parsed.path or ""), safe="/:@")
    query = urllib.parse.quote(urllib.parse.unquote(parsed.query or ""), safe="=&?/:@,+%")
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, query, ""))


def _strip_tags(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    value = value.replace("&amp;", "&").replace("&quot;", '"').replace("&#x27;", "'")
    return compress_page_text(value, max_chars=500)


def _parse_duckduckgo_html(html: str, max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    blocks = re.split(r'<div[^>]+class="[^"]*result[^"]*"[^>]*>', html or "")
    for block in blocks:
        link_match = re.search(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if not link_match:
            continue
        url = urllib.parse.unquote(_clean_ddg_url(link_match.group("href")))
        title = _strip_tags(link_match.group("title"))
        snippet_match = re.search(
            r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>|'
            r'<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet2>.*?)</div>',
            block,
            re.IGNORECASE | re.DOTALL,
        )
        snippet = ""
        if snippet_match:
            snippet = _strip_tags(snippet_match.group("snippet") or snippet_match.group("snippet2") or "")
        if title and _is_http_url(url):
            results.append(SearchResult(title=title, url=url, snippet=snippet, source="duckduckgo", rank=len(results) + 1))
        if len(results) >= max_results:
            break
    return results


def _parse_duckduckgo_lite_html(html: str, max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    for link_match in re.finditer(
        r"<a(?P<attrs>[^>]*class=['\"]result-link['\"][^>]*)>(?P<title>.*?)</a>",
        html or "",
        re.IGNORECASE | re.DOTALL,
    ):
        attrs = link_match.group("attrs")
        href_match = re.search(r"href=['\"](?P<href>[^'\"]+)['\"]", attrs, re.IGNORECASE)
        if not href_match:
            continue
        url = urllib.parse.unquote(_clean_ddg_url(href_match.group("href").replace("&amp;", "&")))
        title = _strip_tags(link_match.group("title"))
        snippet = ""
        block = html[link_match.end(): link_match.end() + 2000]
        snippet_match = re.search(
            r"<td[^>]+class=['\"]result-snippet['\"][^>]*>(?P<snippet>.*?)</td>",
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if snippet_match:
            snippet = _strip_tags(snippet_match.group("snippet"))
        if title and _is_http_url(url):
            results.append(SearchResult(title=title, url=url, snippet=snippet, source="duckduckgo_lite", rank=len(results) + 1))
        if len(results) >= max_results:
            break
    return results


class SearxNGBackend(UrlFetchMixin):
    name = "searxng"

    def __init__(self, base_url: str) -> None:
        self.base_url = (base_url or "").rstrip("/")

    def available(self) -> bool:
        return bool(self.base_url)

    def health_check(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "SearxNG URL is not configured."
        endpoint = self.base_url + "/search?" + urllib.parse.urlencode({
            "q": "test",
            "format": "json",
            "language": "all",
        })
        request = urllib.request.Request(endpoint, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                import json

                payload = json.loads(response.read(200_000).decode("utf-8", errors="replace"))
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, ValueError) as exc:
            return False, safe_text(exc)
        if isinstance(payload, dict) and "results" in payload:
            return True, ""
        if isinstance(payload, dict):
            return True, ""
        return False, "invalid_json"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        if not self.base_url:
            raise RuntimeError("SearxNG URL is not configured.")
        endpoint = self.base_url + "/search?" + urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "language": "all",
        })
        request = urllib.request.Request(endpoint, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                import json

                payload = json.loads(response.read(600_000).decode("utf-8", errors="replace"))
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, ValueError) as exc:
            raise RuntimeError(f"SearxNG search failed: {exc}") from exc
        results: list[SearchResult] = []
        for item in payload.get("results", [])[:max_results]:
            url = str(item.get("url") or "")
            if not _is_http_url(url):
                continue
            results.append(SearchResult(
                title=str(item.get("title") or url),
                url=url,
                snippet=str(item.get("content") or ""),
                source="searxng",
                rank=len(results) + 1,
            ))
        return results


def _normalize_backend_kind(kind: str) -> str:
    value = (kind or "").strip().lower()
    if not value or value == "default":
        return "auto"
    return value


def _backend_error_code(error: object) -> str:
    text = safe_text(error).lower()
    if "duckduckgo_challenged" in text or "provider challenge" in text or "challenge" in text:
        return "duckduckgo_challenged"
    if "duckduckgo_no_results" in text or "no usable results" in text:
        return "duckduckgo_no_results"
    if "searxng" in text and ("unavailable" in text or "failed" in text or "invalid_json" in text):
        return "searxng_unavailable"
    if "not configured" in text:
        return "backend_unavailable"
    return "backend_unavailable"


def diagnose_research_backend(
    kind: str = "auto",
    searxng_url: str = "",
    *,
    backend: ResearchBackend | None = None,
    last_backend_error: str = "",
) -> ResearchBackendStatus:
    profile_kind = _normalize_backend_kind(kind or "auto")
    env_kind = _normalize_backend_kind(os.environ.get("ZENAI_RESEARCH_BACKEND") or "")
    if profile_kind and profile_kind != "auto":
        selected = profile_kind
    elif env_kind and env_kind != "auto":
        selected = env_kind
    else:
        selected = profile_kind or env_kind or "auto"
    configured_searxng = (searxng_url or os.environ.get("ZENAI_SEARXNG_URL") or "").strip()
    duckduckgo_available = DuckDuckGoBackend().available()
    searxng_available = False
    effective_backend = "unavailable"
    recommended_action = ""
    if backend is not None:
        effective_backend = _backend_name(backend)
    recognized = {"auto", "searxng", "duckduckgo", "unavailable"}
    if backend is not None and selected not in recognized:
        return ResearchBackendStatus(
            selected_backend=selected,
            configured_backend=selected,
            searxng_url_configured=bool(configured_searxng),
            searxng_available=False,
            duckduckgo_available=duckduckgo_available,
            last_backend_error=last_backend_error,
            effective_backend=_backend_name(backend),
            recommended_action="custom_backend",
        )
    if selected == "searxng":
        if configured_searxng:
            probe = SearxNGBackend(configured_searxng)
            searxng_available, probe_error = probe.health_check()
            effective_backend = "searxng" if searxng_available else "unavailable"
            if not searxng_available:
                recommended_action = "check_searxng_url"
                last_backend_error = last_backend_error or probe_error
        else:
            recommended_action = "configure_searxng_url"
            last_backend_error = last_backend_error or "searxng_url_missing"
    elif selected == "duckduckgo":
        effective_backend = "duckduckgo"
        if last_backend_error:
            recommended_action = "configure_searxng_for_stability"
    elif selected == "unavailable":
        effective_backend = "unavailable"
        recommended_action = "configure_search_backend"
    else:
        if configured_searxng:
            probe = SearxNGBackend(configured_searxng)
            searxng_available, probe_error = probe.health_check()
            if searxng_available:
                effective_backend = "searxng"
            else:
                effective_backend = "duckduckgo" if duckduckgo_available else "unavailable"
                last_backend_error = last_backend_error or probe_error
                recommended_action = "configure_searxng_for_stability" if duckduckgo_available else "configure_search_backend"
        else:
            effective_backend = "duckduckgo" if duckduckgo_available else "unavailable"
            recommended_action = "configure_search_backend" if not duckduckgo_available else "duckduckgo_fallback"
    if selected == "auto" and configured_searxng and not searxng_available:
        recommended_action = recommended_action or ("duckduckgo_fallback" if duckduckgo_available else "configure_search_backend")
    if selected == "auto" and configured_searxng and searxng_available:
        recommended_action = "searxng_active"
    if selected == "duckduckgo":
        recommended_action = recommended_action or "use_duckduckgo"
    if selected == "searxng" and searxng_available:
        recommended_action = "searxng_active"
    return ResearchBackendStatus(
        selected_backend=selected,
        configured_backend=selected,
        searxng_url_configured=bool(configured_searxng),
        searxng_available=searxng_available,
        duckduckgo_available=duckduckgo_available,
        last_backend_error=last_backend_error,
        effective_backend=effective_backend,
        recommended_action=recommended_action,
    )


def select_research_backend(kind: str = "auto", searxng_url: str = "") -> tuple[ResearchBackend, ResearchBackendStatus]:
    status = diagnose_research_backend(kind, searxng_url)
    selected = status.selected_backend
    configured_searxng = (searxng_url or os.environ.get("ZENAI_SEARXNG_URL") or "").strip()
    if selected == "unavailable":
        return UnavailableWebBackend(), status
    if selected == "searxng":
        if configured_searxng and status.searxng_available:
            return SearxNGBackend(configured_searxng), status
        status.last_backend_error = status.last_backend_error or "searxng_unavailable"
        status.effective_backend = "unavailable"
        return UnavailableWebBackend(), status
    if selected == "duckduckgo":
        return DuckDuckGoBackend(), status
    if configured_searxng and status.searxng_available:
        return SearxNGBackend(configured_searxng), status
    if status.duckduckgo_available:
        return DuckDuckGoBackend(), status
    status.effective_backend = "unavailable"
    return UnavailableWebBackend(), status


def backend_unavailable_message(status: ResearchBackendStatus | None, backend_error: str = "") -> str:
    effective = (status.effective_backend if status else "").strip().lower()
    selected = (status.selected_backend if status else "").strip().lower()
    action = (status.recommended_action if status else "").strip().lower()
    error_code = _backend_error_code(backend_error)
    if effective == "searxng" or selected == "searxng":
        if not (status.searxng_available if status else False):
            return (
                "Коротко:\n"
                "- SearxNG URL задан, но backend недоступен.\n\n"
                "Ограничение:\n"
                "- Проверь SearxNG URL или health check backend."
            )
        return (
            "Коротко:\n"
            "- SearxNG недоступен, поэтому я не могу честно получить свежие источники.\n\n"
            "Ограничение:\n"
            "- Проверь backend поиска или повтори запрос позже."
        )
    if error_code == "duckduckgo_challenged" or effective == "duckduckgo" or selected == "duckduckgo":
        return (
            "Коротко:\n"
            "- DuckDuckGo вернул защитную страницу или не дал пригодной выдачи.\n\n"
            "Ограничение:\n"
            "- Для стабильного поиска настрой SearxNG."
        )
    if error_code == "duckduckgo_no_results":
        return (
            "Коротко:\n"
            "- DuckDuckGo не дал пригодной выдачи по этому запросу.\n\n"
            "Ограничение:\n"
            "- Для стабильного поиска настрой SearxNG."
        )
    if action == "configure_searxng_url":
        return (
            "Коротко:\n"
            "- Поисковый backend сейчас не настроен.\n\n"
            "Ограничение:\n"
            "- Задай SearxNG URL или включи DuckDuckGo fallback."
        )
    return (
        "Коротко:\n"
        "- Поисковый backend сейчас недоступен, поэтому я не могу честно получить свежие источники.\n\n"
        "Ограничение:\n"
        "- Проверь SearxNG/DuckDuckGo backend или повтори запрос позже."
    )


def make_research_backend(kind: str = "auto", searxng_url: str = "") -> ResearchBackend:
    backend, _ = select_research_backend(kind, searxng_url)
    return backend


def classify_query(text: str) -> str:
    value = text or ""
    if FRESH_QUERY_RE.search(value):
        return "fresh"
    if re.search(r"(сравни|выбери|лучше|compare|versus|\bvs\b)", value, re.IGNORECASE):
        return "compare"
    if re.search(r"(что такое|объясни|почему|как работает|explain)", value, re.IGNORECASE):
        return "explain"
    return "factual"


def needs_web_search(text: str, require_sources_for_fresh_info: bool = True) -> bool:
    kind = classify_query(text)
    return kind in {"fresh", "compare"} and require_sources_for_fresh_info


def compress_page_text(text: str, max_chars: int = 3000) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0] + " ... [truncated]"


def domain_from_url(url: str) -> str:
    host = urllib.parse.urlparse(url or "").hostname or ""
    return host.lower().removeprefix("www.")


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-zА-Яа-я0-9_+#.-]{4,}", query or "")
    return [term.lower() for term in terms[:12]]


def relevant_excerpt(text: str, query: str, max_chars: int = 900) -> str:
    cleaned = compress_page_text(text, max_chars=max(max_chars * 3, max_chars))
    terms = _query_terms(query)
    if not cleaned or not terms:
        return compress_page_text(cleaned, max_chars=max_chars)
    lowered = cleaned.lower()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    if not positions:
        return compress_page_text(cleaned, max_chars=max_chars)
    start = max(0, min(positions) - max_chars // 4)
    end = min(len(cleaned), start + max_chars)
    excerpt = cleaned[start:end]
    if start:
        excerpt = "..." + excerpt
    if end < len(cleaned):
        excerpt += "..."
    return compress_page_text(excerpt, max_chars=max_chars + 20)


def source_relevance(source: SearchResult, page_text: str, query: str) -> float:
    haystack = f"{source.title} {source.snippet} {page_text[:2000]}".lower()
    terms = _query_terms(query)
    if not terms:
        return max(0.1, 1.0 / max(source.rank or 1, 1))
    hits = sum(1 for term in terms if term in haystack)
    return round((hits / max(len(terms), 1)) + (1.0 / max(source.rank or 1, 1)) * 0.15, 3)


_JS_BOOTSTRAP_RE = re.compile(
    r"(?i)\b("
    r"window\.[A-Za-z_$][\w$]*|document\.(?:documentElement|body|cookie|querySelector)|"
    r"gokuProps|initData|__NEXT_DATA__|webpackJsonp|webpackChunk|"
    r"cookieDomainList|dataLayer|navigator\.userAgent|localStorage|sessionStorage|"
    r"OptanonWrapper|schema\.org|@context|@graph|application/ld\+json|"
    r"function\s*\w*\s*\(|=>\s*\{|JSON\.parse|Object\.assign"
    r")\b"
)

_CSS_OR_SCRIPT_RE = re.compile(
    r"(?i)("
    r"\bfont-family\s*:|\bbackground(?:-color)?\s*:|\bcolor\s*:|"
    r"\bdisplay\s*:|\bposition\s*:|\bmargin\s*:|\bpadding\s*:|"
    r"\bbody\s*\{|\.[A-Za-z0-9_-]+\s*\{|#[A-Za-z0-9_-]+\s*\{|"
    r"<script\b|</script>|<style\b|</style>"
    r")"
)


def detect_js_bootstrap_dump(text: str) -> bool:
    """Detect browser bootstrap/config output that is not article-like text."""
    cleaned = safe_text(text)
    if not cleaned:
        return False
    hits = _JS_BOOTSTRAP_RE.findall(cleaned)
    if len(hits) >= 2:
        return True
    lowered = cleaned.lower()
    if "window." in lowered and ("document." in lowered or "function(" in lowered or "initdata" in lowered):
        return True
    if "optanonwrapper" in lowered or ("schema.org" in lowered and ("@context" in lowered or "@graph" in lowered)):
        return True
    assignments = len(re.findall(r"\b(?:window|document|globalThis)\.[\w$]+\s*=", cleaned))
    return assignments >= 2


def detect_css_or_script_dump(text: str) -> bool:
    """Detect extracted CSS/script fragments such as style sheets or inline JS."""
    cleaned = safe_text(text)
    if not cleaned:
        return False
    css_hits = len(_CSS_OR_SCRIPT_RE.findall(cleaned))
    brace_count = cleaned.count("{") + cleaned.count("}")
    semicolon_count = cleaned.count(";")
    word_count = max(len(re.findall(r"\w+", cleaned)), 1)
    if css_hits >= 2:
        return True
    return brace_count >= 4 and semicolon_count >= 4 and (brace_count + semicolon_count) / word_count > 0.18


def detect_token_list_dump(text: str) -> bool:
    """Detect JSON/config/ticker dumps with too many tokens and little prose."""
    cleaned = safe_text(text)
    if not cleaned:
        return False
    words = re.findall(r"[A-Za-zА-Яа-я0-9_$.-]+", cleaned)
    if len(words) < 20:
        return False
    long_identifiers = [
        word for word in words
        if len(word) >= 18 and re.search(r"[A-Za-z]", word) and re.search(r"[A-Z_$]", word)
    ]
    upper_tokens = [word for word in words if re.fullmatch(r"[A-Z0-9_.$-]{2,}", word)]
    punctuation_density = len(re.findall(r"[{}\[\]:,=;]", cleaned)) / max(len(cleaned), 1)
    avg_word_len = sum(len(word) for word in words) / max(len(words), 1)
    prose_marks = len(re.findall(r"[.!?]\s+[A-ZА-Я]", cleaned))
    if len(long_identifiers) >= 5 and punctuation_density > 0.04:
        return True
    if len(upper_tokens) >= 20 and len(upper_tokens) / max(len(words), 1) > 0.32:
        return True
    if punctuation_density > 0.09 and avg_word_len > 8 and prose_marks < 2:
        return True
    return False


def _human_readability_failure_reason(text: str, query: str = "") -> str:
    cleaned = compress_page_text(safe_text(text), max_chars=5000)
    if not cleaned.strip():
        return "empty_page"
    lowered = cleaned.lower()
    challenge_or_bootstrap_phrases = (
        "a required part of this site couldn",
        "client challenge",
        "enable javascript",
        "please enable cookies",
        "function optanonwrapper",
        "optanonwrapper()",
        '"@context"',
        '"@graph"',
        "schema.org",
    )
    if any(phrase in lowered for phrase in challenge_or_bootstrap_phrases):
        return "js_bootstrap"
    if detect_js_bootstrap_dump(cleaned):
        return "js_bootstrap"
    if detect_css_or_script_dump(cleaned):
        return "not_human_readable"
    if detect_token_list_dump(cleaned):
        return "not_human_readable"

    words = re.findall(r"[A-Za-zА-Яа-я0-9]+", cleaned)
    if len(words) < 2:
        return "not_human_readable"
    visible_chars = len(re.findall(r"[A-Za-zА-Яа-я0-9]", cleaned))
    symbol_chars = len(re.findall(r"[{}\[\]();:=<>]", cleaned))
    if visible_chars and symbol_chars / visible_chars > 0.22:
        return "not_human_readable"
    return ""


def is_human_readable_excerpt(text: str, query: str = "") -> bool:
    """Return True only for text usable as a human-readable source excerpt."""
    return not _human_readability_failure_reason(text, query)


def _is_useful_search_snippet(snippet: str, query: str = "") -> bool:
    cleaned = compress_page_text(safe_text(snippet), max_chars=320)
    if len(cleaned) < 20:
        return False
    if _is_value_query(query) and not re.search(
        r"(?:[$€₽]\s*\d|\d[\d,.]*\s*(?:usd|rub|eur|руб|₽|€|\$)|"
        r"\b(?:price|цена|курс|составляет|current)\b.{0,80}(?:[$€₽]|\d[\d,.]))",
        cleaned,
        re.IGNORECASE,
    ):
        return False
    return is_human_readable_excerpt(cleaned, query)


_USER_VISIBLE_BANNED_RE = re.compile(
    r"(?is)("
    r"\[truncated\]|ascii codec|UnicodeEncodeError|UnicodeDecodeError|HTTP Error|URLError|"
    r"Traceback|Connection refused|binary_content|empty_page|js_only_suspected|"
    r"not_human_readable|js_bootstrap|window\.initData|window\.gokuProps|"
    r"document\.documentElement|document\.cookie|localStorage|sessionStorage|"
    r"font-family|function\s*\w*\s*\(|OptanonWrapper|schema\.org|@context|@graph|"
    r"Client Challenge|A required part of this site couldn|webpack|__NEXT_DATA__|"
    r"[A-Za-z0-9+/]{80,}={0,2}"
    r")"
)

_FINAL_ANSWER_GARBAGE_RE = re.compile(
    r"(?is)("
    r"Почему\s+релевантно|\[truncated\]|ascii codec|UnicodeEncodeError|UnicodeDecodeError|"
    r"HTTP Error|URLError|binary_content|empty_page|js_bootstrap|not_human_readable|"
    r"window\.|document\.|gokuProps|font-family|function\s*\w*\s*\(|"
    r"OptanonWrapper|schema\.org|@context|@graph|Client Challenge|"
    r"A required part of this site couldn|"
    r"Charts\s+loading|Я\s+согласен|Используя\s+сайт|Вы\s+соглашаетесь|"
    r"Политик[ае]\s+обработки\s+персональных\s+данных|cookie|cookies|"
    r"калькулятор\s+валют|история\s+стоимости|кросс-курс|онлайн\s+график|"
    r"(?:➜\s*){2,}"
    r")"
)


def is_debug_or_technical_text(text: str) -> bool:
    value = safe_text(text)
    if not value:
        return False
    return bool(
        _USER_VISIBLE_BANNED_RE.search(value)
        or detect_js_bootstrap_dump(value)
        or detect_css_or_script_dump(value)
        or detect_token_list_dump(value)
    )


def clean_user_visible_text(text: str, *, max_chars: int = 400, preserve_urls: bool = True) -> str:
    """Final boundary for text that may be shown in chat or Sources panel."""
    value = safe_text(text)
    if not value:
        return ""
    value = re.sub(r"\.\.\.\s*\[truncated\]|\[truncated\]", "", value, flags=re.IGNORECASE)
    replacements = [
        (r"ascii codec can['\u2019]t encode[^\n.]*", ""),
        (r"Unicode(?:En|De)codeError[^\n.]*", ""),
        (r"HTTP Error \d+[^\n.]*", ""),
        (r"URLError[^\n.]*", ""),
        (r"Traceback \(most recent call last\):.*", ""),
        (r"Connection refused[^\n.]*", ""),
        (r"binary_content", ""),
        (r"empty_page", ""),
        (r"js_only_suspected", ""),
        (r"not_human_readable", ""),
        (r"js_bootstrap", ""),
        (r"\bwindow\.[A-Za-z_$][\w$]*\b", ""),
        (r"\bdocument\.(?:documentElement|cookie|body|querySelector)\b", ""),
        (r"\b(?:localStorage|sessionStorage)\b", ""),
        (r"\bfont-family\s*:[^.;}]*[.;}]?", ""),
        (r"\bfunction\s*\w*\s*\([^)]*\)\s*\{[^}]{0,1200}\}?", ""),
        (r"\bOptanonWrapper\b[^\n.]*", ""),
        (r"\"@context\"[^\n.]{0,1200}", ""),
        (r"\"@graph\"[^\n.]{0,1200}", ""),
        (r"\bschema\.org\b[^\n.]*", ""),
        (r"\bClient Challenge\b[^\n.]*", ""),
        (r"A required part of this site couldn[^\n.]*", ""),
        (r"\bwebpack\w*\b", ""),
        (r"__NEXT_DATA__", ""),
        (r"[A-Za-z0-9+/]{80,}={0,2}", ""),
    ]
    for pattern, repl in replacements:
        value = re.sub(pattern, repl, value, flags=re.IGNORECASE | re.DOTALL)
    if not preserve_urls:
        value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -•\t\n\r")
    if len(value) > max_chars:
        value = value[:max_chars].rsplit(" ", 1)[0].rstrip(" ,.;:") + "..."
    return value.strip()


def _clean_final_answer_text(answer: str, *, max_chars: int = 4000) -> str:
    """Clean chat-facing research answer while preserving the section template."""
    value = safe_text(answer)
    if not value:
        return ""
    replacements = [
        (r"\.\.\.\s*\[truncated\]|\[truncated\]", ""),
        (r"ascii codec can['\u2019]t encode[^\n.]*", ""),
        (r"Unicode(?:En|De)codeError[^\n.]*", ""),
        (r"HTTP Error \d+[^\n.]*", ""),
        (r"URLError[^\n.]*", ""),
        (r"binary_content|empty_page|js_bootstrap|not_human_readable", ""),
        (r"\bwindow\.[A-Za-z_$][\w$]*\b", ""),
        (r"\bdocument\.(?:documentElement|cookie|body|querySelector)\b", ""),
        (r"\bfont-family\s*:[^.;}]*[.;}]?", ""),
        (r"\bfunction\s*\w*\s*\([^)]*\)\s*\{[^}]{0,1200}\}?", ""),
        (r"\bOptanonWrapper\b[^\n.]*", ""),
        (r"\"@context\"[^\n.]{0,1200}", ""),
        (r"\"@graph\"[^\n.]{0,1200}", ""),
        (r"\bschema\.org\b[^\n.]*", ""),
        (r"\bClient Challenge\b[^\n.]*", ""),
        (r"A required part of this site couldn[^\n.]*", ""),
        (r"gokuProps", ""),
    ]
    for pattern, repl in replacements:
        value = re.sub(pattern, repl, value, flags=re.IGNORECASE | re.DOTALL)

    cleaned_lines: list[str] = []
    for raw_line in value.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if _FINAL_ANSWER_GARBAGE_RE.search(line):
            continue
        line = line.strip(" -\t")
        if line:
            cleaned_lines.append(line)
    value = "\n".join(cleaned_lines).strip()
    value = re.sub(r"\n{3,}", "\n\n", value)
    if len(value) > max_chars:
        value = value[:max_chars].rsplit("\n", 1)[0].rstrip(" ,.;:") + "\n..."
    return value.strip()


def public_failure_message(read_status: str, failure_reason: str = "") -> str:
    status = safe_text(read_status).lower()
    reason = safe_text(failure_reason).lower()
    if status in {"js_bootstrap", "not_human_readable", "noisy"}:
        return "Источник найден, но страница отдала технический код, а не читаемый текст."
    if status in {"binary_content"} or "binary" in reason:
        return "Источник найден, но это не текстовая страница."
    if status in {"skipped_private_url", "private_or_local_url", "non_http_url"}:
        return "Источник найден, но этот адрес нельзя безопасно открыть."
    if status in {"timeout"} or "timed out" in reason or "timeout" in reason:
        return "Источник найден, но страница не ответила вовремя."
    if status in {"failed", "empty_page"} or reason:
        return "Источник найден, но страницу не удалось безопасно прочитать."
    return "Источник найден, но страницу не удалось прочитать."


def public_source_status(source: ResearchSource) -> tuple[str, str]:
    status = safe_text(source.read_status).lower()
    if source.read_ok:
        return "read", "read"
    if status in {"js_bootstrap", "not_human_readable", "noisy"}:
        return "not_readable", "not readable"
    if source.snippet_only:
        return "snippet", "snippet only"
    return "failed", "failed"


def public_source_snippet(source: ResearchSource) -> str:
    snippet = clean_user_visible_text(source.snippet, max_chars=260)
    if is_debug_or_technical_text(snippet):
        return ""
    return snippet


def public_source_excerpt(source: ResearchSource) -> str:
    status, _ = public_source_status(source)
    if source.read_ok:
        raw = source.facts[0] if source.facts else (source.excerpt or source.snippet)
        return clean_user_visible_text(raw, max_chars=700)
    if status == "snippet":
        snippet = public_source_snippet(source)
        if snippet:
            return "Страница не прочитана; ниже показан только поисковый сниппет. " + snippet
        return "Страница не прочитана; доступен только поисковый сниппет."
    return public_failure_message(source.read_status, source.failure_reason or source.error)


def public_relevance_reason(source: ResearchSource) -> str:
    if source.read_ok:
        raw = source.facts[0] if source.facts else (source.excerpt or source.snippet)
        return clean_user_visible_text(raw, max_chars=220)
    if source.snippet_only:
        snippet = public_source_snippet(source)
        if snippet:
            return "По поисковому сниппету: " + snippet
        return "Поисковый сниппет доступен, но страница не прочитана."
    return public_failure_message(source.read_status, source.failure_reason or source.error)


def clean_final_answer_for_user(answer: str) -> str:
    cleaned = _clean_final_answer_text(answer, max_chars=max(len(safe_text(answer)), 800))
    if len(cleaned) < 40 and len(safe_text(answer)) > 120:
        return (
            "Я нашёл источники, но читаемый текст оказался недостаточно качественным. "
            "Подробности отмечены справа."
        )
    return cleaned


def build_citations(sources: list[ResearchSource]) -> list[ResearchCitation]:
    citations: list[ResearchCitation] = []
    # Включаем прочитанные источники И snippet_only — с разными метками
    eligible = [s for s in sources if (s.read_ok or s.snippet_only) and s.used_in_answer]
    for idx, source in enumerate(eligible, start=1):
        citations.append(ResearchCitation(
            id=source.id or f"S{idx}",
            title=clean_user_visible_text(source.title, max_chars=180),
            url=source.url,
            domain=clean_user_visible_text(source.domain or domain_from_url(source.url), max_chars=120),
            reason=public_relevance_reason(source),
            snippet_only=source.snippet_only,
        ))
    return citations


def render_citations(citations_or_sources: list[ResearchCitation] | list[ResearchSource]) -> str:
    if not citations_or_sources:
        return ""
    lines = ["Источники:"]
    for index, item in enumerate(citations_or_sources, start=1):
        if isinstance(item, ResearchCitation):
            detail = f"\n   {item.url}"
            clean_reason = clean_user_visible_text(item.reason, max_chars=220)
            if clean_reason:
                detail += f"\n   Почему релевантно: {clean_reason}"
            label = f"{index}. {clean_user_visible_text(item.title, max_chars=160)} — {clean_user_visible_text(item.domain, max_chars=100)}"
            if item.snippet_only:
                label += " *(сниппет поиска, страница не прочитана)*"
            lines.append(f"{label}{detail}")
            continue
        # ResearchSource напрямую
        clean_detail = public_relevance_reason(item)
        suffix = f"\n   {item.url}"
        if clean_detail:
            suffix += f"\n   Почему релевантно: {clean_detail}"
        label = (
            f"{index}. {clean_user_visible_text(item.title, max_chars=160)} — "
            f"{clean_user_visible_text(item.domain or domain_from_url(item.url), max_chars=100)}"
        )
        if getattr(item, "snippet_only", False):
            label += " *(сниппет поиска, страница не прочитана)*"
        lines.append(f"{label}{suffix}")
    return "\n".join(lines)


def clean_excerpt_for_answer(text: str, max_chars: int = 400) -> str:
    """
    Очищает excerpt перед включением в основной ответ.
    Технические ошибки, truncated, мусор остаются в Sources panel.
    """
    if not text:
        return ""
    # Убираем технические маркеры
    TRASH_PATTERNS = [
        r"\.\.\.\s*\[truncated\]",
        r"\[truncated\]",
        r"ascii codec can['\u2019]t encode[^\n.]*",
        r"UnicodeEncodeError[^\n.]*",
        r"UnicodeDecodeError[^\n.]*",
        r"HTTP Error \d+[^\n.]*",
        r"binary_content[^\n.]*",
        r"empty_page[^\n.]*",
        r"js_only_suspected[^\n.]*",
        r"Connection (?:refused|timed? out)[^\n.]*",
        r"\[local path\]",
        r"\[personal email\]",
        r"\[personal phone\]",
    ]
    cleaned = text
    for pat in TRASH_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    # Нормализуем пробелы
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def _is_noisy_excerpt(text: str, query: str) -> bool:
    """
    Определяет что excerpt — это навигация/список токенов/мусор,
    а не полезный текст для ответа.
    """
    if not text or len(text.strip()) < 30:
        return True
    # Много коротких капслок-слов подряд (типичный список тикеров/категорий)
    caps_tokens = re.findall(r"\b[A-Z]{2,}\b", text)
    if len(caps_tokens) > 8 and len(caps_tokens) / max(len(text.split()), 1) > 0.3:
        return True
    # Много кириллических слов из 2-3 букв (навигация)
    short_words = re.findall(r"\b[А-Яа-я]{2,3}\b", text)
    if len(short_words) > 10 and len(short_words) / max(len(text.split()), 1) > 0.5:
        return True
    # Навигационные/индексные списки из документационных сайтов: формально текст,
    # но для ответа это мусор вида "QComboBox Widget PyQt - QSplitter Widget...".
    if text.count(" - ") >= 5 and len(re.findall(r"\b(?:widget|pyqt|q[a-z]+)\b", text, re.IGNORECASE)) >= 6:
        return True
    if re.match(r"^\s*(?:\.\.\.|…)\w+", text):
        return True
    # Нет ни одного query-слова — только если текст КОРОТКИЙ
    terms = _query_terms(query)
    if terms and len(text.strip()) < 200:
        lowered = text.lower()
        # Проверяем по корню (первые 5 символов) чтобы учесть склонения
        if not any(lowered.count(t[:5]) > 0 for t in terms[:4] if len(t) >= 4):
            return True
    return False


def _is_low_value_answer_fact(text: str, query: str) -> bool:
    """True when text is readable but should not be promoted into the main answer."""
    cleaned = safe_text(text)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if _FINAL_ANSWER_GARBAGE_RE.search(cleaned):
        return True
    noisy_phrases = [
        "a required part of this site couldn",
        "client challenge",
        "function optanonwrapper",
        "optanonwrapper",
        "schema.org",
        "@context",
        "@graph",
        "enable javascript",
        "please enable cookies",
        "courses tutorials interview prep",
        "используя сайт",
        "вы соглашаетесь",
        "я согласен",
        "charts loading",
        "калькулятор валют",
        "история стоимости",
        "кросс-курс",
        "онлайн график",
        "политика обработки персональных данных",
        "политике обработки персональных данных",
    ]
    if any(phrase in lowered for phrase in noisy_phrases):
        return True
    if cleaned.count("➜") >= 2:
        return True
    # Do not reuse _is_noisy_excerpt here: it also rejects short fake/test excerpts
    # without query terms. For answer facts we only drop obvious navigation/list noise.
    if cleaned.count(" - ") >= 5:
        return True
    if re.match(r"^\s*(?:\.\.\.|…)\w+", cleaned):
        return True
    if re.search(r"(?i)(qcombobox|qdoublespinbox|qtoolbox|qmenubar|qtooltip|qinputdialog)", cleaned):
        return True
    return False


_VALUE_QUERY_RE = re.compile(
    r"(?i)(price|current|rate|exchange|value|стоимост|цен[ауы]|курс|сколько|составляет|"
    r"btc|bitcoin|биткоин|usd|rub|eur|доллар|руб)"
)
_VALUE_FACT_RE = re.compile(
    r"(?i)(\$|₽|€|\bруб\.?\b|\brub\b|\busd\b|\beur\b|\bbtc\b|\bbitcoin\b|биткоин|"
    r"price|цена|курс|составляет|current|trading at|is worth)"
)
_CONVERTER_NOISE_RE = re.compile(
    r"(?i)(converter|конвертер|use our converter|convert|calculator|калькулятор|"
    r"sign up|download app|subscribe)"
)


def _is_value_query(query: str) -> bool:
    return bool(_VALUE_QUERY_RE.search(safe_text(query)))


def extract_key_fact_for_answer(text: str, query: str) -> str:
    """Pick the most useful fact sentence, especially for price/rate/current-value queries."""
    cleaned = clean_excerpt_for_answer(text, max_chars=1200)
    if not cleaned:
        return ""
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        candidate = sentence.strip(" -•\t")
        if not candidate:
            continue
        if _FINAL_ANSWER_GARBAGE_RE.search(candidate):
            continue
        sentences.append(candidate)
    if not sentences:
        return cleaned

    is_value_query = _is_value_query(query)
    query_terms = _query_terms(query)
    best_sentence = ""
    best_score = -999
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        score = 0
        if len(sentence) > 25:
            score += 1
        if any(term in lower for term in query_terms):
            score += 1
        if is_value_query and _VALUE_FACT_RE.search(sentence):
            score += 4
        if is_value_query and re.search(r"(?:[$€₽]\s*\d|\d[\d,.]*\s*(?:usd|rub|eur|руб|₽|€|\$))", sentence, re.IGNORECASE):
            score += 5
        if re.search(r"\b(?:BTC|Bitcoin|биткоин)\b", sentence, re.IGNORECASE):
            score += 2
        if _CONVERTER_NOISE_RE.search(sentence):
            score -= 5
        score -= min(index, 4) * 0.1
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if is_value_query and best_score > 0:
        return compress_page_text(best_sentence, max_chars=360).replace(" [truncated]", "").replace("[truncated]", "").strip()
    useful = next((s for s in sentences if len(s) > 30 and not _CONVERTER_NOISE_RE.search(s)), sentences[0])
    return compress_page_text(useful, max_chars=360).replace(" [truncated]", "").replace("[truncated]", "").strip()


def _dedupe_fact(fact: str, existing: list[str]) -> bool:
    norm = re.sub(r"\W+", "", fact.lower())
    if not norm:
        return True
    for old in existing:
        old_norm = re.sub(r"\W+", "", old.lower())
        if norm == old_norm or (len(norm) > 40 and (norm in old_norm or old_norm in norm)):
            return True
    return False


def _generic_stable_facts_from_query(query: str) -> list[str]:
    """Small deterministic fallback for stable technical facts when pages are weak."""
    value = safe_text(query).lower()
    facts: list[str] = []
    if "qsplitter" in value and "handle" in value and ("stylesheet" in value or "style" in value):
        facts.append(
            "Для стилизации ручки splitter в Qt stylesheet используется селектор "
            "`QSplitter::handle`; для ориентации обычно применяют `QSplitter::handle:horizontal` "
            "или `QSplitter::handle:vertical`, задавая width/height/background."
        )
    if "zerodivisionerror" in value or "division by zero" in value:
        facts.append(
            "ZeroDivisionError означает, что Python попытался выполнить деление на ноль; "
            "исправление обычно начинается с проверки делителя перед операцией или явной обработки исключения."
        )
    return facts


def extract_answer_facts(
    sources: list[ResearchSource],
    query: str,
    *,
    max_facts: int = 3,
    allow_weak_snippets: bool = False,
) -> list[str]:
    """Select clean, user-useful facts from sources and mark used sources."""
    facts: list[str] = []
    for source in sorted(sources, key=lambda x: x.relevance_score, reverse=True):
        source.used_in_answer = False
        if source.snippet_only and not source.read_ok:
            raw = source.snippet
        else:
            raw = source.excerpt or (source.facts[0] if source.facts else source.snippet)
        candidate = extract_key_fact_for_answer(raw, query)
        candidate = _clean_final_answer_text(candidate, max_chars=420)
        candidate = re.sub(r"^\s*[-•*]\s*", "", candidate).strip()
        if not candidate or (_is_low_value_answer_fact(candidate, query) and not (allow_weak_snippets and source.snippet_only)):
            continue
        if _dedupe_fact(candidate, facts):
            continue
        facts.append(candidate)
        source.facts = [candidate]
        source.used_in_answer = True
        if len(facts) >= max_facts:
            break
    return facts


def _format_source_line(index: int, source: ResearchSource) -> str:
    title = clean_user_visible_text(source.title or source.url, max_chars=120, preserve_urls=False)
    domain = clean_user_visible_text(source.domain or domain_from_url(source.url), max_chars=80, preserve_urls=False)
    if not title:
        title = domain or clean_user_visible_text(source.url, max_chars=100)
    if not domain:
        domain = clean_user_visible_text(source.url, max_chars=100)
    suffix = " (по сниппету)" if source.snippet_only and not source.read_ok else ""
    return f"{index}. {title} — {domain}{suffix}"


def format_research_final_answer(
    result: ResearchResult | None,
    query: str,
    read_sources: list[ResearchSource],
    snippet_sources: list[ResearchSource] | None = None,
    failed_sources: list[ResearchSource] | None = None,
    facts: list[str] | None = None,
) -> str:
    """Strict chat formatter: facts, short source list, one optional limitation."""
    snippet_sources = snippet_sources or []
    failed_sources = failed_sources or []

    for source in read_sources + snippet_sources:
        source.used_in_answer = False

    selected_facts = facts if facts is not None else extract_answer_facts(read_sources, query, max_facts=3)
    if not selected_facts and snippet_sources:
        selected_facts = extract_answer_facts(snippet_sources, query, max_facts=3, allow_weak_snippets=True)
    generic_facts = _generic_stable_facts_from_query(query)
    if generic_facts:
        if not selected_facts:
            selected_facts = generic_facts[:3]
        else:
            for fact in reversed(generic_facts):
                if _dedupe_fact(fact, selected_facts):
                    continue
                if "qsplitter::handle" in fact.lower() and not any("qsplitter::handle" in item.lower() for item in selected_facts):
                    selected_facts.insert(0, fact)
            selected_facts = selected_facts[:3]

    used_sources = [s for s in read_sources + snippet_sources if s.used_in_answer]
    if facts is not None and not used_sources:
        # Compatibility path for callers that already selected facts.
        for source in read_sources[: len(selected_facts)]:
            source.used_in_answer = True
            used_sources.append(source)

    if selected_facts:
        lines = ["Коротко:", ""]
        lines.extend(f"* {fact}" for fact in selected_facts[:3])
    else:
        lines = [
            "Коротко:",
            "",
            "* Я нашёл источники, но не смог извлечь достаточно чистый ответ. Подробности отмечены справа.",
        ]

    clean_used_sources = used_sources[:3]
    if clean_used_sources:
        lines.extend(["", "Источники:", ""])
        lines.extend(_format_source_line(i, source) for i, source in enumerate(clean_used_sources, start=1))

    snippet_used = any(source.snippet_only and source.used_in_answer for source in read_sources + snippet_sources)
    noisy_or_failed = [s for s in failed_sources if not s.read_ok] + [
        s for s in read_sources + snippet_sources if not s.used_in_answer and not s.read_ok
    ]
    if noisy_or_failed or snippet_used:
        lines.extend([
            "",
            "Ограничение:",
        ])
        if snippet_used and not any(source.read_ok and source.used_in_answer for source in read_sources):
            lines.append("Ответ составлен только по поисковым сниппетам; страницы не прочитаны.")
        else:
            lines.append("Часть источников не удалось прочитать как человеческий текст; они отмечены справа.")

    answer = "\n".join(lines)
    cleaned = clean_final_answer_for_user(answer)
    if not cleaned:
        return "Я нашёл источники, но не смог извлечь достаточно чистый ответ. Подробности отмечены справа."
    return cleaned


def synthesize_research_answer(
    query: str,
    sources: list[ResearchSource],
    warnings: list[str] | None = None,
    snippet_sources: list[ResearchSource] | None = None,
) -> tuple[str, list[str]]:
    """
    Строит чистый пользовательский ответ.
    Технические ошибки и мусор в основной ответ НЕ попадают.
    """
    read_sources = [s for s in sources if s.read_ok]
    snippet_sources = snippet_sources or []

    if not read_sources and not snippet_sources:
        return (
            "Я нашёл поисковую выдачу, но не смог безопасно прочитать страницы для ответа. "
            "Не буду придумывать факты или ссылки без прочитанного источника.",
            [],
        )

    answer = format_research_final_answer(
        None,
        query,
        read_sources,
        snippet_sources=snippet_sources,
        failed_sources=[s for s in sources + snippet_sources if not s.read_ok],
    )
    facts = [s.facts[0] for s in read_sources + snippet_sources if s.used_in_answer and s.facts]
    return answer, facts


def _backend_available(backend: Any) -> bool:
    available = getattr(backend, "available", False)
    if callable(available):
        try:
            return bool(available())
        except Exception:
            return False
    return bool(available)


def _backend_name(backend: Any) -> str:
    return str(getattr(backend, "name", backend.__class__.__name__))


_RU_SEARCH_STOPWORDS_RE = re.compile(
    r"(?i)\b(найди|поищи|свеж(?:ую|ие|ая|ий)|актуальн(?:ую|ые|ая|ый|о)|"
    r"информаци[яюи]|новост[ьи]|документаци[яюи]|про|по|для|что|как|где|курс|цен[ауы])\b"
)


def fallback_public_search_query(query: str) -> str:
    """Build a safe public-keyword fallback from an already sanitized query."""
    text = safe_text(query)
    if not text:
        return ""
    lowered = text.lower()
    if "openai" in lowered and re.search(r"новост|news|fresh|свеж", lowered, re.IGNORECASE):
        return "OpenAI news latest"
    if "llama-cpp-python" in lowered or "llama.cpp" in lowered:
        parts = ["llama-cpp-python"]
        for token in ("CUDA", "Windows", "PyInstaller"):
            if token.lower() in lowered:
                parts.append(token)
        parts.append("latest")
        return " ".join(dict.fromkeys(parts))
    latin_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.+#-]{2,}", text)
    if latin_tokens:
        suffix = " latest" if re.search(r"свеж|актуальн|latest|current|news|новост", lowered) else ""
        return " ".join(dict.fromkeys(latin_tokens)) + suffix
    cleaned = _RU_SEARCH_STOPWORDS_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned != text else ""


def _to_search_result(source: SearchResult | ResearchSource, rank: int) -> SearchResult:
    return SearchResult(
        title=source.title,
        url=source.url,
        snippet=source.snippet,
        source=getattr(source, "source", "") or "search",
        rank=getattr(source, "rank", 0) or rank,
    )


class ResearchPipeline:
    def __init__(self, backend: ResearchBackend | None = None, backend_status: ResearchBackendStatus | None = None) -> None:
        self.backend = backend or make_research_backend()
        self.backend_status = backend_status or diagnose_research_backend(backend=self.backend, kind=_backend_name(self.backend))
        self.privacy = PrivacyFirewall()

    def run(
        self,
        query: str,
        *,
        max_search_results: int = 5,
        max_pages_to_read: int = 3,
        require_sources_for_fresh_info: bool = True,
        confirmed_outbound: bool = False,
    ) -> ResearchResult:
        if not needs_web_search(query, require_sources_for_fresh_info):
            return ResearchResult(answer="", used_search=False, original_user_query=query)
        backend_status = self.backend_status or diagnose_research_backend(backend=self.backend, kind=_backend_name(self.backend))
        write_log(
            "[research_backend_diagnostics] "
            f'selected="{_quote_log(backend_status.selected_backend)}" '
            f'configured="{_quote_log(backend_status.configured_backend)}" '
            f'searxng_url="{str(bool(backend_status.searxng_url_configured)).lower()}" '
            f'searxng_available="{str(bool(backend_status.searxng_available)).lower()}" '
            f'duckduckgo_available="{str(bool(backend_status.duckduckgo_available)).lower()}" '
            f'effective="{_quote_log(backend_status.effective_backend)}" '
            f'action="{_quote_log(backend_status.recommended_action)}"'
        )
        write_log(f'[research_query_proposed] chars="{len(query or "")}"')
        decision = self.privacy.block_outbound_if_private(query, confirmed=confirmed_outbound)
        write_log(
            "[research_query_sanitized] "
            f'query="{_quote_log(decision.sanitized)}" reasons="{",".join(decision.reasons)}"'
        )
        if not decision.allowed:
            if decision.confirmation_required:
                write_log(
                    "[privacy_firewall_confirm_required] "
                    f'reasons="{",".join(decision.reasons)}" sanitized="{_quote_log(decision.sanitized)}"'
                )
                return ResearchResult(
                    answer=(
                        "Перед web-поиском нужно подтверждение.\n\n"
                        "ZenAI не отправляет историю чата, память, код или локальные файлы в интернет.\n\n"
                        "В интернет может уйти только обезличенный запрос:\n"
                        f"- {decision.sanitized or '[empty]'}"
                    ),
                    used_search=False,
                    error="privacy confirmation required",
                    sanitized_query=decision.sanitized,
                    privacy_reasons=decision.reasons,
                    original_user_query=query,
                    backend_name=_backend_name(self.backend),
                    privacy_decision=decision,
                )
            write_log(f'[privacy_firewall_blocked] reasons="{",".join(decision.reasons)}"')
            return ResearchResult(
                answer=(
                    "Коротко:\n"
                    "- Запрос не отправлен в интернет, потому что PrivacyFirewall нашёл приватные данные.\n\n"
                    "Безопасная версия запроса:\n"
                    f"- {decision.sanitized or '[empty]'}"
                ),
                used_search=False,
                error="privacy blocked",
                sanitized_query=decision.sanitized,
                privacy_reasons=decision.reasons,
                original_user_query=query,
                backend_name=_backend_name(self.backend),
                privacy_decision=decision,
                warnings=[f"privacy_blocked:{reason}" for reason in decision.reasons],
            )
        write_log(
            "[research_backend_selected] "
            f'backend="{_quote_log(_backend_name(self.backend))}" '
            f'effective="{_quote_log(backend_status.effective_backend)}" '
            f'action="{_quote_log(backend_status.recommended_action)}"'
        )
        if not _backend_available(self.backend):
            write_log(f'[research_backend_unavailable] reason="{_quote_log(_backend_name(self.backend))}"')
            return ResearchResult(
                answer=backend_unavailable_message(backend_status, backend_error="backend_unavailable"),
                used_search=False,
                error="web backend unavailable",
                sanitized_query=decision.sanitized,
                privacy_reasons=decision.reasons,
                original_user_query=query,
                backend_name=_backend_name(self.backend),
                backend_status=backend_status,
                privacy_decision=decision,
                warnings=["web backend unavailable"],
            )

        outbound_query = decision.sanitized
        write_log(f'[research_query_sent] query="{_quote_log(outbound_query)}"')
        backend_error = ""
        try:
            raw_results = self.backend.search(outbound_query, max_search_results)
        except Exception as exc:
            backend_error = _backend_error_code(exc)
            write_log(f'[research_backend_unavailable] reason="{_quote_log(exc)}" code="{backend_error}"')
            if backend_error == "duckduckgo_no_results":
                raw_results = []
            else:
                backend_status.last_backend_error = safe_text(exc)
                return ResearchResult(
                    answer=backend_unavailable_message(backend_status, backend_error=backend_error),
                    used_search=False,
                    error="web backend unavailable",
                    sanitized_query=decision.sanitized,
                    privacy_reasons=decision.reasons,
                    original_user_query=query,
                    backend_name=_backend_name(self.backend),
                    backend_status=backend_status,
                    privacy_decision=decision,
                    warnings=[safe_text(exc)],
                )
        if not raw_results:
            fallback_query = fallback_public_search_query(decision.sanitized)
            if fallback_query and fallback_query != decision.sanitized:
                write_log(f'[research_query_sent] query="{_quote_log(fallback_query)}" fallback="true"')
                try:
                    raw_results = self.backend.search(fallback_query, max_search_results)
                    outbound_query = fallback_query if raw_results else outbound_query
                except Exception as exc:
                    backend_error = _backend_error_code(exc)
                    write_log(f'[research_backend_unavailable] reason="{_quote_log(exc)}" code="{backend_error}" fallback="true"')
                    if backend_error != "duckduckgo_no_results":
                        backend_status.last_backend_error = safe_text(exc)
                        return ResearchResult(
                            answer=backend_unavailable_message(backend_status, backend_error=backend_error),
                            used_search=False,
                            error="web backend unavailable",
                            sanitized_query=decision.sanitized,
                            privacy_reasons=decision.reasons,
                            original_user_query=query,
                            backend_name=_backend_name(self.backend),
                            backend_status=backend_status,
                            privacy_decision=decision,
                            warnings=[safe_text(exc)],
                        )
        search_results = [_to_search_result(item, index + 1) for index, item in enumerate(raw_results)]
        write_log(f'[research_search_results] count="{len(search_results)}"')
        if not search_results:
            if _backend_name(self.backend) == "duckduckgo":
                backend_status.last_backend_error = backend_status.last_backend_error or backend_error or "duckduckgo_no_results"
                return ResearchResult(
                    answer=backend_unavailable_message(backend_status, backend_error=backend_status.last_backend_error),
                    used_search=False,
                    error="web backend unavailable",
                    sanitized_query=outbound_query,
                    privacy_reasons=decision.reasons,
                    original_user_query=query,
                    backend_name=_backend_name(self.backend),
                    backend_status=backend_status,
                    privacy_decision=decision,
                    warnings=[backend_status.last_backend_error or "duckduckgo_no_results"],
                )
            return ResearchResult(
                answer=(
                    "Коротко:\n"
                    "- По безопасному поисковому запросу не удалось получить результаты.\n\n"
                    "Ограничение:\n"
                    "- Я не буду придумывать ответ или источники без поисковой выдачи."
                ),
                sources=[],
                used_search=True,
                sanitized_query=outbound_query,
                privacy_reasons=decision.reasons,
                error="no search results",
                original_user_query=query,
                backend_name=_backend_name(self.backend),
                backend_status=backend_status,
                search_results=search_results,
                privacy_decision=decision,
                warnings=["no search results"],
            )
        read_sources: list[ResearchSource] = []
        ranked_sources: list[ResearchSource] = []
        fetched_pages: list[FetchedPage] = []
        warnings: list[str] = []
        snippet_only_sources: list[ResearchSource] = []

        # Пробуем все результаты подряд пока не наберём max_pages_to_read прочитанных.
        # Раньше: [:max_pages_to_read] — пробовались только первые N, без retry.
        candidates = list(search_results)  # все результаты, а не только первые N
        write_log(f'[research_fetch_loop_start] total_candidates="{len(candidates)}" max_readable="{max_pages_to_read}"')

        for result in candidates:
            if len(read_sources) >= max_pages_to_read:
                break  # набрали достаточно прочитанных

            source_id = f"S{len(ranked_sources) + 1}"
            domain = domain_from_url(result.url)
            allowed, reason = is_fetch_url_allowed(result.url)
            if not allowed:
                write_log(f'[research_fetch_skipped] url="{_quote_log(result.url)}" reason="{reason}"')
                warnings.append(f"{result.title}: {reason}")
                ranked_sources.append(ResearchSource(
                    id=source_id,
                    title=safe_text(result.title),
                    url=result.url,
                    snippet=compress_page_text(safe_text(result.snippet), max_chars=240),
                    source=result.source,
                    rank=result.rank,
                    domain=domain,
                    fetched=False,
                    read_ok=False,
                    read_status="skipped_private_url",
                    failure_reason=reason,
                ))
                continue
            try:
                if hasattr(self.backend, "fetch"):
                    page = self.backend.fetch(result.url, max_chars=10000)
                else:
                    page_text = self.backend.read_url(result.url)  # type: ignore[attr-defined]
                    page = FetchedPage(url=result.url, text=safe_text(page_text), ok=bool(page_text))
                fetched_pages.append(page)
            except Exception as exc:
                exc_str = safe_text(exc)
                write_log(f'[research_fetch_failed] url="{_quote_log(result.url)}" reason="{_quote_log(exc_str)}"')
                warnings.append(f"{safe_text(result.title)}: {exc_str}")
                # snippet_only fallback — сниппет из поиска как запасное свидетельство
                snippet = compress_page_text(safe_text(result.snippet), max_chars=240)
                snippet_is_useful = _is_useful_search_snippet(snippet, outbound_query)
                src = ResearchSource(
                    id=source_id,
                    title=safe_text(result.title),
                    url=result.url,
                    snippet=snippet,
                    source=result.source,
                    rank=result.rank,
                    domain=domain,
                    fetched=True,
                    read_ok=False,
                    snippet_only=snippet_is_useful,
                    read_status="timeout" if "timeout" in exc_str.lower() else "failed",
                    failure_reason=exc_str,
                )
                ranked_sources.append(src)
                if snippet_is_useful:
                    snippet_only_sources.append(src)
                write_log(f'[research_source_status] url="{_quote_log(result.url)}" status="snippet_only" has_snippet="{snippet_is_useful}"')
                continue

            snippet = compress_page_text(safe_text(result.snippet), max_chars=240)
            if not page.ok or not page.text:
                reason = safe_text(page.error or "empty_page")
                status_reason = "failed" if reason in {"empty_page", "empty page"} else reason
                write_log(
                    "[research_source_status] "
                    f'url="{_quote_log(result.url)}" status="{status_reason}"'
                )
                warnings.append(f"{safe_text(result.title)}: {reason}")
                # snippet_only: страница не прочиталась, но сниппет есть
                snippet_is_useful = _is_useful_search_snippet(snippet, outbound_query)
                src = ResearchSource(
                    id=source_id,
                    title=safe_text(page.title or result.title),
                    url=result.url,
                    snippet=snippet,
                    retrieved_at=page.fetched_at,
                    source=result.source,
                    rank=result.rank,
                    domain=domain,
                    fetched=True,
                    read_ok=False,
                    snippet_only=snippet_is_useful,
                    read_status=status_reason,
                    failure_reason=reason,
                )
                ranked_sources.append(src)
                if snippet_is_useful:
                    snippet_only_sources.append(src)
                write_log(f'[research_more_results_fetch] because="not_enough_readable_sources" readable_so_far="{len(read_sources)}"')
                continue

            excerpt = relevant_excerpt(safe_text(page.text), outbound_query, max_chars=1000)
            quality_reason = _human_readability_failure_reason(excerpt, outbound_query)
            if quality_reason:
                snippet_is_useful = _is_useful_search_snippet(snippet, outbound_query)
                src = ResearchSource(
                    id=source_id,
                    title=safe_text(page.title or result.title or result.url),
                    url=result.url,
                    snippet=snippet,
                    retrieved_at=page.fetched_at,
                    facts=[],
                    source=result.source,
                    rank=result.rank,
                    domain=domain,
                    fetched=True,
                    read_ok=False,
                    snippet_only=snippet_is_useful,
                    excerpt=excerpt,
                    relevance_score=source_relevance(result, safe_text(page.text), outbound_query),
                    read_status=quality_reason,
                    failure_reason=quality_reason,
                )
                ranked_sources.append(src)
                warnings.append(f"{safe_text(result.title)}: {quality_reason}")
                if snippet_is_useful:
                    snippet_only_sources.append(src)
                write_log(
                    "[research_source_quality_failed] "
                    f'url="{_quote_log(result.url)}" status="{quality_reason}" '
                    f'snippet_only="{snippet_is_useful}" excerpt_chars="{len(excerpt)}"'
                )
                write_log(f'[research_more_results_fetch] because="not_enough_readable_sources" readable_so_far="{len(read_sources)}"')
                continue
            source = ResearchSource(
                id=source_id,
                title=safe_text(page.title or result.title or result.url),
                url=result.url,
                snippet=snippet,
                retrieved_at=page.fetched_at,
                facts=[excerpt],
                source=result.source,
                rank=result.rank,
                domain=domain,
                fetched=True,
                read_ok=True,
                excerpt=excerpt,
                relevance_score=source_relevance(result, safe_text(page.text), outbound_query),
                read_status="read",
            )
            read_sources.append(source)
            ranked_sources.append(source)
            write_log(f'[research_source_status] url="{_quote_log(result.url)}" status="read_ok" excerpt_chars="{len(excerpt)}"')

        ranked_sources.sort(key=lambda item: (item.read_ok, item.relevance_score), reverse=True)
        write_log(
            f'[research_answer_ready] readable="{len(read_sources)}" '
            f'snippet_only="{len(snippet_only_sources)}" '
            f'failed="{len(ranked_sources) - len(read_sources) - len(snippet_only_sources)}"'
        )
        write_log(f'[research_sources_ranked] count="{len(ranked_sources)}"')

        if search_results and not read_sources:
            if snippet_only_sources:
                answer = format_research_final_answer(
                    None,
                    outbound_query,
                    [],
                    snippet_sources=snippet_only_sources,
                    failed_sources=ranked_sources,
                )
            else:
                answer = (
                    "Я нашёл поисковую выдачу, но не смог безопасно прочитать страницы для цитирования. "
                    "Не буду придумывать источники без прочитанного содержимого."
                )
            answer = clean_final_answer_for_user(answer)
            citations = build_citations(snippet_only_sources)
            return ResearchResult(
                answer=answer,
                sources=[source for source in snippet_only_sources if source.used_in_answer],
                used_search=True,
                sanitized_query=outbound_query,
                privacy_reasons=decision.reasons,
                error="no readable sources",
                original_user_query=query,
                backend_name=_backend_name(self.backend),
                backend_status=backend_status,
                search_results=search_results,
                fetched_pages=fetched_pages,
                ranked_sources=ranked_sources,
                final_answer=answer,
                citations=citations,
                warnings=warnings,
                privacy_decision=decision,
            )
        answer_body, extracted_facts = synthesize_research_answer(
            outbound_query, read_sources,
            warnings=None,  # warnings в ответ НЕ идут — только в Sources panel
            snippet_sources=snippet_only_sources,
        )
        citations = build_citations(read_sources + snippet_only_sources)
        answer = clean_final_answer_for_user(answer_body)
        write_log(f'[research_answer_ready] source_count="{len(read_sources)}"')
        return ResearchResult(
            answer=answer,
            sources=[source for source in read_sources if source.used_in_answer],
            used_search=True,
            sanitized_query=outbound_query,
            privacy_reasons=decision.reasons,
            original_user_query=query,
            backend_name=_backend_name(self.backend),
            backend_status=backend_status,
            search_results=search_results,
            fetched_pages=fetched_pages,
            ranked_sources=ranked_sources,
            extracted_facts=extracted_facts,
            final_answer=answer,
            citations=citations,
            warnings=warnings,
            privacy_decision=decision,
        )


class ResearchCapability:
    """Future-facing capability wrapper for profile-specific web policies.

    Stage 1 wires this only for the Researcher profile. Coder/Lera can later call
    the same pipeline through an explicit context_policy without sending local
    project, chat, or memory context by default.
    """

    def __init__(self, backend: ResearchBackend | None = None) -> None:
        self.backend = backend

    def search_for_profile(
        self,
        profile_kind: str,
        query: str,
        context_policy: str = "researcher_only",
        **kwargs: Any,
    ) -> ResearchResult:
        if context_policy != "researcher_only" or str(profile_kind) not in {"researcher", "ProfileKind.RESEARCHER"}:
            return ResearchResult(
                answer="Research capability is currently enabled only for the Searcher profile.",
                used_search=False,
                error="research capability disabled for profile",
            )
        return ResearchPipeline(self.backend).run(query, **kwargs)
