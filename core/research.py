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
    excerpt: str = ""
    relevance_score: float = 0.0
    used_in_answer: bool = False
    read_status: str = "snippet"
    error: str = ""
    failure_reason: str = ""

    def citation(self, index: int) -> str:
        return f"[{index}] {self.title} — {self.url}"

    def to_card(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "domain": self.domain or domain_from_url(self.url),
            "snippet": self.snippet,
            "fetched": self.fetched,
            "read_ok": self.read_ok,
            "excerpt": self.excerpt or (self.facts[0] if self.facts else ""),
            "relevance_score": self.relevance_score,
            "used_in_answer": self.used_in_answer,
            "status": self.read_status,
            "failure_reason": self.failure_reason or self.error,
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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.final_answer:
            self.final_answer = self.answer


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
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "nav", "footer", "header", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag == "title":
            self._title_active = True
        elif tag in {"p", "br", "li", "div", "section", "article", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "nav", "footer", "header", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._title_active = False
        elif tag in {"p", "li", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._title_active:
            self.title_parts.append(text)
        if self._skip_depth or self._title_active:
            return
        self.text_parts.append(text)

    def text(self, max_chars: int) -> str:
        return compress_page_text(" ".join(self.text_parts), max_chars=max_chars)

    def title(self) -> str:
        return compress_page_text(" ".join(self.title_parts), max_chars=200)


def extract_html_text(html: str, max_chars: int = 10000) -> tuple[str, str]:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html or "")
    except Exception:
        plain = re.sub(r"<[^>]+>", " ", html or "")
        return "", compress_page_text(plain, max_chars=max_chars)
    return parser.title(), parser.text(max_chars=max_chars)


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
    user_agent = "ZenAI-Research/1.0 (+https://local)"

    def fetch(self, url: str, max_chars: int = 10000) -> FetchedPage:
        allowed, reason = is_fetch_url_allowed(url)
        if not allowed:
            write_log(f'[research_fetch_skipped] url="{_quote_log(url)}" reason="{reason}"')
            return FetchedPage(url=url, ok=False, error=reason)
        write_log(f'[research_fetch_start] url="{_quote_log(url)}"')
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = int(getattr(response, "status", 0) or 0)
                content_type = response.headers.get("content-type", "")
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
                raw = response.read(max_chars * 4)
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            write_log(f'[research_fetch_skipped] url="{_quote_log(url)}" reason="{_quote_log(exc)}"')
            return FetchedPage(url=url, ok=False, error=str(exc))
        encoding = "utf-8"
        charset_match = re.search(r"charset=([\w.-]+)", content_type, re.IGNORECASE)
        if charset_match:
            encoding = charset_match.group(1)
        decoded = raw.decode(encoding, errors="replace")
        if "html" in content_type.lower():
            title, text = extract_html_text(decoded, max_chars=max_chars)
        else:
            title, text = "", compress_page_text(decoded, max_chars=max_chars)
        write_log(
            "[research_fetch_done] "
            f'url="{_quote_log(url)}" status="{status}" chars="{len(text)}"'
        )
        return FetchedPage(
            url=url,
            title=title,
            text=text,
            status_code=status,
            content_type=content_type,
            ok=bool(text),
            error="" if text else "empty_page",
        )


class DuckDuckGoBackend(UrlFetchMixin):
    name = "duckduckgo"
    search_url = "https://html.duckduckgo.com/html/"

    def available(self) -> bool:
        return True

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        request_url = self.search_url + "?" + urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(request_url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                html = response.read(600_000).decode("utf-8", errors="replace")
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            raise RuntimeError(f"DuckDuckGo search failed: {exc}") from exc
        return _parse_duckduckgo_html(html, max_results=max_results)


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


class SearxNGBackend(UrlFetchMixin):
    name = "searxng"

    def __init__(self, base_url: str) -> None:
        self.base_url = (base_url or "").rstrip("/")

    def available(self) -> bool:
        return bool(self.base_url)

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        if not self.base_url:
            raise RuntimeError("SearxNG URL is not configured.")
        parsed_base = urllib.parse.urlparse(self.base_url)
        if _is_private_host(parsed_base.hostname or "") and parsed_base.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise RuntimeError("SearxNG URL points to a private network host.")
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


def make_research_backend(kind: str = "auto", searxng_url: str = "") -> ResearchBackend:
    selected = (kind or os.environ.get("ZENAI_RESEARCH_BACKEND") or "auto").strip().lower()
    configured_searxng = (searxng_url or os.environ.get("ZENAI_SEARXNG_URL") or "").strip()
    if selected == "unavailable":
        return UnavailableWebBackend()
    if selected == "searxng":
        return SearxNGBackend(configured_searxng) if configured_searxng else UnavailableWebBackend()
    if selected == "duckduckgo":
        return DuckDuckGoBackend()
    if configured_searxng:
        return SearxNGBackend(configured_searxng)
    return DuckDuckGoBackend()


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


def build_citations(sources: list[ResearchSource]) -> list[ResearchCitation]:
    citations: list[ResearchCitation] = []
    for idx, source in enumerate([s for s in sources if s.read_ok and s.used_in_answer], start=1):
        citations.append(ResearchCitation(
            id=source.id or f"S{idx}",
            title=source.title,
            url=source.url,
            domain=source.domain or domain_from_url(source.url),
            reason=compress_page_text(source.snippet or source.excerpt, max_chars=180),
            snippet_only=False,
        ))
    return citations


def render_citations(citations_or_sources: list[ResearchCitation] | list[ResearchSource]) -> str:
    if not citations_or_sources:
        return ""
    lines = ["Источники:"]
    for index, item in enumerate(citations_or_sources, start=1):
        if isinstance(item, ResearchCitation):
            detail = f"\n   {item.url}"
            if item.reason:
                detail += f"\n   Почему релевантно: {item.reason}"
            label = f"{index}. {item.title} — {item.domain}"
            if item.snippet_only:
                label += " (по сниппету поиска)"
            lines.append(f"{label}{detail}")
            continue
        detail = item.snippet or item.excerpt or (item.facts[0] if item.facts else "")
        suffix = f"\n   {item.url}"
        if detail:
            suffix += f"\n   Почему релевантно: {compress_page_text(detail, max_chars=180)}"
        lines.append(f"{index}. {item.title} — {item.domain or domain_from_url(item.url)}{suffix}")
    return "\n".join(lines)


def synthesize_research_answer(query: str, sources: list[ResearchSource], warnings: list[str] | None = None) -> tuple[str, list[str]]:
    read_sources = [source for source in sources if source.read_ok]
    if not read_sources:
        return (
            "Я нашёл поисковую выдачу, но не смог безопасно прочитать страницы для ответа. "
            "Не буду придумывать факты или ссылки без прочитанного источника.",
            [],
        )
    read_sources.sort(key=lambda item: item.relevance_score, reverse=True)
    facts: list[str] = []
    for source in read_sources[:4]:
        excerpt = source.excerpt or (source.facts[0] if source.facts else source.snippet)
        first_sentence = re.split(r"(?<=[.!?。])\s+", excerpt.strip(), maxsplit=1)[0]
        facts.append(f"[{source.id}] {compress_page_text(first_sentence or excerpt, max_chars=320)}")
        source.used_in_answer = True
    warning_text = ""
    if warnings:
        warning_text = "\n\nОграничения: " + "; ".join(warnings[:3])
    answer = (
        f"По запросу `{query}` я прочитал {len(read_sources)} источник(а) и собрал краткий вывод.\n\n"
        "Коротко:\n"
        + "\n".join(f"- {fact}" for fact in facts)
        + warning_text
    )
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


def _to_search_result(source: SearchResult | ResearchSource, rank: int) -> SearchResult:
    return SearchResult(
        title=source.title,
        url=source.url,
        snippet=source.snippet,
        source=getattr(source, "source", "") or "search",
        rank=getattr(source, "rank", 0) or rank,
    )


class ResearchPipeline:
    def __init__(self, backend: ResearchBackend | None = None) -> None:
        self.backend = backend or make_research_backend()
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
                        "Privacy firewall needs confirmation before sending this search query because it "
                        "appears to contain personal/chat/memory data. Suggested anonymized query:\n"
                        f"{decision.sanitized or '[empty]'}"
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
                    "Privacy firewall blocked outbound web search because the query appears to contain "
                    f"private data: {', '.join(decision.reasons)}. "
                    f"Suggested anonymized query: {decision.sanitized or '[empty]'}"
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
        write_log(f'[research_backend_selected] backend="{_quote_log(_backend_name(self.backend))}"')
        if not _backend_available(self.backend):
            write_log(f'[research_backend_unavailable] reason="{_quote_log(_backend_name(self.backend))}"')
            return ResearchResult(
                answer=(
                    "Web search backend is not configured, so I cannot verify fresh/current "
                    "information or provide real citations for this query."
                ),
                used_search=False,
                error="web backend unavailable",
                sanitized_query=decision.sanitized,
                privacy_reasons=decision.reasons,
                original_user_query=query,
                backend_name=_backend_name(self.backend),
                privacy_decision=decision,
                warnings=["web backend unavailable"],
            )

        write_log(f'[research_query_sent] query="{_quote_log(decision.sanitized)}"')
        try:
            raw_results = self.backend.search(decision.sanitized, max_search_results)
        except Exception as exc:
            write_log(f'[research_backend_unavailable] reason="{_quote_log(exc)}"')
            return ResearchResult(
                answer=(
                    "Web search backend is not available right now, so I cannot provide "
                    f"verified citations. Backend error: {exc}"
                ),
                used_search=False,
                error="web backend unavailable",
                sanitized_query=decision.sanitized,
                privacy_reasons=decision.reasons,
                original_user_query=query,
                backend_name=_backend_name(self.backend),
                privacy_decision=decision,
                warnings=[str(exc)],
            )
        search_results = [_to_search_result(item, index + 1) for index, item in enumerate(raw_results)]
        write_log(f'[research_search_results] count="{len(search_results)}"')
        if not search_results:
            return ResearchResult(
                answer="Web search returned no results for the sanitized query.",
                sources=[],
                used_search=True,
                sanitized_query=decision.sanitized,
                privacy_reasons=decision.reasons,
                error="no search results",
                original_user_query=query,
                backend_name=_backend_name(self.backend),
                search_results=search_results,
                privacy_decision=decision,
                warnings=["no search results"],
            )
        read_sources: list[ResearchSource] = []
        ranked_sources: list[ResearchSource] = []
        fetched_pages: list[FetchedPage] = []
        warnings: list[str] = []
        for result in search_results[:max_pages_to_read]:
            source_id = f"S{len(ranked_sources) + 1}"
            domain = domain_from_url(result.url)
            allowed, reason = is_fetch_url_allowed(result.url)
            if not allowed:
                write_log(f'[research_fetch_skipped] url="{_quote_log(result.url)}" reason="{reason}"')
                warnings.append(f"{result.title}: {reason}")
                ranked_sources.append(ResearchSource(
                    id=source_id,
                    title=result.title,
                    url=result.url,
                    snippet=compress_page_text(result.snippet, max_chars=240),
                    source=result.source,
                    rank=result.rank,
                    domain=domain,
                    fetched=False,
                    read_ok=False,
                    read_status="failed",
                    failure_reason=reason,
                ))
                continue
            try:
                if hasattr(self.backend, "fetch"):
                    page = self.backend.fetch(result.url, max_chars=10000)
                else:
                    page_text = self.backend.read_url(result.url)  # type: ignore[attr-defined]
                    page = FetchedPage(url=result.url, text=page_text, ok=bool(page_text))
                fetched_pages.append(page)
            except Exception as exc:
                write_log(f'[research_fetch_skipped] url="{_quote_log(result.url)}" reason="{_quote_log(exc)}"')
                warnings.append(f"{result.title}: {exc}")
                ranked_sources.append(ResearchSource(
                    id=source_id,
                    title=result.title,
                    url=result.url,
                    snippet=compress_page_text(result.snippet, max_chars=240),
                    source=result.source,
                    rank=result.rank,
                    domain=domain,
                    fetched=True,
                    read_ok=False,
                    read_status="failed",
                    failure_reason=str(exc),
                ))
                continue
            if not page.ok or not page.text:
                write_log(
                    "[research_fetch_skipped] "
                    f'url="{_quote_log(result.url)}" reason="{_quote_log(page.error or "empty_page")}"'
                )
                reason = page.error or "empty_page"
                warnings.append(f"{result.title}: {reason}")
                ranked_sources.append(ResearchSource(
                    id=source_id,
                    title=page.title or result.title,
                    url=result.url,
                    snippet=compress_page_text(result.snippet, max_chars=240),
                    retrieved_at=page.fetched_at,
                    source=result.source,
                    rank=result.rank,
                    domain=domain,
                    fetched=True,
                    read_ok=False,
                    read_status="failed",
                    failure_reason=reason,
                ))
                continue
            excerpt = relevant_excerpt(page.text, decision.sanitized, max_chars=1000)
            source = ResearchSource(
                id=source_id,
                title=page.title or result.title or result.url,
                url=result.url,
                snippet=compress_page_text(result.snippet, max_chars=240),
                retrieved_at=page.fetched_at,
                facts=[excerpt],
                source=result.source,
                rank=result.rank,
                domain=domain,
                fetched=True,
                read_ok=True,
                excerpt=excerpt,
                relevance_score=source_relevance(result, page.text, decision.sanitized),
                read_status="read",
            )
            read_sources.append(source)
            ranked_sources.append(source)

        ranked_sources.sort(key=lambda item: (item.read_ok, item.relevance_score), reverse=True)
        write_log(f'[research_sources_ranked] count="{len(ranked_sources)}"')
        if search_results and not read_sources:
            answer = (
                "Я нашёл поисковую выдачу, но не смог безопасно прочитать страницы для цитирования. "
                "Не буду придумывать источники без прочитанного содержимого."
            )
            return ResearchResult(
                answer=answer,
                sources=[],
                used_search=True,
                sanitized_query=decision.sanitized,
                privacy_reasons=decision.reasons,
                error="no readable sources",
                original_user_query=query,
                backend_name=_backend_name(self.backend),
                search_results=search_results,
                fetched_pages=fetched_pages,
                ranked_sources=ranked_sources,
                final_answer=answer,
                warnings=warnings,
                privacy_decision=decision,
            )
        answer_body, extracted_facts = synthesize_research_answer(decision.sanitized, read_sources, warnings)
        citations = build_citations(read_sources)
        answer = f"{answer_body}\n\n{render_citations(citations)}".strip()
        write_log(f'[research_answer_ready] source_count="{len(read_sources)}"')
        return ResearchResult(
            answer=answer,
            sources=[source for source in read_sources if source.used_in_answer],
            used_search=True,
            sanitized_query=decision.sanitized,
            privacy_reasons=decision.reasons,
            original_user_query=query,
            backend_name=_backend_name(self.backend),
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
