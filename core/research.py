"""Researcher/Search profile tool pipeline primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol


FRESH_QUERY_RE = re.compile(
    r"(сейчас|актуальн|последн|новост|сегодня|цена|стоимост|верси[яи]|"
    r"закон|релиз|compare|comparison|latest|current|today|price|news|release|version)",
    re.IGNORECASE,
)


@dataclass
class ResearchSource:
    title: str
    url: str
    snippet: str = ""
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    facts: list[str] = field(default_factory=list)

    def citation(self, index: int) -> str:
        return f"[{index}] {self.title} — {self.url}"


@dataclass
class ResearchResult:
    answer: str
    sources: list[ResearchSource] = field(default_factory=list)
    used_search: bool = False
    error: str = ""


class WebSearchBackend(Protocol):
    available: bool

    def search(self, query: str, max_results: int) -> list[ResearchSource]:
        ...

    def read_url(self, url: str, max_chars: int = 8000) -> str:
        ...


class UnavailableWebBackend:
    available = False

    def search(self, query: str, max_results: int) -> list[ResearchSource]:
        raise RuntimeError("Web search backend is not configured.")

    def read_url(self, url: str, max_chars: int = 8000) -> str:
        raise RuntimeError("Web page reader backend is not configured.")


class FakeWebBackend:
    """Deterministic backend for tests and future dev smoke harnesses."""

    available = True

    def __init__(self, results: list[ResearchSource] | None = None, pages: dict[str, str] | None = None) -> None:
        self.results = results or []
        self.pages = pages or {}
        self.search_calls: list[tuple[str, int]] = []
        self.read_calls: list[str] = []

    def search(self, query: str, max_results: int) -> list[ResearchSource]:
        self.search_calls.append((query, max_results))
        return self.results[:max_results]

    def read_url(self, url: str, max_chars: int = 8000) -> str:
        self.read_calls.append(url)
        return self.pages.get(url, "")[:max_chars]


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


def render_citations(sources: list[ResearchSource]) -> str:
    if not sources:
        return ""
    lines = ["Источники:"]
    for index, source in enumerate(sources, start=1):
        lines.append(source.citation(index))
    return "\n".join(lines)


class ResearchPipeline:
    def __init__(self, backend: WebSearchBackend | None = None) -> None:
        self.backend = backend or UnavailableWebBackend()

    def run(
        self,
        query: str,
        *,
        max_search_results: int = 5,
        max_pages_to_read: int = 3,
        require_sources_for_fresh_info: bool = True,
    ) -> ResearchResult:
        if not needs_web_search(query, require_sources_for_fresh_info):
            return ResearchResult(answer="", used_search=False)
        if not getattr(self.backend, "available", False):
            return ResearchResult(
                answer=(
                    "Web search backend is not configured, so I cannot verify fresh/current "
                    "information or provide real citations for this query."
                ),
                used_search=False,
                error="web backend unavailable",
            )

        sources = self.backend.search(query, max_search_results)
        enriched: list[ResearchSource] = []
        for source in sources[:max_pages_to_read]:
            page_text = self.backend.read_url(source.url)
            facts = list(source.facts)
            if page_text:
                facts.append(compress_page_text(page_text, max_chars=1200))
            enriched.append(ResearchSource(
                title=source.title,
                url=source.url,
                snippet=source.snippet,
                retrieved_at=source.retrieved_at,
                facts=facts[:5],
            ))

        facts_text = "\n".join(
            f"[{idx}] {source.title}: " + "; ".join(source.facts or [source.snippet])
            for idx, source in enumerate(enriched, start=1)
        )
        answer = (
            "Найденные факты:\n"
            f"{facts_text}\n\n"
            f"{render_citations(enriched)}"
        ).strip()
        return ResearchResult(answer=answer, sources=enriched, used_search=True)
