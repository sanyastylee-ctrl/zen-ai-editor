"""QThread wrapper for the Researcher profile web-search pipeline."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from core.profiles import AIProfile
from core.diagnostics import write_log
from core.research import (
    ResearchPipeline,
    diagnose_research_backend,
    render_citations,
    select_research_backend,
)


class ResearchWorker(QThread):
    chunk_received = pyqtSignal(str)
    model_loading = pyqtSignal(str)
    model_loaded = pyqtSignal(str, bool, str)
    status = pyqtSignal(str)
    confirmation_required = pyqtSignal(dict)
    sources_ready = pyqtSignal(list)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        profile: AIProfile,
        user_message: str,
        backend=None,
        confirmed_outbound: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.profile = profile
        self.user_message = user_message
        backend_kind = getattr(profile, "search_backend", "auto")
        searxng_url = getattr(profile, "searxng_url", "")
        if backend is None:
            self.backend, self.backend_status = select_research_backend(backend_kind, searxng_url)
        else:
            self.backend = backend
            self.backend_status = diagnose_research_backend(backend_kind, searxng_url, backend=backend)
        self.confirmed_outbound = confirmed_outbound
        self._stop = False
        self.research_pending_confirmation = False
        self.research_cancelled = False
        self.research_confirmation_payload: dict = {}
        self.last_result = None

    def stop(self) -> None:
        self._stop = True

    def _local_searcher_answer(self) -> str:
        text = (self.user_message or "").lower()
        if "rag" in text or "retrieval" in text:
            return (
                "Коротко:\n"
                "- RAG — это подход, где модель сначала находит релевантные документы или фрагменты, "
                "а потом отвечает с опорой на найденный контекст.\n"
                "- Это помогает отвечать точнее по базе знаний, документации или проекту, не полагаясь "
                "только на память модели.\n\n"
                "Интернет-поиск здесь не нужен: вопрос про стабильное понятие."
            )
        return (
            "Коротко:\n"
            "- Этот вопрос похож на стабильный факт или объяснение, поэтому актуальный web-поиск не обязателен.\n"
            "- Если тебе нужны свежие источники, добавь «найди», «актуальный» или «сейчас»."
        )

    def run(self) -> None:
        try:
            self.status.emit(
                f"Поисковик: backend {getattr(self.backend_status, 'effective_backend', getattr(self.backend, 'name', 'unknown'))}"
            )
            pipeline = ResearchPipeline(self.backend, backend_status=self.backend_status)
            result = pipeline.run(
                self.user_message,
                max_search_results=getattr(self.profile, "max_search_results", 5),
                max_pages_to_read=getattr(self.profile, "max_pages_to_read", 3),
                require_sources_for_fresh_info=getattr(self.profile, "require_sources_for_fresh_info", True),
                confirmed_outbound=self.confirmed_outbound,
            )
            self.last_result = result
            if result.backend_status is None:
                result.backend_status = self.backend_status
            if result.used_search or result.error:
                self.sources_ready.emit([source.to_card() for source in result.ranked_sources])
            if self._stop:
                self.chunk_received.emit("\n[остановлено]")
                return
            if result.error == "privacy confirmation required":
                self.research_pending_confirmation = True
                self.research_confirmation_payload = {
                    "raw_query": self.user_message,
                    "sanitized_query": result.sanitized_query,
                    "reasons": result.privacy_reasons,
                    "profile_id": self.profile.id,
                }
                self.confirmation_required.emit(self.research_confirmation_payload)
                return
            if result.used_search and result.sanitized_query and len(result.sanitized_query.strip()) > 3:
                self.chunk_received.emit(f"Поисковый запрос: {result.sanitized_query}\n\n")
            if result.error:
                self.chunk_received.emit(f"{result.answer}\n")
                return
            if result.used_search:
                self.chunk_received.emit(result.answer)
                if result.sources and "Источники:" not in result.answer:
                    self.chunk_received.emit("\n\n" + render_citations(result.sources))
                return
            # Запрос не требует поиска (стабильный факт, объяснение)
            # НЕ говорим "ищи сам" — просто честно отвечаем что поиск не нужен
            write_log("[searcher_route_local_answer]")
            self.chunk_received.emit(self._local_searcher_answer() + "\n")
        finally:
            self.finished_signal.emit()
