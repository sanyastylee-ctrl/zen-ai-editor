"""QThread wrapper for the Researcher profile web-search pipeline."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from core.profiles import AIProfile
from core.research import ResearchPipeline, make_research_backend, render_citations


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
        self.backend = backend or make_research_backend(
            getattr(profile, "search_backend", "auto"),
            getattr(profile, "searxng_url", ""),
        )
        self.confirmed_outbound = confirmed_outbound
        self._stop = False
        self.research_pending_confirmation = False
        self.research_cancelled = False
        self.research_confirmation_payload: dict = {}
        self.last_result = None

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self.status.emit("Поисковик: проверяю, нужен ли интернет...")
            pipeline = ResearchPipeline(self.backend)
            result = pipeline.run(
                self.user_message,
                max_search_results=getattr(self.profile, "max_search_results", 5),
                max_pages_to_read=getattr(self.profile, "max_pages_to_read", 3),
                require_sources_for_fresh_info=getattr(self.profile, "require_sources_for_fresh_info", True),
                confirmed_outbound=self.confirmed_outbound,
            )
            self.last_result = result
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
            if result.used_search and result.sanitized_query:
                self.chunk_received.emit(f"Поисковый запрос: {result.sanitized_query}\n\n")
            if result.error:
                self.chunk_received.emit(f"{result.answer}\n")
                return
            if result.used_search:
                self.chunk_received.emit(result.answer)
                if result.sources and "Источники:" not in result.answer:
                    self.chunk_received.emit("\n\n" + render_citations(result.sources))
                return
            self.chunk_received.emit(
                "Этот запрос не требует актуального интернет-поиска. "
                "Отвечу через обычную модель Поисковика.\n"
            )
        finally:
            self.finished_signal.emit()
