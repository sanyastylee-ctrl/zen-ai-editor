"""QThread wrapper for the Researcher profile web-search pipeline."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from core.profiles import AIProfile
from core.research import ResearchPipeline, UnavailableWebBackend, render_citations


class ResearchWorker(QThread):
    chunk_received = pyqtSignal(str)
    model_loading = pyqtSignal(str)
    model_loaded = pyqtSignal(str, bool, str)
    status = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        profile: AIProfile,
        user_message: str,
        backend=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.profile = profile
        self.user_message = user_message
        self.backend = backend or UnavailableWebBackend()
        self._stop = False

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
            )
            if self._stop:
                self.chunk_received.emit("\n[остановлено]")
                return
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
