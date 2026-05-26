"""
Worker для асинхронной генерации.

Жизненный цикл:
1. UI создаёт InferenceWorker(profile, user_message, code_context, history).
2. start() — поток запускается.
3. model_loading сигнал — UI показывает "Загружаю модель..." (только при первом запросе).
4. chunk_received — UI стримит токены.
5. finished — UI разблокирует кнопку отправки.

Стоп: stop() ставит флаг, текущий чанк дойдёт, дальше прерываемся.
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from core.model_manager import ModelManager, LLAMA_AVAILABLE
from core.profiles import AIProfile
from core.paths import resolve_model_path
from . import prompt_builder


class InferenceWorker(QThread):
    chunk_received = pyqtSignal(str)
    model_loading = pyqtSignal(str)        # path
    model_loaded = pyqtSignal(str, bool, str)  # path, success, error
    status = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        profile: AIProfile,
        user_message: str,
        code_context: str = "",
        rag_snippets: str = "",
        history: list[tuple[str, str]] | None = None,
        user_name: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.profile = profile
        self.user_message = user_message
        self.code_context = code_context
        self.rag_snippets = rag_snippets
        self.history = history or []
        self.user_name = user_name
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        if not LLAMA_AVAILABLE:
            self.chunk_received.emit("\n[Ошибка: llama-cpp-python не установлен]\n")
            self.finished_signal.emit()
            return

        if not self.profile.model_file:
            self.chunk_received.emit("\n[Не выбран файл модели в настройках]\n")
            self.finished_signal.emit()
            return

        try:
            built = prompt_builder.build(
                profile=self.profile,
                user_message=self.user_message,
                code_context=self.code_context,
                rag_snippets=self.rag_snippets,
                history=self.history,
                user_name=self.user_name,
            )

            if built.code_context_trimmed:
                self.status.emit("⚠ Контекст файла обрезан по лимиту токенов")

            mm = ModelManager.instance()
            model_path = resolve_model_path(self.profile.model_file)

            # подписываемся локально на события загрузки этой конкретной модели
            def _on_start(path: str) -> None:
                if path == model_path:
                    self.model_loading.emit(path)

            def _on_finish(path: str, ok: bool, err: str | None) -> None:
                if path == model_path:
                    self.model_loaded.emit(path, ok, err or "")

            mm.on_load_start(_on_start)
            mm.on_load_finish(_on_finish)

            model = mm.get_model(
                path=model_path,
                n_ctx=self.profile.n_ctx,
                n_gpu_layers=self.profile.n_gpu_layers,
            )

            stop_seq = self.profile.stop_sequences or self._default_stops()

            stream = model(
                built.formatted,
                max_tokens=self.profile.max_tokens,
                temperature=self.profile.temperature,
                top_p=self.profile.top_p,
                top_k=self.profile.top_k,
                repeat_penalty=self.profile.repeat_penalty,
                stop=stop_seq,
                stream=True,
            )

            for chunk in stream:
                if self._stop:
                    self.chunk_received.emit("\n[остановлено]")
                    break
                text = chunk["choices"][0]["text"]
                if text:
                    self.chunk_received.emit(text)

        except Exception as e:
            self.chunk_received.emit(f"\n[Ошибка инференса: {e}]\n")
        finally:
            self.finished_signal.emit()

    def _default_stops(self) -> list[str]:
        """Минимальные стоп-токены чтобы модель не зацикливалась."""
        return ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>"]
