"""
Worker для асинхронной генерации.

Жизненный цикл:
1. UI создаёт InferenceWorker(profile, user_message, ...).
2. start() — поток запускается.
3. model_loading → UI показывает "Загружаю модель...".
4. chunk_received → UI стримит токены.
5. finished_signal → UI разблокирует кнопку отправки.

Стоп: stop() ставит флаг, текущий чанк дойдёт, дальше прерываемся.

Логика:
- Промпт всегда собирается ОДИН РАЗ через prompt_builder.
- Если активен Vision-профиль и в attached_files есть картинки И у модели есть
  chat_handler — идём через create_chat_completion с image_url'ами.
- Иначе — обычный model(string, ...) с готовой форматной строкой.
"""

from __future__ import annotations

import base64
import os
import re

from PyQt6.QtCore import QThread, pyqtSignal

from core.model_manager import ModelManager, LLAMA_AVAILABLE
from core.profiles import AIProfile, ProfileKind
from core.paths import resolve_model_path
from . import prompt_builder


# Подавляем ложные срабатывания детектора повторов: ищем блок 20+ символов,
# который повторяется подряд 3+ раза. Это надёжно отлавливает зацикливание,
# но не триггерится на нормальный код типа "self.x = self.y".
_LOOP_PATTERN = re.compile(r"(.{20,}?)\1{2,}", re.DOTALL)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


class InferenceWorker(QThread):
    chunk_received = pyqtSignal(str)
    model_loading = pyqtSignal(str)            # path
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
        attached_files: list[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.profile = profile
        self.user_message = user_message
        self.code_context = code_context
        self.rag_snippets = rag_snippets
        self.history = history or []
        self.user_name = user_name
        self.attached_files = attached_files or []
        self._stop = False

        # callbacks для отписки от ModelManager в finally
        self._cb_start = None
        self._cb_finish = None

    def stop(self) -> None:
        self._stop = True

    # ============================================================
    # Главный entry
    # ============================================================

    def run(self) -> None:
        if not LLAMA_AVAILABLE:
            self.chunk_received.emit("\n[Ошибка: llama-cpp-python не установлен]\n")
            self.finished_signal.emit()
            return

        if not self.profile.model_file:
            self.chunk_received.emit("\n[Не выбран файл модели в настройках]\n")
            self.finished_signal.emit()
            return

        mm = ModelManager.instance()
        model_path = resolve_model_path(self.profile.model_file)

        # одноразовая подписка на события загрузки конкретно этой модели
        def _on_start(path: str) -> None:
            if path == model_path:
                self.model_loading.emit(path)

        def _on_finish(path: str, ok: bool, err) -> None:
            if path == model_path:
                self.model_loaded.emit(path, ok, err or "")

        self._cb_start = _on_start
        self._cb_finish = _on_finish
        mm.on_load_start(_on_start)
        mm.on_load_finish(_on_finish)

        try:
            # 1) Готовим vision-аргументы если профиль это поддерживает
            mmproj_path = ""
            vision_handler = ""
            if self.profile.kind == ProfileKind.VISION and self.profile.mmproj_file:
                mmproj_path = resolve_model_path(self.profile.mmproj_file)
                vision_handler = self.profile.vision_handler

            # 2) Грузим модель (или достаём из кэша)
            model = mm.get_model(
                path=model_path,
                n_ctx=self.profile.n_ctx,
                n_gpu_layers=self.profile.n_gpu_layers,
                mmproj_path=mmproj_path,
                vision_handler=vision_handler,
            )

            # 3) Определяем картинки
            image_paths = [
                p for p in self.attached_files
                if os.path.splitext(p)[1].lower() in IMAGE_EXTS
            ]

            has_chat_handler = getattr(model, "chat_handler", None) is not None
            use_chat_mode = bool(image_paths) and has_chat_handler

            # Картинки прикреплены к не-Vision модели — предупреждаем и игнорируем
            if image_paths and not has_chat_handler:
                self.chunk_received.emit(
                    "<i style='color:#CE9178;'>[Эта модель не понимает изображения. "
                    "Переключитесь на Vision-профиль для работы с картинками.]</i><br><br>"
                )
                image_paths = []
                use_chat_mode = False

            stop_seq = self.profile.stop_sequences or self._default_stops()

            # 4) Один из двух режимов
            if use_chat_mode:
                self._run_chat_mode(model, image_paths, stop_seq)
            else:
                self._run_completion_mode(model, stop_seq)

        except Exception as e:
            self.chunk_received.emit(f"\n[Ошибка инференса: {e}]\n")
        finally:
            # отписка от ModelManager — иначе callbacks накапливаются
            if self._cb_start:
                mm.off_load_start(self._cb_start)
            if self._cb_finish:
                mm.off_load_finish(self._cb_finish)
            self.finished_signal.emit()

    # ============================================================
    # Режим 1: completion (обычный текст)
    # ============================================================

    def _run_completion_mode(self, model, stop_seq: list[str]) -> None:
        built = prompt_builder.build(
            profile=self.profile,
            user_message=self.user_message,
            code_context=self.code_context,
            rag_snippets=self.rag_snippets,
            history=self.history,
            user_name=self.user_name,
        )

        if built.code_context_trimmed:
            self.status.emit("⚠ Контекст обрезан по лимиту токенов")

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

        generated = ""
        for chunk in stream:
            if self._stop:
                self.chunk_received.emit("\n[остановлено]")
                break
            text = chunk["choices"][0]["text"]
            if text:
                generated += text
                self.chunk_received.emit(text)
                if self._detect_loop(generated):
                    self.chunk_received.emit(
                        "\n<i style='color:#888;'>[прервано: обнаружено зацикливание]</i>"
                    )
                    break

    # ============================================================
    # Режим 2: chat completion (Vision)
    # ============================================================

    def _run_chat_mode(self, model, image_paths: list[str], stop_seq: list[str]) -> None:
        self.status.emit("Кодирую изображения...")

        built = prompt_builder.build_messages(
            profile=self.profile,
            user_message=self.user_message,
            code_context=self.code_context,
            rag_snippets=self.rag_snippets,
            history=self.history,
            user_name=self.user_name,
        )

        if built.code_context_trimmed:
            self.status.emit("⚠ Контекст обрезан по лимиту токенов")

        messages = list(built.messages)

        # Последний user-message превращаем из строки в массив [text + image_url(s)]
        last = messages[-1]
        if last["role"] != "user":
            # на всякий, не должно быть
            messages.append({"role": "user", "content": []})
            last = messages[-1]

        text_content = last["content"] if isinstance(last["content"], str) else self.user_message
        content_list: list[dict] = [{"type": "text", "text": text_content}]

        for img_path in image_paths:
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(img_path)[1].lower().lstrip(".")
                mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            except Exception as e:
                self.status.emit(f"Ошибка чтения {os.path.basename(img_path)}: {e}")

        last["content"] = content_list

        # repeat_penalty в chat-completion жёстче ведёт себя; страхуем минимум 1.1
        rep_pen = max(self.profile.repeat_penalty, 1.1)

        stream = model.create_chat_completion(
            messages=messages,
            max_tokens=self.profile.max_tokens,
            temperature=self.profile.temperature,
            top_p=self.profile.top_p,
            top_k=self.profile.top_k,
            repeat_penalty=rep_pen,
            stop=stop_seq,
            stream=True,
        )

        generated = ""
        for chunk in stream:
            if self._stop:
                self.chunk_received.emit("\n[остановлено]")
                break
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content", "")
            if text:
                generated += text
                self.chunk_received.emit(text)
                if self._detect_loop(generated):
                    self.chunk_received.emit(
                        "\n<i style='color:#888;'>[прервано: обнаружено зацикливание]</i>"
                    )
                    break

    # ============================================================
    # Утилиты
    # ============================================================

    @staticmethod
    def _detect_loop(text: str) -> bool:
        """
        Возвращает True если в хвосте текста есть блок 20+ символов,
        повторяющийся подряд 3+ раза. Это надёжный признак зацикливания,
        который не триггерится на нормальные паттерны кода.
        """
        if len(text) < 80:
            return False
        # ищем только в хвосте — если зацикливание уже началось, оно там
        tail = text[-400:]
        return _LOOP_PATTERN.search(tail) is not None

    def _default_stops(self) -> list[str]:
        return [
            "<|im_end|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "<end_of_turn>",
            "User:",
            "USER:",
            "Assistant:",
            "ASSISTANT:",
        ]
