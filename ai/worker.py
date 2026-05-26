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

import base64
import os
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

            # Расширенные стоп-токены для Vision и Chat API
            stop_seq = self.profile.stop_sequences or self._default_stops()

            # Ищем картинки среди вложений
            image_paths = [
                p for p in self.attached_files 
                if p.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))
            ]

            # Проверяем, есть ли у модели настроенный модуль зрения (chat_handler)
            has_vision = getattr(model, "chat_handler", None) is not None

            if image_paths and not has_vision:
                self.chunk_received.emit(
                    "<i style='color:#CE9178;'>[Системное сообщение: Модель загружена без модуля зрения (mmproj). "
                    "Чтобы избежать галлюцинаций из-за переполнения текстом картинки, она проигнорирована.]</i><br><br>"
                )
                image_paths = []  # Очищаем список картинок, идем по текстовой ветке

            # Переменные для детектора зацикливания
            generated_text = ""
            repetition_threshold = 3  # Максимум повторений одной фразы

            # =========================================================================
            # ВЕТВЬ 1: МУЛЬТИМОДАЛЬНЫЙ ЗАПРОС (ЕСЛИ ЕСТЬ КАРТИНКИ И МОДЕЛЬ VL)
            # =========================================================================
            if image_paths:
                self.status.emit("Кодируем картинки для Vision модели...")
                
                messages = []
                
                # Собираем чистый системный промпт без лишнего мусора
                sys_prompt = self.profile.system_prompt
                if self.code_context:
                    sys_prompt += f"\n\nКонтекст кода:\n{self.code_context}"
                if self.rag_snippets:
                    sys_prompt += f"\n\nФрагменты из проекта:\n{self.rag_snippets}"
                messages.append({"role": "system", "content": sys_prompt})
                
                # Добавляем историю
                for h_u, h_a in self.history:
                    messages.append({"role": "user", "content": h_u})
                    messages.append({"role": "assistant", "content": h_a})

                # Текущий запрос
                content_list = []
                content_list.append({"type": "text", "text": self.user_message})
                
                for img_path in image_paths:
                    try:
                        with open(img_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        
                        ext = img_path.split('.')[-1].lower()
                        mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
                        
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}
                        })
                    except Exception as e:
                        self.status.emit(f"Ошибка загрузки картинки: {e}")

                messages.append({"role": "user", "content": content_list})

                stream = model.create_chat_completion(
                    messages=messages,
                    max_tokens=self.profile.max_tokens,
                    temperature=self.profile.temperature,
                    top_p=self.profile.top_p,
                    top_k=self.profile.top_k,
                    repeat_penalty=self.profile.repeat_penalty if self.profile.repeat_penalty > 1.0 else 1.1,
                    stop=stop_seq,
                    stream=True,
                )

                for chunk in stream:
                    if self._stop:
                        self.chunk_received.emit("\n[остановлено]")
                        break
                    
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        generated_text += text
                        self.chunk_received.emit(text)
                        
                        # Детектор зацикливания: ищем повторы фраз длиннее 15 символов
                        if len(generated_text) > 50:
                            # Проверяем последние куски текста на цикличность
                            tail = generated_text[-60:]
                            # Если фраза дублируется в хвосте слишком часто — рубим поток
                            parts = tail.split(text)
                            if len(parts) > repetition_threshold + 1 and len(text.strip()) > 2:
                                self.chunk_received.emit("\n<i style='color:#888888;'>[Инференс прерван: обнаружен бесконечный цикл повторений]</i>")
                                break

            # =========================================================================
            # ВЕТВЬ 2: ОБЫЧНЫЙ ТЕКСТОВЫЙ ЗАПРОС
            # =========================================================================
            else:
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
                        generated_text += text
                        self.chunk_received.emit(text)
                        
                        # Текстовый предохранитель от зацикливания
                        if len(generated_text) > 100:
                            tail = generated_text[-60:]
                            if text and tail.count(text) > repetition_threshold and len(text.strip()) > 2:
                                self.chunk_received.emit("\n<i style='color:#888888;'>[Инференс прерван: обнаружен бесконечный цикл повторений]</i>")
                                break

        except Exception as e:
            self.chunk_received.emit(f"\n[Ошибка инференса: {e}]\n")
        finally:
            self.finished_signal.emit()

    def _default_stops(self) -> list[str]:
        """Минимальные стоп-токены, включая разметку чата OpenAI/Llava."""
        return [
            "<|im_end|>", 
            "<|eot_id|>", 
            "<|end_of_text|>", 
            "<end_of_turn>", 
            "User:", 
            "USER:", 
            "Assistant:", 
            "ASSISTANT:"
        ]