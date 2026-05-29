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
import hashlib
import os
import re
import time
import uuid
from dataclasses import replace

from PyQt6.QtCore import QThread, pyqtSignal

from core.diagnostics import acceleration_warning, write_log
from core.model_manager import ModelManager, LLAMA_AVAILABLE
from core.profiles import AIProfile, ProfileKind
from core.paths import resolve_model_path
from core.companion import companion_echo_similarity, is_companion_echo_response, normalize_companion_reply_for_repeat
from . import prompt_builder


# Подавляем ложные срабатывания детектора повторов: ищем блок 20+ символов,
# который повторяется подряд 3+ раза. Это надёжно отлавливает зацикливание,
# но не триггерится на нормальный код типа "self.x = self.y".
_LOOP_PATTERN = re.compile(r"(.{20,}?)\1{2,}", re.DOTALL)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
MAX_GENERATION_SECONDS = 180


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
        max_generation_seconds: int = MAX_GENERATION_SECONDS,
        allow_agent_actions: bool = True,
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
        self.max_generation_seconds = max(1, int(max_generation_seconds or MAX_GENERATION_SECONDS))
        self.allow_agent_actions = allow_agent_actions
        self._stop = False
        self.run_id = uuid.uuid4().hex[:12]
        self.companion_response_valid = True
        self.companion_block_reason = ""
        self.validated_response_text = ""

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
            if self.profile.kind == ProfileKind.CODER:
                warning = acceleration_warning(self.profile, model_path)
                if warning:
                    self.status.emit("⚠ CPU backend")
                    self.chunk_received.emit(f"\n[Предупреждение: {warning}]\n")

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
        if self.profile.kind == ProfileKind.COMPANION:
            self._run_companion_completion_mode(model, stop_seq)
            return

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
        started = time.monotonic()
        for chunk in stream:
            if self._stop:
                self.chunk_received.emit("\n[остановлено]")
                break
            if time.monotonic() - started > self.max_generation_seconds:
                self._stop = True
                self.chunk_received.emit("\n[остановлено: превышен лимит времени генерации]\n")
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
        
        # Выполняем действия агента, когда текст сгенерирован полностью
        self._execute_agent_actions(generated)

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]

    @staticmethod
    def _normalize_response(text: str) -> str:
        return normalize_companion_reply_for_repeat(text)

    def _is_companion_repeat(self, response: str, prior_responses: list[str]) -> tuple[bool, str, float]:
        if "[previous repetitive companion reply omitted]" in str(response or ""):
            return True, "history_compaction_marker_leaked", 1.0
        response_norm = self._normalize_response(response)
        if not response_norm:
            return False, "", 0.0
        best_similarity = 0.0
        for index, prior in enumerate(prior_responses):
            prior_norm = self._normalize_response(prior)
            if not prior_norm:
                continue
            similarity = 1.0 if response_norm == prior_norm else companion_echo_similarity(prior_norm, response_norm)
            best_similarity = max(best_similarity, similarity)
            if response_norm == prior_norm:
                return True, f"assistant_repeat_{index}", similarity
            if similarity >= 0.94 and len(response_norm) >= 24:
                return True, f"assistant_near_repeat_{index}", similarity
        return False, "", best_similarity

    def _log_companion(self, event: str, **fields) -> None:
        parts = [f"[companion_{event}]"]
        for key in sorted(fields):
            value = str(fields[key]).replace("\\", "\\\\").replace('"', "'").replace("\n", "\\n")
            parts.append(f'{key}="{value}"')
        write_log(" ".join(parts))

    def _run_companion_completion_mode(self, model, stop_seq: list[str]) -> None:
        prior_responses = [str(assistant or "") for _user, assistant in (self.history[-5:] if self.history else [])]
        previous_response = prior_responses[-1] if prior_responses else ""
        previous_hash = self._hash_text(previous_response)
        latest_hash = self._hash_text(self.user_message)
        total_messages = len(self.history) * 2 + 1
        self._log_companion(
            "send_start",
            run_id=self.run_id,
            session_id=self.profile.id,
            turn_id=len(self.history) + 1,
            text_hash=latest_hash,
            text_len=len(self.user_message or ""),
        )

        retry_note = (
            "\n\nImportant live-turn rule: answer the latest user message, not your previous reply. "
            "Do not repeat your last response; mention the latest marker/message if present. "
            "Do not echo, quote, or parenthesize the user's message as your whole reply. "
            "Reply as Lera with new meaningful content."
        )
        attempts: list[tuple[list[tuple[str, str]], AIProfile]] = [
            (self.history, self.profile),
            ([], replace(self.profile, system_prompt=(self.profile.system_prompt or "") + retry_note)),
        ]

        for attempt, (history, profile) in enumerate(attempts):
            built = prompt_builder.build(
                profile=profile,
                user_message=self.user_message,
                code_context=self.code_context,
                rag_snippets=self.rag_snippets,
                history=history,
                user_name=self.user_name,
            )
            included_messages = built.history_used * 2 + 1
            last_roles = "user,assistant" if built.history_used else ""
            if built.history:
                last_roles = ",".join(["user", "assistant"] * min(2, len(built.history)))
            contains_latest = bool(self.user_message and self.user_message in built.formatted)
            if len(history) != built.history_used:
                self._log_companion(
                    "context_trim",
                    run_id=self.run_id,
                    before=len(history),
                    after=built.history_used,
                    kept_last_user=contains_latest,
                )
            self._log_companion(
                "history_build",
                run_id=self.run_id,
                total_messages=total_messages,
                included_messages=included_messages,
                last_roles=last_roles,
                last_user_hash=latest_hash,
            )
            self._log_companion(
                "memory_used",
                run_id=self.run_id,
                memory_items_count=built.system.count("- ["),
                summary_chars=len(built.system),
            )
            prompt_hash = self._hash_text(built.formatted)
            self._log_companion(
                "prompt_ready",
                run_id=self.run_id,
                prompt_hash=prompt_hash,
                prompt_chars=len(built.formatted),
                contains_latest_user=contains_latest,
                latest_user_hash=latest_hash,
                last_roles=last_roles,
            )
            if not contains_latest:
                msg = "\n[Ошибка Леры: последний user message не попал в prompt; генерация остановлена]\n"
                self.companion_response_valid = False
                self.companion_block_reason = "latest_user_missing"
                self.validated_response_text = ""
                self._log_companion(
                    "prompt_abort",
                    run_id=self.run_id,
                    reason="latest_user_missing",
                    prompt_hash=prompt_hash,
                )
                self.chunk_received.emit(msg)
                return

            self._log_companion("worker_start", run_id=self.run_id, prompt_hash=prompt_hash)
            generated, emitted_to_ui, pending_text = self._generate_completion_buffer(
                model,
                built.formatted,
                stop_seq,
                stream_to_ui=False,
            )
            response_hash = self._hash_text(generated)
            self._log_companion(
                "response_raw",
                run_id=self.run_id,
                response_hash=response_hash,
                response_len=len(generated),
            )
            if not generated.strip():
                action = "retry" if attempt == 0 else "error"
                self._log_companion(
                    "response_blocked" if attempt else "retry",
                    run_id=self.run_id,
                    reason="empty_response",
                    action=action,
                    response_hash=response_hash,
                )
                if attempt == 0:
                    continue
                error = "\n[Лера сгенерировала пустой ответ. Ответ не сохранён.]\n"
                self.companion_response_valid = False
                self.companion_block_reason = "empty_response"
                self.validated_response_text = ""
                self.chunk_received.emit(error)
                self._log_companion("state_cleanup", run_id=self.run_id)
                return
            is_echo, echo_similarity = is_companion_echo_response(self.user_message, generated)
            if is_echo:
                action = "retry" if attempt == 0 else "error"
                self._log_companion(
                    "echo_detected",
                    run_id=self.run_id,
                    user_hash=latest_hash,
                    response_hash=response_hash,
                    similarity=f"{echo_similarity:.3f}",
                    action=action,
                )
                if attempt == 0:
                    self._log_companion("retry", run_id=self.run_id, reason="echo")
                    continue
                error = "\n[Лера сгенерировала повтор/эхо. Ответ не сохранён.]\n"
                self.companion_response_valid = False
                self.companion_block_reason = "echo"
                self.validated_response_text = ""
                self.chunk_received.emit(error)
                self._log_companion(
                    "response_blocked",
                    run_id=self.run_id,
                    reason="echo",
                    response_hash=response_hash,
                )
                self._log_companion("state_cleanup", run_id=self.run_id)
                return

            is_repeat, repeat_reason, repeat_similarity = self._is_companion_repeat(generated, prior_responses)
            if is_repeat:
                action = "retry" if attempt == 0 else "error"
                self._log_companion(
                    "repeat_detected",
                    run_id=self.run_id,
                    previous_response_hash=previous_hash,
                    new_response_hash=response_hash,
                    reason=repeat_reason,
                    similarity=f"{repeat_similarity:.3f}",
                    action=action,
                )
                if attempt == 0:
                    self._log_companion("retry", run_id=self.run_id, reason="repeat")
                    continue
                error = (
                    "\n[Лера остановлена: модель повторила предыдущий ответ вместо ответа "
                    "на новое сообщение. Попробуй переформулировать коротко.]\n"
                )
                self.companion_response_valid = False
                self.companion_block_reason = "repeat"
                self.validated_response_text = ""
                self.chunk_received.emit(error)
                self._log_companion(
                    "response_blocked",
                    run_id=self.run_id,
                    reason="repeat",
                    response_hash=response_hash,
                )
                self._log_companion(
                    "response_finish",
                    run_id=self.run_id,
                    response_hash=self._hash_text(error),
                    response_len=len(error),
                )
                self._log_companion("state_cleanup", run_id=self.run_id)
                return

            if pending_text:
                self.validated_response_text = generated
                self.chunk_received.emit(pending_text)
            else:
                self.validated_response_text = generated
            self._log_companion(
                "response_finish",
                run_id=self.run_id,
                response_hash=response_hash,
                response_len=len(generated),
            )
            self._log_companion("response_saved", run_id=self.run_id, response_hash=response_hash)
            self._log_companion("state_cleanup", run_id=self.run_id)
            return

    def _generate_completion_buffer(
        self,
        model,
        formatted_prompt: str,
        stop_seq: list[str],
        stream_to_ui: bool = False,
    ) -> tuple[str, bool, str]:
        stream = model(
            formatted_prompt,
            max_tokens=self.profile.max_tokens,
            temperature=self.profile.temperature,
            top_p=self.profile.top_p,
            top_k=self.profile.top_k,
            repeat_penalty=self.profile.repeat_penalty,
            stop=stop_seq,
            stream=True,
        )
        generated = ""
        pending = ""
        emitted = False
        started = time.monotonic()
        self._log_companion("stream_start", run_id=self.run_id)
        for chunk in stream:
            if self._stop:
                pending += "\n[остановлено]"
                generated += "\n[остановлено]"
                break
            if time.monotonic() - started > self.max_generation_seconds:
                self._stop = True
                pending += "\n[остановлено: превышен лимит времени генерации]\n"
                generated += "\n[остановлено: превышен лимит времени генерации]\n"
                break
            text = chunk["choices"][0]["text"]
            if text:
                generated += text
                pending += text
                if stream_to_ui:
                    # Keep a small initial buffer so short exact repeats can be
                    # caught before rendering, then stream normally for UI
                    # responsiveness on real companion turns.
                    should_flush = (
                        emitted
                        or len(pending) >= 120
                        or time.monotonic() - started >= 1.5
                    )
                    if should_flush:
                        if not emitted:
                            self._log_companion(
                                "stream_flush",
                                run_id=self.run_id,
                                chars=len(pending),
                                elapsed_ms=int((time.monotonic() - started) * 1000),
                            )
                        self.chunk_received.emit(pending)
                        pending = ""
                        emitted = True
                if self._detect_loop(generated):
                    pending += "\n[прервано: обнаружено зацикливание]"
                    generated += "\n[прервано: обнаружено зацикливание]"
                    break
        if not stream_to_ui:
            pending = generated
        return generated, emitted, pending

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
        started = time.monotonic()
        for chunk in stream:
            if self._stop:
                self.chunk_received.emit("\n[остановлено]")
                break
            if time.monotonic() - started > self.max_generation_seconds:
                self._stop = True
                self.chunk_received.emit("\n[остановлено: превышен лимит времени генерации]\n")
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
        
        # Выполняем действия агента, когда текст сгенерирован полностью
        self._execute_agent_actions(generated)

    # ============================================================
    # Логика Агента (парсинг и исполнение)
    # ============================================================

    def _execute_agent_actions(self, text: str) -> None:
        """
        Парсит ответ модели, ищет операции [FILE:], [DELETE:], [RUN:] и исполняет.

        Главная сложность: модель часто забывает закрывающие теги. Парсер должен
        работать даже когда:
        - забыт [/FILE] / [/CREATE_FILE]
        - после [FILE:] модель пишет ```python (новый формат)
        - после [CREATE_FILE:] модель тоже пишет ```python (старый формат)
        - модель ставит [FILE:] подряд без разделителей

        Стратегия: ищем маркеры начала, для каждого определяем границу как
        "первый из": закрывающая ``` после открытия / следующий [FILE:] или
        [CREATE_FILE:] / явный закрывающий тег / конец текста.
        """
        if not getattr(self.profile, "agent_mode", False) or not self.allow_agent_actions:
            return

        actions = self._parse_agent_actions(text)
        if not actions:
            return

        base_dir = os.path.abspath(os.getcwd())
        created: list[str] = []
        deleted: list[str] = []
        executed: list[str] = []
        errors: list[str] = []

        for action in actions:
            kind = action["kind"]

            if kind == "file":
                filepath = action["path"]
                content = action["content"]
                abs_path = os.path.abspath(os.path.join(base_dir, filepath))
                if not abs_path.startswith(base_dir + os.sep) and abs_path != base_dir:
                    errors.append(f"путь вне проекта: {filepath}")
                    continue
                try:
                    parent = os.path.dirname(abs_path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    with open(abs_path, "w", encoding="utf-8", newline="") as f:
                        f.write(content)
                    created.append(filepath)
                except Exception as e:
                    errors.append(f"запись {filepath}: {e}")

            elif kind == "delete":
                filepath = action["path"]
                abs_path = os.path.abspath(os.path.join(base_dir, filepath))
                if not abs_path.startswith(base_dir + os.sep):
                    errors.append(f"путь вне проекта: {filepath}")
                    continue
                try:
                    if os.path.isfile(abs_path):
                        os.remove(abs_path)
                        deleted.append(filepath)
                    else:
                        errors.append(f"не найден для удаления: {filepath}")
                except Exception as e:
                    errors.append(f"удаление {filepath}: {e}")

            elif kind == "run":
                # Запуск команды через signal в UI — UI решит, исполнять или нет,
                # покажет результат в терминале и логи. Сам worker не запускает
                # подпроцессы (это не его ответственность).
                cmd = action["command"]
                self.chunk_received.emit(
                    f"<br><span style='color:#569CD6;'>[🤖 предлагает выполнить: <code>{self._escape_html(cmd)}</code>]</span><br>"
                )
                executed.append(cmd)

        # Итоговый отчёт в чат
        if created:
            self.chunk_received.emit(
                f"<br><i style='color:#4EC9B0;'>✓ Создано/перезаписано: {', '.join(created)}</i><br>"
            )
        if deleted:
            self.chunk_received.emit(
                f"<br><i style='color:#CDA040;'>✓ Удалено: {', '.join(deleted)}</i><br>"
            )
        if errors:
            for err in errors:
                self.chunk_received.emit(
                    f"<br><b style='color:#CE4040;'>✗ {self._escape_html(err)}</b>"
                )
            self.chunk_received.emit("<br>")

    def _parse_agent_actions(self, text: str) -> list[dict]:
        """
        Возвращает список действий в порядке их появления в тексте.
        Каждое действие — dict с полем 'kind' и доп.полями:
          - kind="file":   path, content
          - kind="delete": path
          - kind="run":    command
        """
        actions: list[dict] = []

        # Все стартовые маркеры в порядке появления в тексте.
        # Поддерживаем оба синтаксиса — новый [FILE:] и старый [CREATE_FILE:].
        marker_iter = re.finditer(
            r"\[(FILE|CREATE_FILE|DELETE|RUN):\s*([^\]\n]+?)\s*\]",
            text,
            re.IGNORECASE,
        )
        markers = [(m.start(), m.end(), m.group(1).upper(), m.group(2).strip())
                   for m in marker_iter]
        if not markers:
            return []

        for idx, (start, after_marker, kind, value) in enumerate(markers):
            # Где этот блок заканчивается? — до следующего маркера или до конца текста.
            next_start = markers[idx + 1][0] if idx + 1 < len(markers) else len(text)
            block_text = text[after_marker:next_start]

            if kind in ("FILE", "CREATE_FILE"):
                content = InferenceWorker._extract_file_content(block_text)
                if content is None:
                    # совсем не нашли содержимое — пропускаем
                    continue
                actions.append({
                    "kind": "file",
                    "path": value,
                    "content": content,
                })

            elif kind == "DELETE":
                actions.append({
                    "kind": "delete",
                    "path": value,
                })

            elif kind == "RUN":
                # Команда у нас в самом маркере (value). Но если кто-то решил
                # написать её в код-блоке после маркера — тоже подхватим.
                cmd = value
                fenced = InferenceWorker._extract_file_content(block_text)
                if fenced and not cmd:
                    cmd = fenced.strip()
                if cmd:
                    actions.append({"kind": "run", "command": cmd})

        return actions

    @staticmethod
    def _extract_file_content(block_text: str) -> str | None:
        """
        Достаёт содержимое файла из блока, идущего сразу после [FILE:] / [CREATE_FILE:].

        Поддержка разных вариантов написания:
          1. ```python\n...\n```           — идеальный случай
          2. ```\n...\n```                  — без указания языка
          3. ```python\n...   (нет закрывающей) — берём всё до конца блока
          4. ```python\n...\n```\n[/CREATE_FILE]  — старый формат, тэг игнорим
          5. ...текст без кавычек...        — берём как есть (последний шанс)
        """
        # 1+2+4: пара тройных кавычек
        m = re.search(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", block_text, re.DOTALL)
        if m:
            return m.group(1)

        # 3: открывающие кавычки есть, закрывающих нет — берём до конца блока
        m = re.search(r"```[a-zA-Z0-9_+\-]*\n(.*)", block_text, re.DOTALL)
        if m:
            # обрезаем хвостовой [/CREATE_FILE] или [/FILE] если случайно влез
            content = m.group(1)
            content = re.sub(r"\[/(?:CREATE_)?FILE\]\s*$", "", content, flags=re.IGNORECASE).rstrip()
            return content

        # 5: тройных кавычек нет вообще — модель написала контент сырым
        cleaned = re.sub(r"\[/(?:CREATE_)?FILE\]\s*$", "", block_text, flags=re.IGNORECASE).strip()
        if cleaned:
            return cleaned
        return None

    @staticmethod
    def _escape_html(s: str) -> str:
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;"))

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
        # ВАЖНО: НЕ добавлять сюда "User:" / "Assistant:" / "USER:" — они рубят
        # стрим посреди нормального ответа, если модель цитирует код типа
        # `Assistant: helper()` или пишет "User: должен быть авторизован".
        # Спецтокены ChatML / Llama-3 / Gemma безопасны — их модель никогда
        # не выдаёт как обычный текст.
        return [
            "<|im_end|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "<end_of_turn>",
        ]
