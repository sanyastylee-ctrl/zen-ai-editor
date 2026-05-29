"""Vision Assist worker.

This worker is intentionally read-only: it can look at attached images and
produce a compact visual_context for the coder agent, but it never edits files
or runs terminal commands.
"""

from __future__ import annotations

import base64
import os
import time
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from core.diagnostics import write_log
from core.model_manager import LLAMA_AVAILABLE, ModelManager
from core.paths import resolve_model_path
from core.profiles import AIProfile


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
MAX_GENERATION_SECONDS = 180


class VisionWorker(QThread):
    chunk_received = pyqtSignal(str)
    model_loading = pyqtSignal(str)
    model_loaded = pyqtSignal(str, bool, str)
    status = pyqtSignal(str)
    visual_context_ready = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        profile: AIProfile,
        user_message: str,
        image_paths: list[str],
        answer_mode: bool = False,
        max_generation_seconds: int = MAX_GENERATION_SECONDS,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.profile = profile
        self.user_message = user_message
        self.image_paths = [
            path for path in image_paths
            if os.path.splitext(path)[1].lower() in IMAGE_EXTS
        ]
        self.answer_mode = answer_mode
        self.max_generation_seconds = max(1, int(max_generation_seconds or MAX_GENERATION_SECONDS))
        self._stop = False
        self._cb_start = None
        self._cb_finish = None

    def stop(self) -> None:
        self._stop = True

    def _log(self, event: str, **fields) -> None:
        parts = [f"[vision_{event}]"]
        for key in sorted(fields):
            value = str(fields[key]).replace("\n", "\\n")
            parts.append(f'{key}="{value}"')
        write_log(" ".join(parts))

    def run(self) -> None:
        self._log(
            "start",
            answer_mode=self.answer_mode,
            image_count=len(self.image_paths),
            max_generation_seconds=self.max_generation_seconds,
        )
        for idx, path in enumerate(self.image_paths, 1):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = -1
            self._log("image", index=idx, path=path, size=size)

        if not LLAMA_AVAILABLE:
            self._fail("llama-cpp-python не установлен")
            self.finished_signal.emit()
            return
        if not self.image_paths:
            self._fail("нет изображений для Vision Assist")
            self.finished_signal.emit()
            return
        if not getattr(self.profile, "vision_model_file", ""):
            self._fail("не выбрана vision model для Vision Assist")
            self.finished_signal.emit()
            return
        if not getattr(self.profile, "mmproj_file", ""):
            self._fail("не выбран mmproj файл для Vision Assist")
            self.finished_signal.emit()
            return
        if not getattr(self.profile, "vision_handler", ""):
            self._fail("не выбран vision handler для Vision Assist")
            self.finished_signal.emit()
            return

        mm = ModelManager.instance()
        model_path = resolve_model_path(self.profile.vision_model_file)
        mmproj_path = resolve_model_path(self.profile.mmproj_file)

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
            self.status.emit("Vision Assist: загружаю модель...")
            self._log("model_load_begin", model_path=model_path, mmproj_path=mmproj_path)
            model = mm.get_model(
                path=model_path,
                n_ctx=self.profile.n_ctx,
                n_gpu_layers=self.profile.n_gpu_layers,
                mmproj_path=mmproj_path,
                vision_handler=self.profile.vision_handler,
            )
            self._log("model_load_end", model_path=model_path, ok=True)
            messages = self._build_messages()
            self._log("inference_begin", image_count=len(self.image_paths))
            stream = model.create_chat_completion(
                messages=messages,
                max_tokens=min(self.profile.max_tokens, 2048),
                temperature=min(max(self.profile.temperature, 0.1), 0.35),
                top_p=self.profile.top_p,
                top_k=self.profile.top_k,
                repeat_penalty=max(self.profile.repeat_penalty, 1.1),
                stream=True,
            )
            generated = ""
            started = time.monotonic()
            self.status.emit("Vision Assist: анализирую изображение...")
            for chunk in stream:
                if self._stop:
                    if self.answer_mode:
                        self.chunk_received.emit("\n[остановлено]")
                    break
                if time.monotonic() - started > self.max_generation_seconds:
                    self._stop = True
                    if self.answer_mode:
                        self.chunk_received.emit("\n[остановлено: превышен лимит времени Vision]\n")
                    self._fail("Vision inference timeout")
                    self._log("timeout", seconds=self.max_generation_seconds)
                    return
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    generated += text
                    if self.answer_mode:
                        self.chunk_received.emit(text)

            context = self._compact(generated)
            if not context.strip():
                self._fail("Vision returned empty visual_context")
                return
            self._log("inference_end", output_chars=len(generated), context_chars=len(context))
            self._log("visual_context_ready", chars=len(context))
            self.visual_context_ready.emit(context)
        except Exception as e:
            self._log("error", error=str(e), traceback=traceback.format_exc())
            self._fail(str(e))
        finally:
            if self._cb_start:
                mm.off_load_start(self._cb_start)
            if self._cb_finish:
                mm.off_load_finish(self._cb_finish)
            self._log("finish")
            self.finished_signal.emit()

    def _build_messages(self) -> list[dict]:
        system = (
            "You are ZenAI Vision Assist. You only analyze attached images. "
            "You never edit files, never run terminal commands, and never claim "
            "that code changes were applied.\n"
        )
        if self.answer_mode:
            system += (
                "Answer the user's image question directly in Russian. If text is visible, "
                "transcribe it. If uncertain, say what is uncertain."
            )
        else:
            system += (
                "Return a compact structured visual_context in Russian for a coding agent. "
                "Use these sections: visible_summary, ocr_text, ui_elements, suspected_issue, "
                "likely_files_or_components, confidence, uncertainty. Keep it concise."
            )

        text = self.user_message.strip() or "Проанализируй изображение."
        content: list[dict] = [{"type": "text", "text": text}]
        for path in self.image_paths:
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(path)[1].lower().lstrip(".")
                mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            except OSError as e:
                content.append({"type": "text", "text": f"[image read error: {os.path.basename(path)}: {e}]"})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    def _compact(self, text: str) -> str:
        value = (text or "").strip()
        limit = int(getattr(self.profile, "max_visual_context_chars", 4000) or 4000)
        limit = max(1000, min(limit, 12000))
        if len(value) > limit:
            value = value[:limit].rstrip() + "\n[visual_context truncated]"
        return value

    def _fail(self, message: str) -> None:
        text = f"Vision Assist error: {message}"
        self._log("failed", error=text)
        self.error_signal.emit(text)
        if self.answer_mode:
            self.chunk_received.emit(f"\n[{text}]\n")
