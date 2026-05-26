"""
ModelManager — кэш загруженных моделей. По умолчанию max_loaded=1: одна модель
в VRAM за раз, переключение через выгрузку.

Vision-handler грузится ТОЛЬКО если в профиле явно указан mmproj_file и
vision_handler. Никакого автодетекта по папке (это ломало текстовые модели).

Маппинг строк vision_handler → класс:
    "qwen25vl"   → Qwen25VLChatHandler
    "llava15"    → Llava15ChatHandler
    "llava16"    → Llava16ChatHandler
    "minicpmv26" → MiniCPMv26ChatHandler

Загрузка модели — синхронная (3-15 сек). Вызывать из QThread.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore
    LLAMA_AVAILABLE = False


# Vision-handlers подгружаем опционально. Не все версии llama-cpp-python их имеют.
_VISION_HANDLERS: dict[str, type] = {}
if LLAMA_AVAILABLE:
    try:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        _VISION_HANDLERS["llava15"] = Llava15ChatHandler
    except ImportError:
        pass
    try:
        from llama_cpp.llama_chat_format import Llava16ChatHandler
        _VISION_HANDLERS["llava16"] = Llava16ChatHandler
    except ImportError:
        pass
    try:
        from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
        _VISION_HANDLERS["minicpmv26"] = MiniCPMv26ChatHandler
    except ImportError:
        pass
    try:
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        _VISION_HANDLERS["qwen25vl"] = Qwen25VLChatHandler
    except ImportError:
        pass


if TYPE_CHECKING:
    from llama_cpp import Llama as LlamaType


def available_vision_handlers() -> list[str]:
    """Возвращает список названий vision-handler'ов, доступных в текущей сборке."""
    return list(_VISION_HANDLERS.keys())


class ModelManager:
    """Singleton-кэш. Использовать через ModelManager.instance()."""

    _instance: "ModelManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, max_loaded: int = 1) -> None:
        self._max_loaded = max_loaded
        # ключ — (model_path, n_ctx, mmproj_path, vision_handler)
        self._models: "OrderedDict[tuple, LlamaType]" = OrderedDict()
        self._lock = threading.RLock()
        self._on_load_start: list = []
        self._on_load_finish: list = []
        self._on_evict: list = []

    @classmethod
    def instance(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ---------- настройки ----------

    def set_max_loaded(self, n: int) -> None:
        with self._lock:
            self._max_loaded = max(1, n)
            self._evict_if_needed()

    def get_max_loaded(self) -> int:
        return self._max_loaded

    # ---------- наблюдатели ----------

    def on_load_start(self, cb) -> None:
        self._on_load_start.append(cb)

    def on_load_finish(self, cb) -> None:
        self._on_load_finish.append(cb)

    def on_evict(self, cb) -> None:
        self._on_evict.append(cb)

    def off_load_start(self, cb) -> None:
        try:
            self._on_load_start.remove(cb)
        except ValueError:
            pass

    def off_load_finish(self, cb) -> None:
        try:
            self._on_load_finish.remove(cb)
        except ValueError:
            pass

    def _emit(self, listeners: list, *args) -> None:
        for cb in list(listeners):
            try:
                cb(*args)
            except Exception:
                pass

    # ---------- основная логика ----------

    def get_model(
        self,
        path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        mmproj_path: str = "",
        vision_handler: str = "",
    ) -> "LlamaType":
        """
        Возвращает прогретую Llama-инстанцию. Если её нет — загружает.

        Если переданы mmproj_path + vision_handler — создаёт chat_handler для Vision.
        Иначе грузит как обычную текстовую.
        """
        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama-cpp-python не установлен")

        key = (path, n_ctx, mmproj_path or "", vision_handler or "")

        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key]
            self._evict_if_needed(reserve=1)

        self._emit(self._on_load_start, path)

        # Vision-handler (опционально)
        chat_handler = None
        if mmproj_path and vision_handler:
            handler_cls = _VISION_HANDLERS.get(vision_handler)
            if handler_cls is None:
                err = (
                    f"Vision-handler '{vision_handler}' не найден в установленной версии "
                    f"llama-cpp-python. Доступно: {', '.join(available_vision_handlers()) or 'ничего'}"
                )
                self._emit(self._on_load_finish, path, False, err)
                raise RuntimeError(err)
            if not os.path.exists(mmproj_path):
                err = f"mmproj не найден: {mmproj_path}"
                self._emit(self._on_load_finish, path, False, err)
                raise FileNotFoundError(err)
            try:
                chat_handler = handler_cls(clip_model_path=mmproj_path, verbose=False)
            except Exception as e:
                self._emit(self._on_load_finish, path, False, f"Ошибка загрузки mmproj: {e}")
                raise

        try:
            kwargs = dict(
                model_path=path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            if chat_handler is not None:
                kwargs["chat_handler"] = chat_handler
            model = Llama(**kwargs)
        except Exception as e:
            self._emit(self._on_load_finish, path, False, str(e))
            raise

        with self._lock:
            if key in self._models:
                self._safe_close(model)
                self._models.move_to_end(key)
                self._emit(self._on_load_finish, path, True, None)
                return self._models[key]
            self._evict_if_needed(reserve=1)
            self._models[key] = model

        self._emit(self._on_load_finish, path, True, None)
        return model

    def unload(self, path: str) -> None:
        """Выгружает все варианты модели с этим путём."""
        with self._lock:
            to_remove = [k for k in self._models if k[0] == path]
            for k in to_remove:
                model = self._models.pop(k)
                self._safe_close(model)
                self._emit(self._on_evict, k[0])

    def unload_all(self) -> None:
        with self._lock:
            while self._models:
                _, model = self._models.popitem(last=False)
                self._safe_close(model)

    def loaded(self) -> list[str]:
        with self._lock:
            return [k[0] for k in self._models]

    # ---------- внутренние ----------

    def _evict_if_needed(self, reserve: int = 0) -> None:
        while len(self._models) + reserve > self._max_loaded and self._models:
            key, model = self._models.popitem(last=False)
            self._safe_close(model)
            self._emit(self._on_evict, key[0])

    @staticmethod
    def _safe_close(model) -> None:
        try:
            del model
        except Exception:
            pass
