"""
ModelManager — держит одновременно несколько моделей в VRAM, переключается без перезагрузки.

Стратегия:
- max_loaded моделей одновременно (по умолчанию 2: coder + companion).
- При попытке загрузить ещё одну выгружаем наименее недавно использованную.
- get_model() возвращает уже прогретую модель за миллисекунды.
- Сравнение по (path, n_ctx) — если юзер сменил n_ctx, это другая инстанция.

Загрузка модели делается синхронно в вызывающем потоке. Это значит, что для UI
её нужно вызывать из QThread (см. ai/worker.py), иначе UI зависнет на старте.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

try:
    from llama_cpp import Llama
    # Импортируем обработчик для Vision моделей (проекторов mmproj)
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    LLAMA_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore
    Llava15ChatHandler = None
    LLAMA_AVAILABLE = False


if TYPE_CHECKING:
    from llama_cpp import Llama as LlamaType


class ModelManager:
    """
    Singleton-кэш моделей. Использовать через ModelManager.instance().
    """

    _instance: "ModelManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, max_loaded: int = 1) -> None:
        self._max_loaded = max_loaded
        self._models: "OrderedDict[tuple[str, int], LlamaType]" = OrderedDict()
        self._lock = threading.RLock()
        # колбэки прогресса (status_text) — UI подписывается, чтобы показывать "Загружаю модель..."
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

    def _emit(self, listeners: list, *args) -> None:
        for cb in listeners:
            try:
                cb(*args)
            except Exception:
                # колбэки не должны крашить менеджер
                pass

    # ---------- основная логика ----------

    def get_model(
        self,
        path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
    ) -> "LlamaType":
        """
        Возвращает прогретую Llama-инстанцию. Если её нет — загружает.

        Вызывает блокирующую загрузку на 3-15 секунд в зависимости от размера
        модели. Поэтому вызывать из QThread.
        """
        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama-cpp-python не установлен")

        key = (path, n_ctx)

        with self._lock:
            # Hit — двигаем в конец (recently used)
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key]

            # Miss — нужно загрузить, возможно выгрузив самый старый
            self._evict_if_needed(reserve=1)

        # Загрузка вне lock — она долгая, не хотим блокировать остальных
        self._emit(self._on_load_start, path)

        # Подключение модуля зрения (Vision)
        chat_handler = None
        if Llava15ChatHandler is not None:
            model_dir = os.path.dirname(path)
            if os.path.exists(model_dir):
                # Ищем любой файл, содержащий "mmproj" (проектор) в той же папке
                for f in os.listdir(model_dir):
                    if "mmproj" in f.lower() and f.lower().endswith(".gguf"):
                        mmproj_path = os.path.join(model_dir, f)
                        self._emit(self._on_load_start, f"Загрузка модуля зрения (mmproj)...")
                        try:
                            # Инициализируем зрение
                            chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)
                        except Exception as e:
                            print(f"[ModelManager] Ошибка загрузки mmproj: {e}")
                        break

        try:
            # Передаем chat_handler внутрь Llama. 
            # Если он None, модель загрузится как обычная текстовая.
            model = Llama(
                model_path=path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                chat_handler=chat_handler,
                verbose=False,
            )
        except Exception as e:
            self._emit(self._on_load_finish, path, False, str(e))
            raise

        with self._lock:
            # Кто-то мог загрузить ту же модель пока мы её грузили — приоритет первому
            if key in self._models:
                # Мы зря потратили время, но не страшно. Новую выбрасываем.
                self._safe_close(model)
                self._models.move_to_end(key)
                self._emit(self._on_load_finish, path, True, None)
                return self._models[key]

            self._evict_if_needed(reserve=1)
            self._models[key] = model

        self._emit(self._on_load_finish, path, True, None)
        return model

    def unload(self, path: str, n_ctx: int | None = None) -> None:
        """Выгружает конкретную модель (или все с этим путём, если n_ctx не указан)."""
        with self._lock:
            to_remove = [
                k for k in self._models
                if k[0] == path and (n_ctx is None or k[1] == n_ctx)
            ]
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
        """Под локом. Выгружает LRU пока не освободится место под reserve новых."""
        while len(self._models) + reserve > self._max_loaded and self._models:
            key, model = self._models.popitem(last=False)  # LRU = первый
            self._safe_close(model)
            self._emit(self._on_evict, key[0])

    @staticmethod
    def _safe_close(model) -> None:
        """llama-cpp не имеет публичного close, но мы можем подсказать GC."""
        try:
            # llama_cpp.Llama имеет __del__, который освобождает контекст
            del model
        except Exception:
            pass