import threading
import gc

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

class ModelManager:
    _lock = threading.Lock()
    _model = None
    _model_path = ""

    @classmethod
    def get_model(cls, path, n_ctx: int = 4096):
        with cls._lock:
            if cls._model_path != path or cls._model is None:
                if cls._model is not None:
                    del cls._model
                    cls._model = None
                    gc.collect()
                cls._model = Llama(
                    model_path=path, n_gpu_layers=-1, n_ctx=n_ctx, verbose=False
                )
                cls._model_path = path
            return cls._model