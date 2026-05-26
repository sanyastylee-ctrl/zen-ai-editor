from PyQt6.QtCore import QThread, pyqtSignal
from ai.model_manager import ModelManager, Llama

class LlamaCppWorker(QThread):
    chunk_received  = pyqtSignal(str)
    status_signal   = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, prompt, code_context, model_path,
                 system_prompt="", coder_system_prompt="",
                 temperature=0.7, max_tokens=2048, mode="coder"):
        super().__init__()
        self.prompt              = prompt
        self.code_context        = code_context
        self.model_path          = model_path
        self.system_prompt       = system_prompt
        self.coder_system_prompt = coder_system_prompt
        self.temperature         = temperature
        self.max_tokens          = max_tokens
        self.mode                = mode
        self._stop_requested     = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        if Llama is None:
            self.chunk_received.emit("\n[Ошибка: llama-cpp-python не установлен]\n")
            self.finished_signal.emit()
            return
        try:
            model = ModelManager.get_model(self.model_path)
            mn = self.model_path.lower()

            if self.mode == "coder":
                sys_p    = self.coder_system_prompt or "Ты — опытный программист."
                user_msg = f"Контекст кода:\n{self.code_context}\n\nЗапрос: {self.prompt}"
            else:
                sys_p    = self.system_prompt
                user_msg = self.prompt
                if self.code_context:
                    user_msg += f"\n\nПрикреплённые данные:\n{self.code_context}"

            if "llama-3" in mn or "llama3" in mn:
                fp = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_p}<|eot_id|>"
                      f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
                      f"<|start_header_id|>assistant<|end_header_id|>\n\n")
            elif "mistral" in mn or "mixtral" in mn:
                fp = f"<s>[INST] {sys_p}\n\n{user_msg} [/INST]"
            elif "gemma" in mn:
                fp = (f"<start_of_turn>user\n{sys_p}\n\n{user_msg}<end_of_turn>\n"
                      f"<start_of_turn>model\n")
            else:
                fp = (f"<|im_start|>system\n{sys_p}<|im_end|>\n"
                      f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                      f"<|im_start|>assistant\n")

            output = model(fp, max_tokens=self.max_tokens,
                           temperature=self.temperature, stream=True)
            for chunk in output:
                if self._stop_requested:
                    self.chunk_received.emit("\n<i style='color:#888888;'>[Генерация остановлена]</i>")
                    break
                text = chunk['choices'][0]['text']
                if text: self.chunk_received.emit(text)
        except Exception as e:
            self.chunk_received.emit(f"\n[Критическая ошибка: {str(e)}]\n")
        finally:
            self.finished_signal.emit()