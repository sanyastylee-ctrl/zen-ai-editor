import json
import tempfile
import urllib.request
from PyQt6.QtCore import QThread, pyqtSignal

def _build_comfyui_workflow(positive: str, negative: str = "", steps: int = 20, cfg: float = 7.0, width: int = 512, height: int = 512) -> dict:
    return {
        "prompt": {
            "1": { "class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.ckpt"} },
            "2": { "class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]} },
            "3": { "class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]} },
            "4": { "class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1} },
            "5": { "class_type": "KSampler", "inputs": { "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "seed": 42, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0 } },
            "6": { "class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]} },
            "7": { "class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "zen_ai"} }
        }
    }

class ComfyUIWorker(QThread):
    image_ready    = pyqtSignal(str)
    status_signal  = pyqtSignal(str)
    error_signal   = pyqtSignal(str)

    def __init__(self, base_url: str, positive: str, negative: str = "", steps: int = 20, cfg: float = 7.0):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.positive = positive
        self.negative = negative
        self.steps    = steps
        self.cfg      = cfg

    def _post(self, path: str, data: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=10) as r:
            return json.loads(r.read())

    def run(self):
        try:
            self.status_signal.emit("🎨 ComfyUI: отправка промпта...")
            wf = _build_comfyui_workflow(self.positive, self.negative, self.steps, self.cfg)
            resp = self._post("/prompt", wf)
            prompt_id = resp.get("prompt_id")
            if not prompt_id:
                self.error_signal.emit("ComfyUI не вернул prompt_id")
                return

            self.status_signal.emit("🎨 ComfyUI: генерация...")
            for _ in range(120):
                self.msleep(1000)
                history = self._get(f"/history/{prompt_id}")
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    for node_out in outputs.values():
                        images = node_out.get("images", [])
                        if images:
                            img_info = images[0]
                            url = f"{self.base_url}/view?filename={img_info['filename']}&subfolder={img_info.get('subfolder', '')}&type={img_info.get('type', 'output')}"
                            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="zen_comfyui_")
                            with urllib.request.urlopen(url, timeout=30) as r:
                                tmp.write(r.read())
                            tmp.close()
                            self.image_ready.emit(tmp.name)
                            return
            self.error_signal.emit("ComfyUI: таймаут (120 сек)")
        except Exception as e:
            self.error_signal.emit(f"ComfyUI ошибка: {e}")