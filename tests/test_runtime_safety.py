from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ai.agent import AgentWorker
from core import app_data, diagnostics
from core.model_manager import ModelManager
from core.profiles import AIProfile, ChatTemplate, ProfileKind


def coder_profile(**overrides) -> AIProfile:
    data = dict(
        id="coder",
        name="Coder",
        kind=ProfileKind.CODER,
        model_file="coder.gguf",
        chat_template=ChatTemplate.CHATML,
        n_ctx=8192,
        n_gpu_layers=-1,
        max_tokens=64,
        temperature=0.1,
        top_p=0.9,
        top_k=20,
        repeat_penalty=1.05,
        agent_mode=True,
    )
    data.update(overrides)
    return AIProfile(**data)


class RuntimeSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_agent_loop_stops_at_max_steps(self):
        class FakeModel:
            def __call__(self, *args, **kwargs):
                return iter([
                    {"choices": [{"text": '<tool name="list_files"><path>.</path></tool>'}]}
                ])

        class FakeManager:
            def on_load_start(self, cb): pass
            def on_load_finish(self, cb): pass
            def off_load_start(self, cb): pass
            def off_load_finish(self, cb): pass
            def get_model(self, **kwargs): return FakeModel()

        with tempfile.TemporaryDirectory() as td:
            Path(td, "file.txt").write_text("x", encoding="utf-8")
            worker = AgentWorker(
                coder_profile(),
                "inspect",
                project_root=td,
                max_agent_steps=2,
                max_tool_calls=10,
                max_generation_seconds=30,
            )
            chunks: list[str] = []
            worker.chunk_received.connect(chunks.append)

            with mock.patch("ai.agent.LLAMA_AVAILABLE", True), \
                 mock.patch("ai.agent.resolve_model_path", return_value=str(Path(td) / "coder.gguf")), \
                 mock.patch("ai.agent.ModelManager.instance", return_value=FakeManager()), \
                 mock.patch("ai.agent.acceleration_warning", return_value=""):
                worker.run()

            self.assertIn("лимит итераций", "".join(chunks))

    def test_agent_completion_stops_on_generation_timeout(self):
        class SlowModel:
            def __call__(self, *args, **kwargs):
                return iter([{"choices": [{"text": "late"}]}])

        worker = AgentWorker(
            coder_profile(),
            "timeout",
            max_generation_seconds=1,
        )
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)

        with mock.patch("ai.agent.time.monotonic", side_effect=[0, 2]):
            result = worker._complete(SlowModel(), ["task"])

        self.assertEqual(result, "")
        self.assertTrue(worker._stop)
        self.assertEqual(worker._generation_interrupted_reason, "превышен лимит времени генерации")
        self.assertEqual(chunks, [])

    def test_model_manager_passes_n_gpu_layers_to_llama_and_cache_key(self):
        manager = ModelManager(max_loaded=3)
        created = [object(), object()]
        with mock.patch("core.model_manager.LLAMA_AVAILABLE", True), \
             mock.patch("core.model_manager.log_runtime_diagnostics"), \
             mock.patch("core.model_manager.Llama", side_effect=created) as llama:
            first = manager.get_model("model.gguf", n_ctx=4096, n_gpu_layers=0)
            second = manager.get_model("model.gguf", n_ctx=4096, n_gpu_layers=35)

        self.assertIs(first, created[0])
        self.assertIs(second, created[1])
        self.assertEqual(llama.call_args_list[0].kwargs["n_gpu_layers"], 0)
        self.assertEqual(llama.call_args_list[1].kwargs["n_gpu_layers"], 35)

    def test_diagnostics_create_log_file(self):
        with tempfile.TemporaryDirectory() as td:
            old_logs = app_data.LOGS_DIR
            app_data.LOGS_DIR = Path(td) / "logs"
            try:
                diagnostics.log_runtime_diagnostics(
                    profile=coder_profile(n_gpu_layers=0),
                    model_path=str(Path(td) / "models" / "coder.gguf"),
                    source="test",
                )
                log_path = app_data.LOGS_DIR / "zenai.log"
                self.assertTrue(log_path.exists())
                text = log_path.read_text(encoding="utf-8")
                self.assertIn("n_gpu_layers=0", text)
                self.assertIn("runtime_diagnostics", text)
            finally:
                app_data.LOGS_DIR = old_logs

    def test_diagnostics_work_with_frozen_paths(self):
        with tempfile.TemporaryDirectory() as td:
            old_logs = app_data.LOGS_DIR
            app_data.LOGS_DIR = Path(td) / "logs"
            exe = Path(td) / "ZenAI" / "ZenAI.exe"
            try:
                with mock.patch.object(diagnostics.sys, "frozen", True, create=True), \
                     mock.patch.object(diagnostics.sys, "executable", str(exe)):
                    data = diagnostics.log_runtime_diagnostics(
                        profile=coder_profile(),
                        model_path=str(exe.parent / "models" / "coder.gguf"),
                        source="frozen-test",
                    )
                self.assertEqual(data["models_dir"], str(exe.parent / "models"))
                self.assertTrue((app_data.LOGS_DIR / "zenai.log").exists())
            finally:
                app_data.LOGS_DIR = old_logs


if __name__ == "__main__":
    unittest.main()
