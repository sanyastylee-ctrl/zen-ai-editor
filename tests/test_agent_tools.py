from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ai.agent import AgentWorker, parse_tool_response
from ai.agent import strip_tool_blocks
from core import app_data
from core.profiles import AIProfile, ChatTemplate, ProfileKind
from core.tools import ToolCall
from core.tools.write import WriteFileTool


def agent_profile() -> AIProfile:
    return AIProfile(
        id="agent",
        name="Agent",
        kind=ProfileKind.CODER,
        model_file="fake.gguf",
        chat_template=ChatTemplate.CHATML,
        n_ctx=8192,
        max_tokens=128,
        agent_mode=True,
    )


class ScriptedModel:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def __call__(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        text = self.outputs.pop(0) if self.outputs else "Done."
        return iter([{"choices": [{"text": text}]}])


class FakeManager:
    def __init__(self, model):
        self.model = model

    def on_load_start(self, cb): pass
    def on_load_finish(self, cb): pass
    def off_load_start(self, cb): pass
    def off_load_finish(self, cb): pass
    def get_model(self, **kwargs): return self.model


class AgentToolFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_logs = app_data.LOGS_DIR
        app_data.LOGS_DIR = self.root / "logs"

    def tearDown(self):
        app_data.LOGS_DIR = self.old_logs
        self.tmp.cleanup()

    def run_worker(self, outputs: list[str], **kwargs):
        model = ScriptedModel(outputs)
        worker = AgentWorker(
            agent_profile(),
            "do work",
            project_root=str(self.root),
            max_agent_steps=kwargs.pop("max_agent_steps", 5),
            **kwargs,
        )
        chunks: list[str] = []
        finished: list[dict] = []
        worker.chunk_received.connect(chunks.append)
        worker.tool_finished.connect(finished.append)
        worker._needs_confirmation = lambda tool: False
        with mock.patch("ai.agent.LLAMA_AVAILABLE", True), \
             mock.patch("ai.agent.resolve_model_path", return_value=str(self.root / "fake.gguf")), \
             mock.patch("ai.agent.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.agent.acceleration_warning", return_value=""):
            worker.run()
        return model, chunks, finished

    def test_create_and_write_tools_execute_in_order_and_feed_result_back(self):
        first = """
```xml
<tool name="create_file">
<path>hello.py</path>
<content>print('created')
</content>
</tool>
<tool name="write_file">
<path>hello.py</path>
<content>print('hi')
</content>
</tool>
```
"""
        model, _, results = self.run_worker([first, "Done."], max_agent_steps=2)

        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), "print('hi')\n")
        self.assertEqual([item["name"] for item in results], ["create_file", "write_file"])
        self.assertIn("Tool result for write_file", model.prompts[1])
        self.assertIn("[ok: wrote hello.py", model.prompts[1])

    def test_edit_and_patch_tools_modify_existing_file_in_order(self):
        (self.root / "hello.py").write_text("print('hi')\n", encoding="utf-8")
        response = """
<tool name="edit_file">
<path>hello.py</path>
<old_str>print('hi')</old_str>
<new_str>print('hello')</new_str>
</tool>
<tool name="patch_file">
<patch>
--- a/hello.py
+++ b/hello.py
@@ -1 +1 @@
-print('hello')
+print('hello world')
</patch>
</tool>
"""
        _, _, results = self.run_worker([response, "Done."], max_agent_steps=2)

        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), "print('hello world')\n")
        self.assertEqual([item["name"] for item in results], ["edit_file", "patch_file"])
        self.assertTrue(all(item["ok"] for item in results))

    def test_run_command_alias_returns_terminal_output_to_loop(self):
        (self.root / "hello.py").write_text("print('terminal-ok')\n", encoding="utf-8")
        response = """
<tool name="run_command">
<command>python hello.py</command>
</tool>
"""
        model, _, results = self.run_worker([response, "Done."], max_agent_steps=2)

        self.assertEqual(results[0]["name"], "run_command")
        self.assertIn("terminal-ok", results[0]["output"])
        self.assertIn("[exit 0]", results[0]["output"])
        self.assertIn("terminal-ok", model.prompts[1])

    def test_malformed_tool_call_is_visible_logged_and_not_executed(self):
        model, chunks, results = self.run_worker([
            "<tool name=\"write_file\"><path>bad.py</path><content>oops",
            "I could not complete the tool call.",
        ], max_agent_steps=2)

        self.assertFalse((self.root / "bad.py").exists())
        self.assertTrue(any(item["name"] == "parser_error" and not item["ok"] for item in results))
        self.assertIn("Ошибка разбора tool call", "".join(chunks))
        self.assertIn("Tool parser error", model.prompts[1])
        self.assertIn("agent_parser_error", (app_data.LOGS_DIR / "zenai.log").read_text(encoding="utf-8"))

    def test_write_without_content_is_rejected_instead_of_creating_empty_file(self):
        _, chunks, _ = self.run_worker([
            "<tool name=\"write_file\"><path>empty.py</path></tool>",
            "Fixed after parser feedback.",
        ], max_agent_steps=2)

        self.assertFalse((self.root / "empty.py").exists())
        self.assertIn("missing <content>", "".join(chunks))

    def test_truncated_generation_keeps_visible_partial_text(self):
        class PartialModel:
            def __call__(self, prompt: str, **kwargs):
                return iter([
                    {"choices": [{"text": "Начал ответ"}]},
                    {"choices": [{"text": " который не должен появиться"}]},
                ])

        model = PartialModel()
        worker = AgentWorker(
            agent_profile(), "answer", project_root=str(self.root), max_generation_seconds=1
        )
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)
        with mock.patch("ai.agent.LLAMA_AVAILABLE", True), \
             mock.patch("ai.agent.resolve_model_path", return_value=str(self.root / "fake.gguf")), \
             mock.patch("ai.agent.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.agent.acceleration_warning", return_value=""), \
             mock.patch("ai.agent.time.monotonic", side_effect=[0, 0, 2]):
            worker.run()

        rendered = "".join(chunks)
        self.assertIn("Начал ответ", rendered)
        self.assertNotIn("который не должен появиться", rendered)
        self.assertIn("частичный ответ сохранён", rendered)
        self.assertIn("agent_generation_stopped", (app_data.LOGS_DIR / "zenai.log").read_text(encoding="utf-8"))

    def test_legacy_saved_prompt_action_still_writes_content(self):
        calls, errors = parse_tool_response(
            "[CREATE_FILE: legacy.py]\nprint('legacy')\n[/CREATE_FILE]"
        )
        self.assertEqual(errors, [])
        result = WriteFileTool(str(self.root)).execute(calls[0])
        self.assertTrue(result.ok)
        self.assertEqual((self.root / "legacy.py").read_text(encoding="utf-8"), "print('legacy')\n")

    def test_write_content_is_preserved_literally(self):
        calls, errors = parse_tool_response(
            "<tool name=\"write_file\"><path>page.html</path>"
            "<content>&lt;tag&gt; &amp; literal</content></tool>"
        )
        self.assertEqual(errors, [])
        result = WriteFileTool(str(self.root)).execute(calls[0])
        self.assertTrue(result.ok)
        self.assertEqual(
            (self.root / "page.html").read_text(encoding="utf-8"),
            "&lt;tag&gt; &amp; literal",
        )

    def test_inline_tool_mention_is_not_parser_error(self):
        text = "Я использовал инструмент `<tool name=\"write_file\">` для записи файла."
        calls, errors = parse_tool_response(text)

        self.assertEqual(calls, [])
        self.assertEqual(errors, [])
        self.assertIn("<tool name=\"write_file\">", strip_tool_blocks(text))

    def test_path_traversal_is_rejected(self):
        outside = self.root.parent / "outside.py"
        result = WriteFileTool(str(self.root)).execute(
            ToolCall(name="write_file", args={"path": "../outside.py", "content": "bad"})
        )
        self.assertFalse(result.ok)
        self.assertIn("path outside project", result.output)
        self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()
