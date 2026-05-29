from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ai.agent import AgentWorker, sanitize_agent_history
from ai.vision import VisionWorker
from core import app_data
from core.profiles import AIProfile, ChatTemplate, ProfileKind
from core.tools import ToolCall, ToolResult


def agent_profile() -> AIProfile:
    return AIProfile(
        id="agent",
        name="Agent",
        kind=ProfileKind.CODER,
        model_file="fake.gguf",
        chat_template=ChatTemplate.CHATML,
        n_ctx=8192,
        max_tokens=256,
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


class CoderAgentBenchmarks(unittest.TestCase):
    """Fast fake-model benchmark gates for Coder Agent v2 behavior."""

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

    def run_worker(self, outputs: list[str], user_message: str, **kwargs):
        model = ScriptedModel(outputs)
        worker = AgentWorker(
            agent_profile(),
            user_message,
            project_root=str(self.root),
            max_agent_steps=kwargs.pop("max_agent_steps", 8),
            max_tool_calls=kwargs.pop("max_tool_calls", 20),
            **kwargs,
        )
        chunks: list[str] = []
        results: list[dict] = []
        worker.chunk_received.connect(chunks.append)
        worker.tool_finished.connect(results.append)
        worker._needs_confirmation = lambda tool, call=None: False
        with mock.patch("ai.agent.LLAMA_AVAILABLE", True), \
             mock.patch("ai.agent.resolve_model_path", return_value=str(self.root / "fake.gguf")), \
             mock.patch("ai.agent.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.agent.acceleration_warning", return_value=""):
            worker.run()
        return model, chunks, results, worker

    def assert_project_safe(self):
        self.assertFalse((self.root / ".zen_ai").exists())
        self.assertFalse((self.root / "models").exists())

    def test_01_create_small_cli_project_from_scratch(self):
        write_main = """
<tool name="write_file"><path>main.py</path><content>import json, sys
path = "notes.json"
def load():
    try:
        return json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        return []
def save(items):
    json.dump(items, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
cmd = sys.argv[1]
items = load()
if cmd == "add":
    items.append({"text": sys.argv[2], "done": False}); save(items)
elif cmd == "list":
    [print(f"{i+1}. {x['text']}") for i, x in enumerate(items)]
elif cmd == "done":
    items[int(sys.argv[2])-1]["done"] = True; save(items)
elif cmd == "clear":
    save([])
</content></tool>
"""
        write_readme = """
<tool name="write_file"><path>README.md</path><content># Task notes
python main.py add "first task"
python main.py list
python main.py done 1
python main.py clear
</content></tool>
"""
        commands = [
            '<tool name="run_terminal"><command>python main.py add "first task"</command></tool>',
            '<tool name="run_terminal"><command>python main.py list</command></tool>',
            '<tool name="run_terminal"><command>python main.py done 1</command></tool>',
            '<tool name="run_terminal"><command>python main.py clear</command></tool>',
            "Готово: CLI создан и проверен.",
        ]
        _, chunks, results, worker = self.run_worker(
            [write_main, write_readme, *commands],
            'Создай CLI. Проверь:\npython main.py add "first task"\npython main.py list\npython main.py done 1\npython main.py clear',
        )

        self.assertTrue((self.root / "main.py").exists())
        self.assertTrue((self.root / "README.md").exists())
        self.assertTrue((self.root / "notes.json").exists())
        self.assertEqual(
            set(worker._command_goals_done),
            {
                'python main.py add "first task"',
                "python main.py list",
                "python main.py done 1",
                "python main.py clear",
            },
        )
        self.assertTrue(all("[exit 0]" in item["output"] for item in results if item["name"] == "run_terminal"))
        self.assertNotIn("лимиту контекста", "".join(chunks))
        self.assertIsNone(worker.continuation_state)
        self.assert_project_safe()

    def test_02_edit_existing_multi_file_project(self):
        (self.root / "app").mkdir()
        (self.root / "app" / "controller.py").write_text("VALUE = 'old'\n", encoding="utf-8")
        (self.root / "app" / "model.py").write_text("NAME = 'old'\n", encoding="utf-8")
        outputs = [
            '<tool name="read_file"><path>app/controller.py</path></tool>',
            '<tool name="edit_file"><path>app/controller.py</path><old_str>VALUE = \'old\'</old_str><new_str>VALUE = \'new\'</new_str></tool>',
            '<tool name="read_file"><path>app/model.py</path></tool>',
            '<tool name="edit_file"><path>app/model.py</path><old_str>NAME = \'old\'</old_str><new_str>NAME = \'new\'</new_str></tool>',
            "Готово.",
        ]
        _, _, _, worker = self.run_worker(outputs, "измени controller.py и model.py")

        self.assertIn("VALUE = 'new'", (self.root / "app" / "controller.py").read_text(encoding="utf-8"))
        self.assertIn("NAME = 'new'", (self.root / "app" / "model.py").read_text(encoding="utf-8"))
        self.assertEqual(worker._verified_files(), ["app/controller.py", "app/model.py"])

    def test_03_fix_bug_from_traceback(self):
        (self.root / "main.py").write_text("print(1/0)\n", encoding="utf-8")
        outputs = [
            '<tool name="read_file"><path>main.py</path></tool>',
            '<tool name="edit_file"><path>main.py</path><old_str>print(1/0)</old_str><new_str>print(1)</new_str></tool>',
            '<tool name="run_terminal"><command>python main.py</command></tool>',
            "Исправил падение.",
        ]
        _, _, results, _ = self.run_worker(outputs, "исправь traceback ZeroDivisionError и выполни python main.py")

        self.assertIn("print(1)", (self.root / "main.py").read_text(encoding="utf-8"))
        self.assertIn("[exit 0]", "\n".join(item["output"] for item in results))

    def test_04_add_tests_and_make_them_pass(self):
        (self.root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        outputs = [
            '<tool name="read_file"><path>calc.py</path></tool>',
            '<tool name="edit_file"><path>calc.py</path><old_str>return a - b</old_str><new_str>return a + b</new_str></tool>',
            '<tool name="write_file"><path>tests/test_calc.py</path><content>from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n</content></tool>',
            '<tool name="run_terminal"><command>python -m pytest</command></tool>',
            "Тесты проходят.",
        ]
        _, _, results, _ = self.run_worker(outputs, "почини calc.py, добавь тесты и запусти pytest")

        self.assertTrue((self.root / "tests" / "test_calc.py").exists())
        self.assertIn("[exit 0]", "\n".join(item["output"] for item in results))

    def test_05_install_dependency_requires_confirmation(self):
        worker = AgentWorker(agent_profile(), "установи requests", project_root=str(self.root), confirmation_policy="auto_confirm")
        call = ToolCall(name="run_terminal", args={"command": "pip install requests"})

        self.assertTrue(worker._needs_confirmation(worker._tools["run_terminal"], call))

    def test_06_git_status_and_diff_are_safe_terminal_checks(self):
        worker = AgentWorker(agent_profile(), "покажи git status", project_root=str(self.root), confirmation_policy="auto_confirm")

        self.assertFalse(worker._needs_confirmation(worker._tools["run_terminal"], ToolCall(name="run_terminal", args={"command": "git status"})))
        self.assertFalse(worker._needs_confirmation(worker._tools["run_terminal"], ToolCall(name="run_terminal", args={"command": "git diff"})))

    def test_07_continue_after_limit_uses_checkpoint(self):
        first = self.run_worker(
            ['<tool name="write_file"><path>main.py</path><content>print("a")</content></tool>'],
            "создай main.py и app/model.py",
            max_agent_steps=1,
            max_context_chars=200000,
        )[3]
        self.assertIsNotNone(first.continuation_state)
        resumed = self.run_worker(
            ['<tool name="write_file"><path>app/model.py</path><content>class Model: pass</content></tool>', "Готово."],
            "продолжай",
            continuation_state=first.continuation_state,
            max_context_chars=200000,
        )[3]

        self.assertTrue((self.root / "app" / "model.py").exists())
        self.assertIsNone(resumed.continuation_state)

    def test_08_avoid_repeating_same_file_read(self):
        (self.root / "main.py").write_text("print('ok')\n", encoding="utf-8")
        repeated = '<tool name="read_file"><path>main.py</path></tool>'
        _, _, results, _ = self.run_worker([repeated, repeated, repeated, "Готово."], "прочитай main.py")

        self.assertTrue(any("same file content as previous read" in item["output"] for item in results))

    def test_09_never_use_assistant_prose_as_task(self):
        history = [("создай app/controller.py", "План:\n```python\nprint('fake')\n```\nTool result for read_file: fake")]
        _, _, _, worker = self.run_worker(
            ['<tool name="write_file"><path>app/controller.py</path><content>print("real")</content></tool>', "Готово."],
            "давай",
            history=history,
        )

        self.assertEqual(worker.resolved_task, "создай app/controller.py")
        self.assertIn("real", (self.root / "app" / "controller.py").read_text(encoding="utf-8"))

    def test_10_finish_with_clean_summary(self):
        _, chunks, _, worker = self.run_worker(
            ['<tool name="write_file"><path>main.py</path><content>print("ok")</content></tool>', "Готово: main.py создан."],
            "создай main.py",
        )

        self.assertIn("Готово", "".join(chunks))
        self.assertIsNone(worker.continuation_state)

    def test_11_image_only_question_routes_to_vision_not_coder_tools(self):
        from ui.main_window import ZenEditor
        window = ZenEditor.__new__(ZenEditor)
        window.attached_files = []
        window._agent_continuation_by_profile = {}

        self.assertTrue(ZenEditor._is_vision_only_request(window, "Что на скрине?"))
        self.assertFalse(ZenEditor._has_agent_intent(window, "Что на скрине?", "agent"))

    def test_12_image_fix_gives_visual_context_then_coder_tools(self):
        (self.root / "ui.py").write_text("COLOR = 'red'\n", encoding="utf-8")
        _, _, _, worker = self.run_worker(
            [
                '<tool name="read_file"><path>ui.py</path></tool>',
                '<tool name="edit_file"><path>ui.py</path><old_str>COLOR = \'red\'</old_str><new_str>COLOR = \'green\'</new_str></tool>',
                "Исправил по скриншоту.",
            ],
            "исправь это по скрину",
            visual_context="visible_summary: button color is wrong\nlikely_files_or_components: ui.py",
        )

        self.assertIn("green", (self.root / "ui.py").read_text(encoding="utf-8"))
        self.assertIn("ui.py", "\n".join(worker._initial_transcript()))

    def test_13_no_image_with_vision_assist_behaves_as_regular_coder(self):
        from ui.main_window import ZenEditor
        profile = agent_profile()
        profile.enable_vision_assist = True
        profile.vision_model_file = "vision.gguf"
        profile.mmproj_file = "mmproj.gguf"
        profile.vision_handler = "qwen25vl"
        window = ZenEditor.__new__(ZenEditor)

        self.assertFalse(ZenEditor._should_run_vision_assist(window, profile, []))

    def test_14_vision_output_not_used_as_user_task(self):
        worker = AgentWorker(agent_profile(), "продолжай", project_root=str(self.root), visual_context="visible_summary: old plan")

        self.assertEqual(worker.resolved_task, "")
        self.assertTrue(worker._requires_clarification)

    def test_15_visual_context_is_compressed_in_continuation(self):
        worker = AgentWorker(agent_profile(), "исправь по скрину", project_root=str(self.root), visual_context="x" * 10000)
        worker._save_continuation_state("test", ["tail"])

        self.assertLess(len(worker.visual_context), 6100)
        self.assertLess(len(worker.continuation_state["summary"]["visual_context"]), 1300)

    def test_16_vision_worker_compacts_context_and_is_read_only(self):
        image = self.root / "screen.png"
        image.write_bytes(b"fake")
        profile = agent_profile()
        profile.vision_model_file = "vision.gguf"
        profile.mmproj_file = "mmproj.gguf"
        profile.vision_handler = "qwen25vl"
        profile.max_visual_context_chars = 1000
        worker = VisionWorker(profile, "исправь это", [str(image)])

        self.assertLessEqual(len(worker._compact("x" * 5000)), 1030)
        self.assertFalse(hasattr(worker, "_tools"))


if __name__ == "__main__":
    unittest.main()
