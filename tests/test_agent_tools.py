from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ai.agent import AgentWorker, parse_tool_response, sanitize_agent_history, sanitize_agent_assistant_text
from ai.agent import strip_tool_blocks
from ai.coder_agent import (
    AgentPhase,
    AgentRunStateV3,
    CommandGoal,
    CoderAgentController,
    FileGoal,
    FileGoalStatus,
    TaskLedger,
    TaskLedgerItem,
    TaskStatus,
    TaskType,
    build_project_map,
    detect_lazy_placeholders,
    evaluate_final_readiness,
    extract_file_goals,
    normalize_command,
    parse_traceback,
    verify_file_goal,
)
from ai.coder_agent.guards import final_with_pending_goals_guard
from ai.worker import InferenceWorker
from core import app_data
from core.profiles import AIProfile, ChatTemplate, ProfileKind
from core.tools import ToolCall, ToolResult
from core.tools.edit import EditFileTool
from core.tools.read import ReadFileTool
from core.tools.term import RunTerminalTool
from core.tools.write import WriteFileTool
from ui.agent_progress import AgentProgressOverlay


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


class ScriptedVisionModel:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.messages = []

    def create_chat_completion(self, messages, **kwargs):
        self.messages.append(messages)
        return iter([
            {"choices": [{"delta": {"content": chunk}}]}
            for chunk in self.chunks
        ])


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
        patch_confirmation = kwargs.pop("patch_confirmation", True)
        worker = AgentWorker(
            agent_profile(),
            kwargs.pop("user_message", "do work"),
            project_root=str(self.root),
            max_agent_steps=kwargs.pop("max_agent_steps", 5),
            **kwargs,
        )
        chunks: list[str] = []
        finished: list[dict] = []
        worker.chunk_received.connect(chunks.append)
        worker.tool_finished.connect(finished.append)
        if patch_confirmation:
            worker._needs_confirmation = lambda tool, call=None: False
        with mock.patch("ai.agent.LLAMA_AVAILABLE", True), \
             mock.patch("ai.agent.resolve_model_path", return_value=str(self.root / "fake.gguf")), \
             mock.patch("ai.agent.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.agent.acceleration_warning", return_value=""):
            worker.run()
        return model, chunks, finished, worker

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
        model, _, results, _ = self.run_worker([first, "Done."], max_agent_steps=2)

        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), "print('hi')\n")
        self.assertEqual(
            [item["name"] for item in results],
            ["create_file", "read_file", "write_file", "read_file"],
        )
        self.assertIn("Tool result for write_file", model.prompts[1])
        self.assertIn("[ok: wrote hello.py", model.prompts[1])
        self.assertIn("Verification result for read_file", model.prompts[1])

    def test_edit_and_patch_tools_modify_existing_file_in_order(self):
        (self.root / "hello.py").write_text("print('hi')\n", encoding="utf-8")
        response = """
<tool name="read_file">
<path>hello.py</path>
</tool>
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
        _, _, results, _ = self.run_worker([response, "Done."], max_agent_steps=2)

        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), "print('hello world')\n")
        self.assertEqual(
            [item["name"] for item in results],
            ["read_file", "edit_file", "read_file", "patch_file", "read_file"],
        )
        self.assertTrue(all(item["ok"] for item in results))

    def test_edit_file_preserves_base_indent_for_multiline_replacement(self):
        (self.root / "calculator.py").write_text(
            "def divide(a, b):\n    return a / b\n",
            encoding="utf-8",
        )
        tool = EditFileTool(str(self.root))

        result = tool.execute(
            ToolCall(
                name="edit_file",
                args={
                    "path": "calculator.py",
                    "old_str": "return a / b",
                    "new_str": "if b == 0:\n    return \"Error: Division by zero\"\nelse:\n    return a / b",
                },
            )
        )

        self.assertTrue(result.ok, result.output)
        self.assertEqual(
            (self.root / "calculator.py").read_text(encoding="utf-8"),
            (
                "def divide(a, b):\n"
                "    if b == 0:\n"
                "        return \"Error: Division by zero\"\n"
                "    else:\n"
                "        return a / b\n"
            ),
        )

    def test_run_command_alias_returns_terminal_output_to_loop(self):
        (self.root / "hello.py").write_text("print('terminal-ok')\n", encoding="utf-8")
        response = """
<tool name="run_command">
<command>python hello.py</command>
</tool>
"""
        model, _, results, _ = self.run_worker(
            [response, "Done."],
            user_message="выполни python hello.py",
            max_agent_steps=2,
        )

        self.assertEqual(results[0]["name"], "run_command")
        self.assertIn("terminal-ok", results[0]["output"])
        self.assertIn("[exit 0]", results[0]["output"])
        self.assertIn("terminal-ok", model.prompts[1])

    def test_explicit_run_request_falls_back_to_terminal_when_model_omits_tool(self):
        (self.root / "hello.py").write_text("print('fallback-ok')\n", encoding="utf-8")

        model, _, results, _ = self.run_worker(
            ["I will run it.", "Done."],
            user_message="выполни python hello.py",
            max_agent_steps=2,
        )

        self.assertEqual(results[0]["name"], "run_terminal")
        self.assertIn("fallback-ok", results[0]["output"])
        self.assertIn("Tool result for run_terminal", model.prompts[1])

    def test_action_intent_markdown_only_retries_then_executes_write_file(self):
        markdown_only = """План:
1. Создам файл.
2. Проверю результат.

```python
print('hi')
```
"""
        write = """
<tool name="write_file">
<path>hello.py</path>
<content>print('hi')
</content>
</tool>
"""

        model, chunks, results, _ = self.run_worker(
            [markdown_only, write, "Готово."],
            user_message="создай hello.py",
            max_agent_steps=4,
        )

        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), "print('hi')\n")
        self.assertEqual([item["name"] for item in results], ["write_file", "read_file"])
        self.assertIn("текст/markdown не меняет файлы", "".join(chunks))
        self.assertNotIn("```python", "".join(chunks))
        self.assertIn("Assistant response without required tool", model.prompts[1])

    def test_rejected_greeting_plan_is_not_rendered_before_tool_retry(self):
        greeting_plan = """Привет! План:
1. Создам hello.py.
2. Запишу код.

```python
print('hi')
```
"""
        write = """
<tool name="write_file">
<path>hello.py</path>
<content>print('hi')
</content>
</tool>
"""

        _, chunks, _, _ = self.run_worker(
            [greeting_plan, write, "Готово."],
            user_message="создай hello.py",
            max_agent_steps=4,
        )

        rendered = "".join(chunks)
        self.assertNotIn("Привет! План", rendered)
        self.assertNotIn("```python", rendered)
        self.assertIn("текст/markdown не меняет файлы", rendered)

    def test_davai_after_plan_requires_tools(self):
        history = [(
            "создай hello.py и запиши print('hi')",
            "План:\n1. Создам hello.py.\n2. Запишу print('hi').\nНапишите «давай», чтобы продолжить.",
        )]
        write = """
<tool name="write_file">
<path>hello.py</path>
<content>print('hi')
</content>
</tool>
"""

        _, _, results, worker = self.run_worker(
            [write, "Готово."],
            user_message="Давай",
            history=history,
            max_agent_steps=3,
        )

        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), "print('hi')\n")
        self.assertEqual(results[0]["name"], "write_file")
        self.assertEqual(worker.resolved_task, "создай hello.py и запиши print('hi')")
        self.assertTrue(AgentWorker.is_continue_request("Давай"))

    def test_agent_safe_history_removes_markdown_code_and_tool_results(self):
        dirty = """План:
1. Сделаю файл.

```python
print('this must not enter prompt')
```

Tool result for read_file:
# app/controller.py
Теперь, когда у нас есть содержимое файла app/controller.py, пожалуйста, уточните изменения.
"""

        cleaned = sanitize_agent_assistant_text(dirty)
        safe = sanitize_agent_history([("создай app/controller.py", dirty)])

        self.assertNotIn("```python", cleaned)
        self.assertNotIn("Tool result for", cleaned)
        self.assertNotIn("пожалуйста, уточните", cleaned.lower())
        self.assertIn("assistant proposed a plan", cleaned)
        self.assertEqual(safe, [("создай app/controller.py", cleaned)])

    def test_main_window_agent_safe_history_filters_restored_dirty_json_history(self):
        from ui.main_window import ZenEditor

        window = ZenEditor.__new__(ZenEditor)
        window._histories = {
            "coder": [
                (
                    "создай app/controller.py",
                    "План:\n1. Создам.\n```python\nbad()\n```\nTool result for read_file:\n# fake",
                )
            ]
        }

        safe = ZenEditor._agent_safe_history(window, "coder")
        joined = "\n".join(user + "\n" + assistant for user, assistant in safe)

        self.assertIn("создай app/controller.py", joined)
        self.assertNotIn("```python", joined)
        self.assertNotIn("Tool result for", joined)
        self.assertNotIn("# fake", joined)

    def test_agent_prompt_hard_guard_blocks_dirty_assistant_history(self):
        history = [(
            "создай app/controller.py",
            "План:\n1. Создам файл.\n```python\nprint('bad')\n```\n"
            "Tool result for read_file:\n# app/controller.py\n"
            "Пожалуйста, уточните изменения.",
        )]
        write = """
<tool name="write_file">
<path>app/controller.py</path>
<content>print('ok')
</content>
</tool>
"""

        model, _, _, worker = self.run_worker(
            [write, "Готово."],
            user_message="Давай",
            history=history,
        )

        prompt = model.prompts[0]
        self.assertEqual(worker.resolved_task, "создай app/controller.py")
        self.assertNotIn("```python", prompt)
        self.assertNotIn("Tool result for", prompt)
        self.assertNotIn("Пожалуйста, уточните", prompt)
        self.assertNotIn("План:\n1. Создам файл", prompt)
        self.assertIn("Previous result summary: [summary: assistant proposed a plan", prompt)

    def test_conversational_allo_does_not_trigger_agentworker(self):
        from ui.main_window import ZenEditor

        window = ZenEditor.__new__(ZenEditor)
        window.attached_files = []
        window._agent_continuation_by_profile = {}

        self.assertFalse(ZenEditor._has_agent_intent(window, "Алло?", "coder"))
        self.assertTrue(ZenEditor._has_agent_intent(window, "Проверь запуск через терминал.", "coder"))

    def test_profile_switcher_hides_vision_profiles_from_main_ui(self):
        from ui.profile_switcher import ProfileSwitcher

        coder = agent_profile()
        companion = AIProfile(
            id="companion",
            name="Лера",
            kind=ProfileKind.COMPANION,
            model_file="companion.gguf",
        )
        vision = AIProfile(
            id="vision",
            name="Глаза",
            kind=ProfileKind.VISION,
            model_file="vision.gguf",
        )
        switcher = ProfileSwitcher()

        switcher.set_profiles([coder, companion, vision], "vision")

        self.assertNotIn("vision", switcher._buttons)
        self.assertIn("agent", switcher._buttons)
        self.assertIn("companion", switcher._buttons)
        self.assertEqual(switcher.active_id(), "agent")

    def test_coder_profile_persists_vision_assist_capability(self):
        profile = agent_profile()
        profile.enable_vision_assist = True
        profile.vision_model_file = "qwen2.5-vl.gguf"
        profile.mmproj_file = "mmproj-qwen2.5-vl.gguf"
        profile.vision_handler = "qwen25vl"
        profile.max_visual_context_chars = 5000
        profile.vision_first_policy = "always"

        restored = AIProfile.from_dict(profile.to_dict())

        self.assertEqual(restored.kind, ProfileKind.CODER)
        self.assertTrue(restored.enable_vision_assist)
        self.assertEqual(restored.vision_model_file, "qwen2.5-vl.gguf")
        self.assertEqual(restored.mmproj_file, "mmproj-qwen2.5-vl.gguf")
        self.assertEqual(restored.vision_first_policy, "always")

    def test_profile_editor_exposes_coder_vision_assist_settings(self):
        from ui.profile_editor import ProfileEditor

        profile = agent_profile()
        with mock.patch("ui.profile_editor.list_available_models", return_value=[
            "qwen2.5-coder.gguf",
            "qwen2.5-vl.gguf",
            "mmproj-qwen2.5-vl.gguf",
        ]):
            editor = ProfileEditor(profile)

        self.assertTrue(hasattr(editor, "vision_assist_check"))
        self.assertTrue(hasattr(editor, "vision_model_combo"))
        self.assertTrue(hasattr(editor, "mmproj_combo"))
        editor.vision_assist_check.setChecked(True)
        editor.vision_model_combo.setCurrentText("qwen2.5-vl.gguf")
        editor.mmproj_combo.setCurrentText("mmproj-qwen2.5-vl.gguf")
        editor.max_visual_context_spin.setValue(5000)
        idx = editor.vision_policy_combo.findData("always")
        editor.vision_policy_combo.setCurrentIndex(idx)

        updated = editor.apply_to_profile()

        self.assertTrue(updated.enable_vision_assist)
        self.assertEqual(updated.vision_model_file, "qwen2.5-vl.gguf")
        self.assertEqual(updated.mmproj_file, "mmproj-qwen2.5-vl.gguf")
        self.assertEqual(updated.max_visual_context_chars, 5000)
        self.assertEqual(updated.vision_first_policy, "always")

    def test_vision_assist_routing_is_coder_capability(self):
        from ui.main_window import ZenEditor

        window = ZenEditor.__new__(ZenEditor)
        profile = agent_profile()
        image = str(self.root / "screen.png")

        self.assertFalse(ZenEditor._should_run_vision_assist(window, profile, [image]))
        profile.enable_vision_assist = True
        profile.vision_model_file = "qwen2.5-vl.gguf"
        profile.mmproj_file = "mmproj-qwen2.5-vl.gguf"
        profile.vision_handler = "qwen25vl"

        self.assertTrue(ZenEditor._should_run_vision_assist(window, profile, [image]))
        self.assertTrue(ZenEditor._is_vision_only_request(window, ""))
        self.assertTrue(ZenEditor._is_vision_only_request(window, "Что на скрине?"))
        self.assertFalse(ZenEditor._is_vision_only_request(window, "Исправь ошибку на скрине"))

    def test_vision_assist_does_not_forward_images_to_coder_worker(self):
        from ui.main_window import ZenEditor

        window = ZenEditor.__new__(ZenEditor)
        profile = agent_profile()
        captured = {}
        record = {}

        window._update_chat_record = lambda profile_id, rec: None
        window._persist_chat_session = lambda profile_id: None
        class FakeWorker:
            def start(self):
                captured["started"] = True

        def start_generation(**kwargs):
            captured.update(kwargs)
            window.worker = FakeWorker()

        window._start_generation_worker = start_generation
        image = str(self.root / "screen.png")
        other = str(self.root / "notes.txt")

        ZenEditor._on_vision_context_ready(
            window,
            profile=profile,
            full_text="Исправь ошибку на скрине",
            code_context="project tree",
            rag_snippets="",
            history=[],
            attached_paths=[image, other],
            context="visible_summary: traceback\nlikely_files_or_components: app.py",
            answer_mode=False,
            record=record,
        )

        self.assertEqual(captured["attached_paths"], [other])
        self.assertTrue(captured["started"])
        self.assertIn("Visual context from Vision Assist", captured["code_context"])
        self.assertIn("traceback", captured["visual_context"])

    def test_vision_assist_error_is_surfaced_in_tool_card(self):
        from ui.main_window import ZenEditor

        window = ZenEditor.__new__(ZenEditor)
        record = {}
        updated = {}
        window._update_chat_record = lambda profile_id, rec: updated.update(rec)
        window._persist_chat_session = lambda profile_id: None
        window.stop_btn = mock.Mock()
        window.worker = object()
        window._current_ai_response = "partial"
        window._current_message_buffer = "partial"
        window._current_response_profile_id = "coder"
        window._current_assistant_record = {"streaming": True}

        ZenEditor._on_vision_assist_error(
            window,
            "coder",
            record,
            "Vision Assist error: boom",
        )

        self.assertFalse(updated["ok"])
        self.assertEqual(updated["output"], "Vision analysis failed: boom")
        window.stop_btn.setEnabled.assert_called_once_with(False)
        self.assertIsNone(window.worker)
        self.assertEqual(window._current_ai_response, "")
        self.assertEqual(window._current_message_buffer, "")
        self.assertEqual(window._current_response_profile_id, "")
        self.assertIsNone(window._current_assistant_record)

    def test_coder_vision_without_context_does_not_start_agentworker(self):
        from ui.main_window import ZenEditor

        window = ZenEditor.__new__(ZenEditor)
        profile = agent_profile()
        record = {}
        started = []
        messages = []
        window._update_chat_record = lambda profile_id, rec: None
        window._persist_chat_session = lambda profile_id: None
        window._append_chat = lambda profile_id, html: messages.append(html)
        window._on_generation_done = lambda profile_id, text: None
        window._start_generation_worker = lambda **kwargs: started.append(kwargs)

        ZenEditor._on_vision_context_ready(
            window,
            profile=profile,
            full_text="Исправь ошибку на скрине",
            code_context="project tree",
            rag_snippets="",
            history=[],
            attached_paths=[str(self.root / "screen.png")],
            context="",
            answer_mode=False,
            record=record,
        )

        self.assertEqual(started, [])
        self.assertTrue(any("Vision Assist не смог" in message for message in messages))

    def test_agent_prompt_includes_visual_context_as_evidence_not_task(self):
        worker = AgentWorker(
            agent_profile(),
            "Исправь ошибку на скрине",
            project_root=str(self.root),
            visual_context="visible_summary: traceback in settings dialog\nlikely_files_or_components: ui/settings_dialog.py",
        )
        transcript = "\n".join(worker._initial_transcript())

        self.assertEqual(worker.resolved_task, "Исправь ошибку на скрине")
        self.assertIn("Visual context from Vision Assist", transcript)
        self.assertIn("not a user task", transcript)
        self.assertIn("ui/settings_dialog.py", transcript)

    def _make_send_window(self, profile: AIProfile, text: str, attachments: list[str] | None = None):
        from ui.main_window import ZenEditor

        class FakeLineEdit:
            def __init__(self, value: str):
                self._value = value
                self.cleared = False

            def text(self):
                return self._value

            def clear(self):
                self.cleared = True
                self._value = ""

        class FakeButton:
            def __init__(self):
                self.enabled = False

            def setEnabled(self, value):
                self.enabled = bool(value)

            def isEnabled(self):
                return self.enabled

        class FakeStatus:
            def __init__(self):
                self.text = ""

            def setText(self, value):
                self.text = value

        class FakeSwitcher:
            def active_id(self):
                return profile.id

        class FakeProjects:
            current = str(self.root)

        class FakeWorker:
            def __init__(self, running=False):
                self.running = running
                self.started = False

            def isRunning(self):
                return self.running

            def start(self):
                self.started = True

        window = ZenEditor.__new__(ZenEditor)
        window.chat_input = FakeLineEdit(text)
        window.stop_btn = FakeButton()
        window.model_status = FakeStatus()
        window.profile_switcher = FakeSwitcher()
        window.projects = FakeProjects()
        window.attached_files = list(attachments or [])
        window.worker = None
        window.app_settings = {"use_rag": False, "agent_confirmation_policy": "confirm_changes"}
        window._rag_chunks = 0
        window.rag = mock.Mock()
        window._histories = {profile.id: []}
        window._chat_records = {profile.id: []}
        window._terminal_history = []
        window._agent_continuation_by_profile = {}
        window._current_ai_response = ""
        window._current_message_buffer = ""
        window._current_response_profile_id = ""
        window._current_assistant_record = None
        window._current_assistant_sender = "Ассистент"
        window._current_assistant_kind = ""
        window._active_profile = lambda: profile
        window._append_chat_active = lambda html: None
        window._add_chat_record = lambda profile_id, record: window._chat_records.setdefault(profile_id, []).append(record)
        window._get_project_tree = lambda root: ""
        window._get_editor_text = lambda: ""
        window._update_attached_label = mock.Mock()
        window._load_attached_files = lambda: ""
        window._persist_chat_session = lambda profile_id: None
        return window, FakeWorker

    def _log_text(self) -> str:
        path = self.root / "logs" / "zenai.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def test_controller_phase_policy_blocks_tools_in_planning_phase(self):
        state = AgentRunStateV3("run", "task", "task")
        ledger = TaskLedger()
        controller = CoderAgentController(state, ledger)
        controller.set_phase(AgentPhase.BUILD_PLAN)

        decision = controller.validate_tool_call(ToolCall("read_file", {"path": "main.py"}, ""))

        self.assertTrue(decision.blocked)
        self.assertIn("not allowed", decision.reason)

    def test_controller_project_map_phase_allows_only_mapping_tools(self):
        state = AgentRunStateV3("run", "task", "task")
        controller = CoderAgentController(state, TaskLedger())
        controller.set_phase(AgentPhase.PROJECT_MAP)

        allowed = controller.validate_tool_call(ToolCall("list_files", {"path": "."}, ""))
        blocked = controller.validate_tool_call(ToolCall("write_file", {"path": "main.py", "content": ""}, ""))

        self.assertFalse(allowed.blocked)
        self.assertTrue(blocked.blocked)

    def test_controller_blocks_run_before_required_file_creation(self):
        state = AgentRunStateV3("run", "task", "task")
        ledger = TaskLedger([
            TaskLedgerItem(
                id="create-main",
                description="Create main.py",
                type=TaskType.CREATE_FILE,
                required_tool="write_file",
                target_file="main.py",
            ),
            TaskLedgerItem(
                id="cmd-1",
                description="Run app",
                type=TaskType.RUN_COMMAND,
                required_tool="run_terminal",
                command="python main.py",
            ),
        ])
        controller = CoderAgentController(state, ledger)
        controller.set_phase(AgentPhase.EXECUTE)

        decision = controller.validate_tool_call(ToolCall("run_terminal", {"command": "python main.py"}, ""))

        self.assertTrue(decision.blocked)
        self.assertIn("pending", decision.reason)

    def test_controller_blocks_unrelated_file_for_current_ledger_item(self):
        state = AgentRunStateV3("run", "task", "task")
        ledger = TaskLedger([
            TaskLedgerItem(
                id="edit-calc",
                description="Edit calculator",
                type=TaskType.EDIT_FILE,
                required_tool="apply_patch",
                target_file="app/calculator.py",
            )
        ])
        controller = CoderAgentController(state, ledger)
        controller.set_phase(AgentPhase.EXECUTE)

        decision = controller.validate_tool_call(
            ToolCall("edit_file", {"path": "README.md", "old_str": "a", "new_str": "b"}, "")
        )

        self.assertTrue(decision.blocked)
        self.assertIn("unrelated file", decision.reason)

    def test_controller_blocks_reading_done_file_when_current_file_goal_differs(self):
        state = AgentRunStateV3("run", "task", "task")
        ledger = TaskLedger([
            TaskLedgerItem(
                id="create-model",
                description="Create app/model.py",
                type=TaskType.CREATE_FILE,
                required_tool="write_file",
                target_file="app/model.py",
            )
        ])
        controller = CoderAgentController(state, ledger)
        controller.set_phase(AgentPhase.EXECUTE)

        decision = controller.validate_tool_call(ToolCall("read_file", {"path": "main.py"}, ""))

        self.assertTrue(decision.blocked)
        self.assertIn("read_file target does not match", decision.reason)

    def test_controller_marks_current_ledger_item_doing_only_after_allowed_tool(self):
        state = AgentRunStateV3("run", "task", "task")
        item = TaskLedgerItem(
            id="edit-calc",
            description="Edit calculator",
            type=TaskType.EDIT_FILE,
            required_tool="edit_file",
            target_file="app/calculator.py",
        )
        ledger = TaskLedger([item])
        controller = CoderAgentController(state, ledger)
        controller.set_phase(AgentPhase.EXECUTE)

        decision = controller.validate_tool_call(
            ToolCall("edit_file", {"path": "app/calculator.py", "old_str": "a", "new_str": "b"}, "")
        )

        self.assertFalse(decision.blocked)
        self.assertEqual(item.status, TaskStatus.DOING)

    def test_controller_model_prose_cannot_mark_ledger_done_or_allow_final(self):
        state = AgentRunStateV3("run", "task", "task")
        ledger = TaskLedger([
            TaskLedgerItem(id="create-main", description="Create main.py", type=TaskType.CREATE_FILE)
        ])
        controller = CoderAgentController(state, ledger)

        decision = controller.validate_final()

        self.assertTrue(decision.blocked)
        self.assertFalse(state.final_allowed)
        self.assertEqual(ledger.items[0].status, TaskStatus.TODO)

    def test_controller_blocks_out_of_order_command_goal(self):
        state = AgentRunStateV3("run", "task", "task")
        ledger = TaskLedger([
            TaskLedgerItem(
                id="cmd-1",
                description="Run add",
                type=TaskType.RUN_COMMAND,
                required_tool="run_terminal",
                command='python main.py add "first task"',
            )
        ])
        controller = CoderAgentController(state, ledger)
        controller.set_phase(AgentPhase.EXECUTE)

        decision = controller.validate_tool_call(ToolCall("run_terminal", {"command": "python main.py list"}, ""))

        self.assertTrue(decision.blocked)
        self.assertIn("out-of-order", decision.reason)

    def test_controller_blocks_repair_rerun_before_relevant_patch(self):
        state = AgentRunStateV3("run", "task", "task")
        ledger = TaskLedger([
            TaskLedgerItem(id="repair-1", description="Fix traceback", type=TaskType.FIX)
        ])
        controller = CoderAgentController(state, ledger)
        controller.set_phase(AgentPhase.REPAIR)

        decision = controller.validate_tool_call(
            ToolCall("run_terminal", {"command": "python main.py"}, ""),
            {"active_repair": {"failed_command": "python main.py", "touched_relevant": False}},
        )

        self.assertTrue(decision.blocked)
        self.assertIn("patch", decision.corrective.lower())

    def test_controller_blocks_duplicate_done_command_goal(self):
        state = AgentRunStateV3("run", "task", "task")
        state.command_goals_done = ["python main.py list"]
        controller = CoderAgentController(state, TaskLedger())
        controller.set_phase(AgentPhase.EXECUTE)

        decision = controller.validate_tool_call(ToolCall("run_terminal", {"command": "python main.py list"}, ""))

        self.assertTrue(decision.blocked)
        self.assertIn("already done", decision.reason)

    def test_agentworker_preflight_uses_controller_dispatch_policy(self):
        worker = AgentWorker(agent_profile(), "task", project_root=str(self.root))
        worker._controller.set_phase(AgentPhase.PROJECT_MAP)
        call = ToolCall("write_file", {"path": "main.py", "content": "print('x')\n"}, "")

        result = worker._preflight_tool_call(call, worker._tools["write_file"])

        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertTrue(result.meta["controller_blocked"])

    def test_verified_read_shortcut_still_respects_current_file_goal(self):
        (self.root / "app").mkdir()
        (self.root / "app" / "view.py").write_text("def display(message):\n    print(message)\n", encoding="utf-8")
        worker = AgentWorker(agent_profile(), "создай main.py app/view.py README.md", project_root=str(self.root))
        worker._ledger = TaskLedger([
            TaskLedgerItem(
                id="file-readme",
                description="create README.md",
                type=TaskType.CREATE_FILE,
                required_tool="write_file",
                target_file="README.md",
            )
        ])
        worker._controller = CoderAgentController(worker.run_state_v3, worker._ledger)
        worker._set_phase("execute_tools")
        worker._file_states["app/view.py"] = {
            "read_count": 1,
            "mutation_count": 1,
            "verified_after_mutation": True,
            "last_read_after_mutation": True,
            "done_candidate": True,
            "last_content_hash": "hash",
            "needs_reread": False,
        }

        result = worker._execute_tool(ToolCall("read_file", {"path": "app/view.py"}, ""))

        self.assertFalse(result.ok)
        self.assertTrue(result.meta["controller_blocked"])
        self.assertIn("README.md", result.output)

    def test_agent_progress_snapshot_emits_phase_and_ledger_state(self):
        worker = AgentWorker(
            agent_profile(),
            'создай CLI. Команда должна работать:\npython main.py add "first task"',
            project_root=str(self.root),
        )
        states: list[dict] = []
        phases: list[dict] = []
        worker.agent_state_updated.connect(states.append)
        worker.agent_phase_changed.connect(phases.append)

        worker._set_phase("execute_tools")

        self.assertTrue(states)
        self.assertTrue(phases)
        self.assertEqual(states[-1]["current_phase"], "execute_tools")
        self.assertGreaterEqual(states[-1]["total_steps"], 1)
        self.assertIn('python main.py add "first task"', states[-1]["command_goals"])

    def test_agent_progress_tool_start_finish_updates_current_tool(self):
        (self.root / "hello.py").write_text("print('hi')\n", encoding="utf-8")
        worker = AgentWorker(agent_profile(), "прочитай файл", project_root=str(self.root))
        snapshots: list[dict] = []
        worker.agent_state_updated.connect(snapshots.append)
        call = ToolCall(name="read_file", args={"path": "hello.py"}, raw="")

        result = worker._execute_tool(call)

        self.assertTrue(result.ok)
        self.assertTrue(any(item.get("event") == "tool_started" for item in snapshots))
        self.assertEqual(snapshots[-1]["event"], "tool_finished")
        self.assertEqual(snapshots[-1]["current_tool"], "read_file")

    def test_soft_limit_with_pending_ledger_creates_auto_continue_checkpoint(self):
        worker = AgentWorker(
            agent_profile(),
            'создай CLI. Проверь python main.py add "first task"',
            project_root=str(self.root),
        )
        events: list[dict] = []
        worker.agent_auto_continue.connect(events.append)

        ok = worker._request_auto_continue("достигнут лимит tool calls", ["transcript"])

        self.assertTrue(ok)
        self.assertTrue(worker.auto_continue_requested)
        self.assertIsNotNone(worker.continuation_state)
        self.assertEqual(worker.continuation_state["summary"]["auto_continue_count"], 1)
        self.assertTrue(events)

    def test_max_auto_continues_stops_with_blocker(self):
        profile = agent_profile()
        profile.max_auto_continues_per_task = 1
        worker = AgentWorker(
            profile,
            'создай CLI. Проверь python main.py add "first task"',
            project_root=str(self.root),
            continuation_state={
                "summary": {
                    "task": 'создай CLI. Проверь python main.py add "first task"',
                    "auto_continue_count": 1,
                    "command_goals": ['python main.py add "first task"'],
                }
            },
        )
        blocked: list[dict] = []
        worker.agent_blocked.connect(blocked.append)

        ok = worker._request_auto_continue("достигнут лимит tool calls", ["transcript"])

        self.assertFalse(ok)
        self.assertFalse(worker.auto_continue_requested)
        self.assertTrue(blocked)
        self.assertIn("лимит tool calls", blocked[-1]["blocker_reason"])

    def test_agent_progress_overlay_renders_snapshot_and_collapses(self):
        overlay = AgentProgressOverlay()
        snapshot = {
            "event": "state",
            "current_phase": "execute_tools",
            "current_step": "Запускаю проверку",
            "done_steps": 1,
            "total_steps": 3,
            "ledger": [
                {"status": "done", "description": "Создать main.py"},
                {"status": "doing", "description": "Проверить CLI"},
            ],
            "command_goals": ["python main.py list"],
            "command_goals_done": [],
        }

        overlay.update_state(snapshot)
        self.assertTrue(overlay.isVisible())
        self.assertIn("1/3", overlay.title_label.text())
        self.assertIn("Проверить CLI", overlay.details_label.text())
        overlay.details_btn.click()
        self.assertFalse(overlay.details_label.isVisible())

    def test_main_window_auto_continue_restarts_agent_without_history_append(self):
        from ui.main_window import ZenEditor

        class FakeTimer:
            def isActive(self):
                return False

            def stop(self):
                pass

        class FakeStop:
            def __init__(self):
                self.enabled = False

            def setEnabled(self, value):
                self.enabled = bool(value)

        class FakeProfileManager:
            def __init__(self, profile):
                self.profile = profile

            def get(self, profile_id):
                return self.profile if profile_id == self.profile.id else None

        class FakeWorker:
            def __init__(self):
                self.started = False

            def start(self):
                self.started = True

        profile = agent_profile()
        worker = AgentWorker(
            profile,
            "создай CLI",
            project_root=str(self.root),
            continuation_state={"summary": {"task": "создай CLI"}},
        )
        worker.auto_continue_requested = True
        worker.continuation_state = {
            "summary": {"task": "создай CLI", "auto_continue_count": 1},
            "reason": "достигнут лимит tool calls",
        }

        window = ZenEditor.__new__(ZenEditor)
        window.worker = worker
        window.pm = FakeProfileManager(profile)
        window._stream_update_timer = FakeTimer()
        window._current_assistant_record = None
        window._current_message_buffer = ""
        window._current_ai_response = ""
        window._histories = {profile.id: []}
        window._agent_continuation_by_profile = {}
        window._terminal_history = []
        window.stop_btn = FakeStop()
        window.projects = mock.Mock(current=str(self.root))
        window._current_response_profile_id = profile.id
        window._current_assistant_sender = "Ассистент"
        window._current_assistant_kind = profile.kind.value
        window._persist_chat_session = lambda profile_id: None
        window._capture_companion_memory = lambda profile_id, user_msg: None
        window._update_chat_record = lambda profile_id, record: None
        window._get_project_tree = lambda root: "project tree"
        started: dict = {}

        def start_generation(**kwargs):
            started.update(kwargs)
            window.worker = FakeWorker()

        window._start_generation_worker = start_generation

        ZenEditor._on_generation_done(window, profile.id, "создай CLI")

        self.assertEqual(window._histories[profile.id], [])
        self.assertEqual(started["full_text"], "создай CLI")
        self.assertEqual(started["code_context"], "project tree")
        self.assertTrue(window.worker.started)

    def test_send_text_triggers_route_log_and_enqueue(self):
        from ui.main_window import ZenEditor

        profile = agent_profile()
        profile.agent_mode = False
        window, FakeWorker = self._make_send_window(profile, "привет")
        captured = {}

        def start_generation(**kwargs):
            captured.update(kwargs)
            window.worker = FakeWorker()

        window._start_generation_worker = start_generation

        with mock.patch("ui.main_window.LLAMA_AVAILABLE", True):
            ZenEditor.send_message(window)

        log = self._log_text()
        self.assertIn("[ui_send_start]", log)
        self.assertIn('route="normal"', log)
        self.assertIn("[ui_send_finish_enqueue]", log)
        self.assertTrue(window.worker.started)
        self.assertEqual(captured["full_text"], "привет")

    def test_send_image_only_with_vision_assist_routes_to_vision(self):
        from ui.main_window import ZenEditor

        image = str(self.root / "screen.png")
        Path(image).write_bytes(b"png")
        profile = agent_profile()
        profile.enable_vision_assist = True
        profile.vision_model_file = "vision.gguf"
        profile.mmproj_file = "mmproj.gguf"
        profile.vision_handler = "qwen25vl"
        window, FakeWorker = self._make_send_window(profile, "", [image])
        captured = {}

        def load_attachments():
            self.assertEqual(window.attached_files, [image])
            return ""

        def start_vision(**kwargs):
            captured.update(kwargs)
            window.worker = FakeWorker()

        window._load_attached_files = load_attachments
        window._start_vision_assist = start_vision

        with mock.patch("ui.main_window.LLAMA_AVAILABLE", True):
            ZenEditor.send_message(window)

        log = self._log_text()
        self.assertIn('route="vision"', log)
        self.assertEqual(captured["image_paths"], [image])
        self.assertEqual(captured["attached_paths"], [image])
        self.assertEqual(window.attached_files, [])
        window._update_attached_label.assert_called_once()

    def test_send_image_plus_fix_routes_to_coder_vision(self):
        from ui.main_window import ZenEditor

        image = str(self.root / "screen.png")
        Path(image).write_bytes(b"png")
        profile = agent_profile()
        profile.enable_vision_assist = True
        profile.vision_model_file = "vision.gguf"
        profile.mmproj_file = "mmproj.gguf"
        profile.vision_handler = "qwen25vl"
        window, FakeWorker = self._make_send_window(profile, "исправь ошибку на скрине", [image])
        window._start_vision_assist = lambda **kwargs: setattr(window, "worker", FakeWorker())

        with mock.patch("ui.main_window.LLAMA_AVAILABLE", True):
            ZenEditor.send_message(window)

        self.assertIn('route="coder_vision"', self._log_text())

    def test_stale_worker_state_does_not_permanently_block_send(self):
        from ui.main_window import ZenEditor

        profile = agent_profile()
        profile.agent_mode = False
        window, FakeWorker = self._make_send_window(profile, "привет")
        stale_worker = FakeWorker(running=False)
        window.worker = stale_worker
        window._start_generation_worker = lambda **kwargs: setattr(window, "worker", FakeWorker())

        with mock.patch("ui.main_window.LLAMA_AVAILABLE", True):
            ZenEditor.send_message(window)

        log = self._log_text()
        self.assertIn('event="clear_stale_worker"', log)
        self.assertNotIn('reason="worker_busy"', log)
        self.assertTrue(window.worker.started)

    def test_attachments_are_not_cleared_when_send_is_blocked(self):
        from ui.main_window import ZenEditor

        image = str(self.root / "screen.png")
        Path(image).write_bytes(b"png")
        profile = agent_profile()
        profile.model_file = ""
        window, _ = self._make_send_window(profile, "привет", [image])

        with mock.patch("ui.main_window.LLAMA_AVAILABLE", True):
            ZenEditor.send_message(window)

        self.assertEqual(window.attached_files, [image])
        window._update_attached_label.assert_not_called()
        self.assertIn('reason="missing_model_file"', self._log_text())

    def test_vision_worker_builds_image_messages_and_compacts_context(self):
        from ai.vision import VisionWorker

        image = self.root / "screen.png"
        image.write_bytes(b"fake-png")
        profile = agent_profile()
        profile.vision_model_file = "qwen2.5-vl.gguf"
        profile.mmproj_file = "mmproj-qwen2.5-vl.gguf"
        profile.vision_handler = "qwen25vl"
        profile.max_visual_context_chars = 1000
        worker = VisionWorker(profile, "Что на скрине?", [str(image)], answer_mode=True)

        messages = worker._build_messages()
        content = messages[-1]["content"]

        self.assertEqual(messages[0]["role"], "system")
        self.assertTrue(any(item.get("type") == "image_url" for item in content))
        self.assertLessEqual(len(worker._compact("x" * 2000)), 1030)

    def test_vision_worker_success_emits_context_and_logs(self):
        from ai.vision import VisionWorker

        image = self.root / "screen.png"
        image.write_bytes(b"fake-png")
        profile = agent_profile()
        profile.vision_model_file = "qwen2.5-vl.gguf"
        profile.mmproj_file = "mmproj-qwen2.5-vl.gguf"
        profile.vision_handler = "qwen25vl"
        worker = VisionWorker(profile, "Что на скрине?", [str(image)], answer_mode=False)
        contexts = []
        errors = []
        worker.visual_context_ready.connect(contexts.append)
        worker.error_signal.connect(errors.append)

        with mock.patch("ai.vision.LLAMA_AVAILABLE", True), \
             mock.patch("ai.vision.resolve_model_path", side_effect=lambda name: str(self.root / name)), \
             mock.patch("ai.vision.ModelManager.instance", return_value=FakeManager(ScriptedVisionModel(["visible_summary: ok"]))):
            worker.run()

        self.assertEqual(contexts, ["visible_summary: ok"])
        self.assertEqual(errors, [])
        log = (app_data.LOGS_DIR / "zenai.log").read_text(encoding="utf-8")
        self.assertIn("[vision_start]", log)
        self.assertIn("[vision_model_load_begin]", log)
        self.assertIn("[vision_inference_end]", log)
        self.assertIn("[vision_visual_context_ready]", log)
        self.assertIn("[vision_finish]", log)

    def test_vision_worker_exception_surfaces_error_and_traceback(self):
        from ai.vision import VisionWorker

        class RaisingManager(FakeManager):
            def get_model(self, **kwargs):
                raise RuntimeError("load failed")

        image = self.root / "screen.png"
        image.write_bytes(b"fake-png")
        profile = agent_profile()
        profile.vision_model_file = "qwen2.5-vl.gguf"
        profile.mmproj_file = "mmproj-qwen2.5-vl.gguf"
        profile.vision_handler = "qwen25vl"
        worker = VisionWorker(profile, "Что на скрине?", [str(image)], answer_mode=True)
        errors = []
        chunks = []
        worker.error_signal.connect(errors.append)
        worker.chunk_received.connect(chunks.append)

        with mock.patch("ai.vision.LLAMA_AVAILABLE", True), \
             mock.patch("ai.vision.resolve_model_path", side_effect=lambda name: str(self.root / name)), \
             mock.patch("ai.vision.ModelManager.instance", return_value=RaisingManager(None)):
            worker.run()

        self.assertEqual(errors, ["Vision Assist error: load failed"])
        self.assertIn("Vision Assist error: load failed", "".join(chunks))
        log = (app_data.LOGS_DIR / "zenai.log").read_text(encoding="utf-8")
        self.assertIn("[vision_error]", log)
        self.assertIn("RuntimeError: load failed", log)

    def test_vision_worker_timeout_surfaces_error(self):
        from ai.vision import VisionWorker

        image = self.root / "screen.png"
        image.write_bytes(b"fake-png")
        profile = agent_profile()
        profile.vision_model_file = "qwen2.5-vl.gguf"
        profile.mmproj_file = "mmproj-qwen2.5-vl.gguf"
        profile.vision_handler = "qwen25vl"
        worker = VisionWorker(
            profile,
            "Что на скрине?",
            [str(image)],
            answer_mode=True,
            max_generation_seconds=1,
        )
        errors = []
        contexts = []
        worker.error_signal.connect(errors.append)
        worker.visual_context_ready.connect(contexts.append)

        with mock.patch("ai.vision.LLAMA_AVAILABLE", True), \
             mock.patch("ai.vision.resolve_model_path", side_effect=lambda name: str(self.root / name)), \
             mock.patch("ai.vision.ModelManager.instance", return_value=FakeManager(ScriptedVisionModel(["late"]))), \
             mock.patch("ai.vision.time.monotonic", side_effect=[0, 2]):
            worker.run()

        self.assertEqual(contexts, [])
        self.assertEqual(errors, ["Vision Assist error: Vision inference timeout"])
        log = (app_data.LOGS_DIR / "zenai.log").read_text(encoding="utf-8")
        self.assertIn("[vision_timeout]", log)
        self.assertIn("[vision_failed]", log)

    def test_inference_worker_can_disable_legacy_agent_file_actions(self):
        old_cwd = os.getcwd()
        os.chdir(self.root)
        try:
            worker = InferenceWorker(
                agent_profile(),
                "Что ты сделал?",
                allow_agent_actions=False,
            )
            worker._execute_agent_actions("[FILE: README.md]\n```markdown\nbad\n```")
        finally:
            os.chdir(old_cwd)

        self.assertFalse((self.root / "README.md").exists())

    def test_realizovyvai_after_plan_triggers_tools_without_raw_assistant_prose(self):
        from ui.main_window import ZenEditor

        window = ZenEditor.__new__(ZenEditor)
        window.attached_files = []
        window._agent_continuation_by_profile = {}
        self.assertTrue(ZenEditor._has_agent_intent(window, "Реализовывай", "coder"))

        history = [(
            "создай hello.py",
            "План:\n1. Создам hello.py.\n```python\nprint('bad')\n```",
        )]
        write = """
<tool name="write_file">
<path>hello.py</path>
<content>print('hi')
</content>
</tool>
"""

        model, _, results, worker = self.run_worker(
            [write, "Готово."],
            user_message="Реализовывай",
            history=history,
        )

        self.assertEqual(worker.resolved_task, "создай hello.py")
        self.assertEqual(results[0]["name"], "write_file")
        self.assertNotIn("```python", model.prompts[0])
        self.assertNotIn("План:\n1. Создам", model.prompts[0])

    def test_pdf_combiner_plan_then_davai_writes_real_files_and_finishes(self):
        task = "Сделай простой каркас PDF combiner: app/controller.py и main.py"
        history = [(
            task,
            "План:\n1. Создам app/controller.py.\n2. Создам main.py.\nНапишите «давай», чтобы продолжить.",
        )]
        write = """
<tool name="write_file">
<path>app/controller.py</path>
<content>class PdfController:
    pass
</content>
</tool>
"""
        write_main = """
<tool name="write_file">
<path>main.py</path>
<content>from app.controller import PdfController
</content>
</tool>
"""

        _, chunks, results, worker = self.run_worker(
            [write, write_main, "Готово: каркас создан и проверен."],
            user_message="Давай",
            history=history,
        )

        self.assertEqual(worker.resolved_task, task)
        self.assertIn("class PdfController", (self.root / "app" / "controller.py").read_text(encoding="utf-8"))
        self.assertIn("PdfController", (self.root / "main.py").read_text(encoding="utf-8"))
        self.assertEqual([item["name"] for item in results], ["write_file", "read_file", "write_file", "read_file"])
        self.assertIn("Готово: каркас создан", "".join(chunks))
        self.assertNotIn("опишите изменения", "".join(chunks).lower())
        self.assertIsNone(worker.continuation_state)

    def test_agent_prompt_does_not_pass_old_greeting_as_assistant_role(self):
        history = [("Привет", "Привет! Готов помочь."), ("создай hello.py", "План:\n1. Создам hello.py.\nНапишите «давай».")]
        write = """
<tool name="write_file">
<path>hello.py</path>
<content>print('hi')
</content>
</tool>
"""

        model, _, _, _ = self.run_worker(
            [write, "Готово."],
            user_message="Давай",
            history=history,
            max_agent_steps=3,
        )

        self.assertNotIn("<|im_start|>assistant\nПривет! Готов помочь.", model.prompts[0])
        self.assertNotIn("Привет! Готов помочь.", model.prompts[0])
        self.assertNotIn("План:\n1. Создам hello.py", model.prompts[0])
        self.assertIn("Current resolved user task:\nсоздай hello.py", model.prompts[0])

    def test_action_intent_markdown_code_without_tool_is_not_final(self):
        response = "Вот код:\n```python\nprint('hi')\n```"

        model, chunks, results, worker = self.run_worker(
            [response, response, response],
            user_message="создай hello.py",
            max_agent_steps=4,
        )

        self.assertFalse((self.root / "hello.py").exists())
        self.assertEqual(results, [])
        self.assertIn("модель вернула текст вместо tool call", worker.continuation_state["reason"])
        self.assertIn("Агент не выполнил действие", "".join(chunks))
        self.assertEqual(len(model.prompts), 3)

    def test_informational_question_can_answer_without_tools(self):
        model, chunks, results, worker = self.run_worker(
            ["Это просто объяснение без действий."],
            user_message="объясни код",
            max_agent_steps=3,
        )

        self.assertEqual(results, [])
        self.assertIn("объяснение", "".join(chunks))
        self.assertIsNone(worker.continuation_state)
        self.assertEqual(len(model.prompts), 1)

    def test_action_intent_requires_mutating_success_before_final(self):
        read = """
<tool name="read_file">
<path>hello.py</path>
</tool>
"""
        write = """
<tool name="write_file">
<path>hello.py</path>
<content>print('hi')
</content>
</tool>
"""

        model, chunks, results, _ = self.run_worker(
            [read, "Готово, файл создан.", write, "Теперь готово."],
            user_message="создай hello.py",
            max_agent_steps=5,
        )

        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), "print('hi')\n")
        self.assertEqual([item["name"] for item in results], ["read_file", "write_file", "read_file"])
        self.assertIn("текст/markdown не меняет файлы", "".join(chunks))
        self.assertIn("Assistant response without required tool", model.prompts[2])

    def test_malformed_tool_call_is_visible_logged_and_not_executed(self):
        model, chunks, results, _ = self.run_worker([
            "<tool name=\"write_file\"><path>bad.py</path><content>oops",
            "I could not complete the tool call.",
        ], max_agent_steps=2)

        self.assertFalse((self.root / "bad.py").exists())
        self.assertTrue(any(item["name"] == "parser_error" and not item["ok"] for item in results))
        self.assertIn("Ошибка разбора tool call", "".join(chunks))
        self.assertIn("Tool parser error", model.prompts[1])
        self.assertIn("agent_parser_error", (app_data.LOGS_DIR / "zenai.log").read_text(encoding="utf-8"))

    def test_read_file_returns_filesystem_content_and_logs_source(self):
        path = self.root / "app" / "controller.py"
        path.parent.mkdir()
        path.write_text("FILESYSTEM_UNIQUE = 42\n", encoding="utf-8")

        result = ReadFileTool(str(self.root)).execute(
            ToolCall(name="read_file", args={"path": "app/controller.py"})
        )

        self.assertTrue(result.ok)
        self.assertIn("FILESYSTEM_UNIQUE = 42", result.output)
        self.assertNotIn("Теперь, когда у нас есть содержимое", result.output)
        self.assertEqual(result.meta["source"], "filesystem")
        self.assertEqual(Path(result.meta["absolute_path"]).resolve(), path.resolve())
        log = (app_data.LOGS_DIR / "zenai.log").read_text(encoding="utf-8")
        self.assertIn("source=filesystem", log)
        self.assertIn("FILESYSTEM_UNIQUE = 42", log)

    def test_previous_assistant_text_never_becomes_resolved_task(self):
        history = [(
            "измени app/controller.py и добавь обработчик",
            "Теперь, когда у нас есть содержимое файла app/controller.py, пожалуйста, опишите изменения.",
        )]
        worker = AgentWorker(
            agent_profile(),
            "давай",
            project_root=str(self.root),
            history=history,
        )

        self.assertEqual(worker.resolved_task, "измени app/controller.py и добавь обработчик")
        self.assertNotIn("Теперь, когда", worker._initial_transcript()[0])

    def test_vague_coder_followup_without_active_task_clarifies_without_read(self):
        model, chunks, results, worker = self.run_worker(
            ["<tool name=\"read_file\"><path>app/controller.py</path></tool>"],
            user_message="ты же кодер, делай",
            history=[("Привет", "Привет! Готов помочь.")],
        )

        self.assertEqual(model.prompts, [])
        self.assertEqual(results, [])
        self.assertIn("Уточните конкретную задачу", "".join(chunks))
        self.assertIn("ожидается конкретная пользовательская задача", worker.continuation_state["reason"])

    def test_continue_after_completed_answer_does_not_repeat_last_mutation(self):
        model, chunks, results, worker = self.run_worker(
            ["<tool name=\"write_file\"><path>wrong.py</path><content>bad</content></tool>"],
            user_message="продолжай",
            history=[("создай hello.py", "Готово: файл создан и проверен.")],
        )

        self.assertEqual(model.prompts, [])
        self.assertEqual(results, [])
        self.assertFalse((self.root / "wrong.py").exists())
        self.assertIn("Уточните конкретную задачу", "".join(chunks))
        self.assertEqual(worker.resolved_task, "")

    def test_action_task_rejects_repeated_clarification_after_read(self):
        path = self.root / "app" / "controller.py"
        path.parent.mkdir()
        path.write_text("class Controller:\n    pass\n", encoding="utf-8")
        read = "<tool name=\"read_file\"><path>app/controller.py</path></tool>"
        clarify = "Теперь файл прочитан. Пожалуйста, опишите изменения, которые нужно сделать."

        model, chunks, results, worker = self.run_worker(
            [read, clarify, clarify],
            user_message="измени app/controller.py и добавь метод run",
        )

        self.assertEqual([item["name"] for item in results], ["read_file"])
        self.assertEqual(len(model.prompts), 3)
        self.assertIn("повторяющийся цикл", "".join(chunks))
        self.assertIn("повторяет уточнение", worker.continuation_state["reason"])

    def test_same_file_read_sequence_stops_without_infinite_loop(self):
        path = self.root / "app" / "controller.py"
        path.parent.mkdir()
        path.write_text("REAL_FILE = True\n", encoding="utf-8")
        read = "<tool name=\"read_file\"><path>app/controller.py</path></tool>"

        _, chunks, results, worker = self.run_worker(
            [read, read, read, read],
            user_message="покажи app/controller.py",
        )

        self.assertEqual([item["name"] for item in results], ["read_file", "read_file"])
        self.assertIn("повторяющийся цикл", "".join(chunks))
        self.assertIn("последовательность tools", worker.continuation_state["reason"])

    def test_read_result_remains_tool_evidence_not_user_task(self):
        path = self.root / "app" / "controller.py"
        path.parent.mkdir()
        path.write_text("value = 1\n", encoding="utf-8")
        read = "<tool name=\"read_file\"><path>app/controller.py</path></tool>"
        write = "<tool name=\"write_file\"><path>app/controller.py</path><content>value = 2\n</content></tool>"

        model, _, _, worker = self.run_worker(
            [read, write, "Готово."],
            user_message="измени app/controller.py и установи value = 2",
        )

        self.assertEqual(worker.resolved_task, "измени app/controller.py и установи value = 2")
        self.assertIn("Tool result for read_file:\n# app", model.prompts[1])
        self.assertNotIn("Current resolved user task:\n# app", model.prompts[1])

    def test_repeated_malformed_tool_call_stops_as_no_progress_cycle(self):
        bad = "<tool name=\"write_file\"><path>bad.py</path><content>oops"
        _, chunks, results, worker = self.run_worker(
            [bad, bad, bad, bad, bad],
            user_message="создай bad.py",
            max_agent_steps=1,
        )

        self.assertFalse((self.root / "bad.py").exists())
        self.assertTrue(any(item["name"] == "parser_error" for item in results))
        self.assertIn("Ошибка разбора tool call", "".join(chunks))
        self.assertIsNotNone(worker.continuation_state)
        self.assertIn("лимит итераций", worker.continuation_state["reason"])

    def test_write_without_content_is_rejected_instead_of_creating_empty_file(self):
        _, chunks, _, _ = self.run_worker([
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

    def test_edit_old_str_missing_but_new_str_present_is_idempotent_success(self):
        (self.root / "hello.py").write_text('print("hello world")\n', encoding="utf-8")

        result = EditFileTool(str(self.root)).execute(
            ToolCall(
                name="edit_file",
                args={
                    "path": "hello.py",
                    "old_str": 'print("hi")',
                    "new_str": 'print("hello world")',
                },
            )
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.meta.get("idempotent"))
        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), 'print("hello world")\n')

    def test_duplicate_successful_edit_file_is_not_executed_twice(self):
        (self.root / "hello.py").write_text('print("hi")\n', encoding="utf-8")
        first_edit = """
<tool name="read_file">
<path>hello.py</path>
</tool>
<tool name="edit_file">
<path>hello.py</path>
<old_str>print("hi")</old_str>
<new_str>print("hello world")</new_str>
</tool>
"""
        edit = """
<tool name="edit_file">
<path>hello.py</path>
<old_str>print("hi")</old_str>
<new_str>print("hello world")</new_str>
</tool>
"""

        model, chunks, results, worker = self.run_worker([first_edit, edit, "Done."], max_agent_steps=5)

        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), 'print("hello world")\n')
        self.assertEqual(sum(1 for item in results if item["name"] == "edit_file"), 2)
        edit_results = [item for item in results if item["name"] == "edit_file"]
        self.assertTrue(edit_results[0]["ok"])
        self.assertTrue(edit_results[1]["ok"])
        self.assertIn("duplicate", edit_results[1]["output"])
        self.assertNotIn("лимит итераций", "".join(chunks))
        self.assertNotIn("обнаружен повторяющийся цикл", "".join(chunks))
        self.assertIsNone(worker.continuation_state)

    def test_failed_edit_feedback_tells_model_to_read_before_retry(self):
        (self.root / "hello.py").write_text('print("different")\n', encoding="utf-8")
        bad_edit = """
<tool name="edit_file">
<path>hello.py</path>
<old_str>print("hi")</old_str>
<new_str>print("hello world")</new_str>
</tool>
"""

        model, _, results, _ = self.run_worker([bad_edit, "I will inspect first."], max_agent_steps=2)

        self.assertFalse(results[0]["ok"])
        self.assertIn("сначала read_file", results[0]["output"])
        self.assertIn("Before retrying a failed edit_file, call read_file", model.prompts[1])
        self.assertIn("Tool result for edit_file", model.prompts[1])

    def test_edit_file_without_read_file_is_blocked(self):
        (self.root / "hello.py").write_text('print("hi")\n', encoding="utf-8")
        response = """
<tool name="edit_file">
<path>hello.py</path>
<old_str>print("hi")</old_str>
<new_str>print("hello")</new_str>
</tool>
"""
        _, _, results, _ = self.run_worker([response, "I will read first."], max_agent_steps=2)

        self.assertFalse(results[0]["ok"])
        self.assertIn("сначала read_file", results[0]["output"])
        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), 'print("hi")\n')

    def test_edit_file_after_read_file_passes(self):
        (self.root / "hello.py").write_text('print("hi")\n', encoding="utf-8")
        response = """
<tool name="read_file">
<path>hello.py</path>
</tool>
<tool name="edit_file">
<path>hello.py</path>
<old_str>print("hi")</old_str>
<new_str>print("hello")</new_str>
</tool>
"""
        _, _, results, _ = self.run_worker([response, "Done."], max_agent_steps=2)

        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), 'print("hello")\n')
        self.assertTrue([item for item in results if item["name"] == "edit_file"][0]["ok"])

    def test_write_file_triggers_read_file_verification(self):
        response = """
<tool name="write_file">
<path>hello.py</path>
<content>print("hi")
</content>
</tool>
"""
        model, _, results, _ = self.run_worker([response, "Done."], max_agent_steps=2)

        self.assertEqual([item["name"] for item in results], ["write_file", "read_file"])
        self.assertTrue(results[1]["ok"])
        self.assertIn("# hello.py", results[1]["output"])
        self.assertIn("Verification result for read_file", model.prompts[1])

    def test_repeated_read_after_verified_write_is_blocked(self):
        write = """
<tool name="write_file">
<path>app/controller.py</path>
<content>class Controller:
    pass
</content>
</tool>
"""
        repeated_read = "<tool name=\"read_file\"><path>app/controller.py</path></tool>"

        model, _, results, worker = self.run_worker(
            [write, repeated_read, "Готово."],
            user_message="создай app/controller.py",
        )

        self.assertEqual(
            [item["name"] for item in results],
            ["write_file", "read_file", "read_file"],
        )
        self.assertIn("already verified", results[-1]["output"])
        self.assertTrue(results[-1]["meta"]["already_verified"])
        self.assertIn("[same file content as previous read; omitted]", model.prompts[2])
        self.assertIn("app/controller.py", worker._verified_files())

    def test_repeated_same_file_plan_after_verified_write_is_corrected_not_final(self):
        write = """
<tool name="write_file">
<path>app/controller.py</path>
<content>class Controller:
    pass
</content>
</tool>
"""
        repeated_plan = (
            "Проверю текущее состояние app/controller.py.\n"
            "Внесу изменение через write_file.\n"
            "Проверю результат."
        )

        model, chunks, _, _ = self.run_worker(
            [write, repeated_plan, "Готово: app/controller.py создан."],
            user_message="создай app/controller.py",
        )

        rendered = "".join(chunks)
        self.assertNotIn("Проверю текущее состояние app/controller.py", rendered)
        self.assertIn("Already changed and verified: app/controller.py", model.prompts[2])

    def test_repeated_same_read_output_is_omitted_from_transcript(self):
        (self.root / "app").mkdir()
        (self.root / "app" / "controller.py").write_text("VALUE = 1\n", encoding="utf-8")
        read = "<tool name=\"read_file\"><path>app/controller.py</path></tool>"

        model, _, results, _ = self.run_worker(
            [read, read, "Done."],
            user_message="прочитай app/controller.py",
        )

        self.assertEqual([item["name"] for item in results], ["read_file", "read_file"])
        self.assertIn("[same file content as previous read; omitted]", results[1]["output"])
        self.assertIn("[same file content as previous read; omitted]", model.prompts[2])
        self.assertEqual(model.prompts[2].count("Tool result for read_file:\n# app/controller.py\nVALUE = 1"), 1)

    def test_multifile_task_moves_from_controller_to_model(self):
        write_controller = """
<tool name="write_file">
<path>app/controller.py</path>
<content>class Controller:
    pass
</content>
</tool>
"""
        write_model = """
<tool name="write_file">
<path>app/model.py</path>
<content>class Model:
    pass
</content>
</tool>
"""

        _, chunks, results, worker = self.run_worker(
            [write_controller, write_model, "Готово."],
            user_message="создай app/controller.py и app/model.py",
        )

        self.assertTrue((self.root / "app" / "controller.py").exists())
        self.assertTrue((self.root / "app" / "model.py").exists())
        self.assertEqual(
            [item["name"] for item in results],
            ["write_file", "read_file", "write_file", "read_file"],
        )
        self.assertNotIn("повторяющийся цикл", "".join(chunks))
        self.assertEqual(worker._verified_files(), ["app/controller.py", "app/model.py"])

    def test_continue_after_context_limit_does_not_restart_verified_first_file(self):
        large = "x = '" + ("a" * 20000) + "'\n"
        write_controller = f"""
<tool name="write_file">
<path>app/controller.py</path>
<content>{large}</content>
</tool>
"""
        _, _, _, first = self.run_worker(
            [write_controller],
            user_message="создай app/controller.py и app/model.py",
            max_context_chars=15000,
        )

        self.assertIsNotNone(first.continuation_state)
        self.assertIn("app/controller.py", first.continuation_state["summary"]["verified_files"])

        read_controller = "<tool name=\"read_file\"><path>app/controller.py</path></tool>"
        write_model = """
<tool name="write_file">
<path>app/model.py</path>
<content>class Model:
    pass
</content>
</tool>
"""
        model = ScriptedModel([read_controller, write_model, "Готово."])
        resumed = AgentWorker(
            agent_profile(),
            "продолжай",
            project_root=str(self.root),
            continuation_state=first.continuation_state,
            max_context_chars=200000,
        )
        results: list[dict] = []
        resumed.tool_finished.connect(results.append)
        resumed._needs_confirmation = lambda tool, call=None: False
        with mock.patch("ai.agent.LLAMA_AVAILABLE", True), \
             mock.patch("ai.agent.resolve_model_path", return_value=str(self.root / "fake.gguf")), \
             mock.patch("ai.agent.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.agent.acceleration_warning", return_value=""):
            resumed.run()

        self.assertTrue((self.root / "app" / "model.py").exists())
        self.assertFalse(results[0]["ok"])
        self.assertIn("read_file target does not match current ledger item", results[0]["output"])
        self.assertIn("app/model.py", results[0]["output"])
        self.assertIn("app/model.py", resumed._verified_files())

    def test_cycle_guard_saves_continuation_state_and_continue_resumes(self):
        first_tool = '<tool name="list_files"><path>.</path></tool>'
        _, chunks, _, worker = self.run_worker(
            [first_tool, first_tool, first_tool, first_tool, first_tool],
            user_message="осмотри проект",
            max_agent_steps=1,
        )

        self.assertIn("лимиту итераций", "".join(chunks))
        self.assertIsNotNone(worker.continuation_state)
        self.assertIn("лимит итераций", worker.continuation_state["reason"])
        self.assertIn("summary", worker.continuation_state)

        model = ScriptedModel(["Продолжаю с сохранённого места."])
        resumed = AgentWorker(
            agent_profile(),
            "продолжай",
            project_root=str(self.root),
            continuation_state=worker.continuation_state,
            max_agent_steps=2,
        )
        resumed_chunks: list[str] = []
        resumed.chunk_received.connect(resumed_chunks.append)
        resumed._needs_confirmation = lambda tool, call=None: False
        with mock.patch("ai.agent.LLAMA_AVAILABLE", True), \
             mock.patch("ai.agent.resolve_model_path", return_value=str(self.root / "fake.gguf")), \
             mock.patch("ai.agent.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.agent.acceleration_warning", return_value=""):
            resumed.run()

        self.assertIn("Continue from the saved state", model.prompts[0])
        self.assertIn("Продолжаю", "".join(resumed_chunks))
        self.assertIsNone(resumed.continuation_state)

    def test_continuation_summary_contains_task_files_errors_and_next_step(self):
        response = """
<tool name="write_file">
<path>hello.py</path>
<content>print("hi")
</content>
</tool>
<tool name="edit_file">
<path>hello.py</path>
<old_str>missing</old_str>
<new_str>print("hello")</new_str>
</tool>
"""
        bad_edit = """
<tool name="edit_file">
<path>hello.py</path>
<old_str>missing</old_str>
<new_str>print("hello")</new_str>
</tool>
"""
        _, _, _, worker = self.run_worker(
            [response, bad_edit, bad_edit, bad_edit, bad_edit],
            user_message="создай hello.py",
            max_agent_steps=1,
        )

        summary = worker.continuation_state["summary"]
        self.assertEqual(summary["task"], "создай hello.py")
        self.assertIn("hello.py", summary["changed_files"])
        self.assertTrue(any("edit_file" in item for item in summary["errors"]))
        self.assertTrue(summary["next_step"])

    def test_duplicate_tool_call_is_not_repeated_after_continuation(self):
        (self.root / "hello.py").write_text('print("hi")\n', encoding="utf-8")
        first_edit = """
<tool name="read_file">
<path>hello.py</path>
</tool>
<tool name="edit_file">
<path>hello.py</path>
<old_str>print("hi")</old_str>
<new_str>print("hello world")</new_str>
</tool>
"""
        edit = """
<tool name="edit_file">
<path>hello.py</path>
<old_str>print("hi")</old_str>
<new_str>print("hello world")</new_str>
</tool>
"""
        _, _, _, first = self.run_worker([first_edit, edit, edit, edit, edit], max_agent_steps=1)
        self.assertIsNotNone(first.continuation_state)

        model = ScriptedModel([edit])
        resumed = AgentWorker(
            agent_profile(),
            "continue",
            project_root=str(self.root),
            continuation_state=first.continuation_state,
            max_agent_steps=5,
        )
        results: list[dict] = []
        chunks: list[str] = []
        resumed.tool_finished.connect(results.append)
        resumed.chunk_received.connect(chunks.append)
        resumed._needs_confirmation = lambda tool, call=None: False
        with mock.patch("ai.agent.LLAMA_AVAILABLE", True), \
             mock.patch("ai.agent.resolve_model_path", return_value=str(self.root / "fake.gguf")), \
             mock.patch("ai.agent.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.agent.acceleration_warning", return_value=""):
            resumed.run()

        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), 'print("hello world")\n')
        self.assertEqual(len(results), 1)
        self.assertIn("duplicate", results[0]["output"])
        self.assertNotIn("лимит итераций", "".join(chunks))
        self.assertIsNone(resumed.continuation_state)

    def test_auto_confirm_requires_confirmation_for_terminal_but_not_writes(self):
        worker = AgentWorker(
            agent_profile(),
            "run",
            project_root=str(self.root),
            confirmation_policy="auto_confirm",
        )

        safe = ToolCall(name="run_terminal", args={"command": "python hello.py"})
        install = ToolCall(name="run_terminal", args={"command": "pip install requests"})

        self.assertFalse(worker._needs_confirmation(worker._tools["run_terminal"], safe))
        self.assertTrue(worker._needs_confirmation(worker._tools["run_terminal"], install))
        self.assertFalse(worker._needs_confirmation(worker._tools["write_file"]))

    def test_safe_python_command_runs(self):
        (self.root / "hello.py").write_text("print('safe-ok')\n", encoding="utf-8")

        result = RunTerminalTool(str(self.root)).execute(
            ToolCall(name="run_terminal", args={"command": "python hello.py"})
        )

        self.assertTrue(result.ok)
        self.assertIn("safe-ok", result.output)
        self.assertIn("[exit 0]", result.output)

    def test_agent_blocks_direct_python_run_when_user_did_not_ask(self):
        (self.root / "hello.py").write_text("print('safe-ok')\n", encoding="utf-8")
        response = """
<tool name="run_terminal">
<command>python hello.py</command>
</tool>
"""
        _, _, results, _ = self.run_worker([response, "Done."], max_agent_steps=2)

        self.assertFalse(results[0]["ok"])
        self.assertIn("python <file>", results[0]["output"])

    def test_terminal_check_request_allows_direct_python_run(self):
        worker = AgentWorker(
            agent_profile(),
            "Проверь запуск через терминал.",
            project_root=str(self.root),
        )
        call = ToolCall(name="run_terminal", args={"command": "python main.py list"})

        self.assertTrue(worker._user_asked_to_run())
        self.assertIsNone(worker._preflight_tool_call(call, worker._tools["run_terminal"]))

    def test_done_word_completed_does_not_trigger_run_intent(self):
        worker = AgentWorker(
            agent_profile(),
            "done <id> отмечает заметку выполненной",
            project_root=str(self.root),
        )

        self.assertFalse(worker._detect_run_intent())
        self.assertFalse(worker._user_asked_to_run())

    def test_continue_preserves_terminal_check_intent(self):
        worker = AgentWorker(
            agent_profile(),
            "Продолжай, если что-то осталось.",
            project_root=str(self.root),
            continuation_state={
                "summary": {
                    "task": "Проверь запуск через терминал.",
                    "current_user_task": "Проверь запуск через терминал.",
                    "next_step": "Continue terminal verification.",
                },
                "tool_history": [],
                "successful_tool_keys": [],
            },
        )
        call = ToolCall(name="run_terminal", args={"command": "python main.py list"})

        self.assertEqual(worker.resolved_task, "Проверь запуск через терминал.")
        self.assertTrue(worker._user_asked_to_run())
        self.assertIsNone(worker._preflight_tool_call(call, worker._tools["run_terminal"]))

    def test_continuation_state_compacts_long_tool_errors_and_read_outputs(self):
        worker = AgentWorker(agent_profile(), "Проверь запуск через терминал.", project_root=str(self.root))
        long_error = "$ python main.py list\n" + ("traceback line\n" * 300) + "[exit 1]"
        worker._record_tool_call(
            ToolCall(name="run_terminal", args={"command": "python main.py list"}),
            ToolResult.error(long_error),
        )
        transcript = [
            "Tool result for read_file:\n" + ("content\n" * 500),
            "Tool result for run_terminal:\n" + long_error,
        ]

        worker._save_continuation_state("test context stop", transcript)
        state = worker.continuation_state

        self.assertIsNotNone(state)
        self.assertLess(len(state["summary"]["errors"][0]), 1500)
        self.assertIn("omitted", state["summary"]["errors"][0])
        self.assertIn("read_file content omitted", state["transcript_tail"][0])
        self.assertLess(len(state["transcript_tail"][1]), 2200)

    def test_duplicate_read_batch_stops_before_tool_call_limit(self):
        (self.root / "notes").mkdir()
        (self.root / "notes" / "notes.json").write_text("{}", encoding="utf-8")
        duplicate_reads = "\n".join(
            '<tool name="read_file"><path>notes/notes.json</path></tool>'
            for _ in range(8)
        )
        _, chunks, results, worker = self.run_worker(
            [
                '<tool name="write_file"><path>notes/notes.json</path><content>{}</content></tool>',
                duplicate_reads,
                "Готово.",
            ],
            user_message="создай notes/notes.json",
            max_agent_steps=4,
            max_tool_calls=16,
        )

        self.assertLess(len(results), 16)
        self.assertNotIn("лимиту tool calls", "".join(chunks))
        self.assertIsNone(worker.continuation_state)

    def test_plan_text_after_successful_tools_does_not_force_context_loop(self):
        _, chunks, _, worker = self.run_worker(
            [
                '<tool name="write_file"><path>main.py</path><content>print("ok")</content></tool>',
                "План:\n1. Подготовлю main.py.\n2. Проверю результат.",
            ],
            user_message="создай main.py",
            max_agent_steps=4,
        )

        text = "".join(chunks)
        self.assertIn("План:", text)
        self.assertNotIn("лимиту контекста", text)
        self.assertIsNone(worker.continuation_state)

    def test_python_edit_in_run_verification_auto_compiles(self):
        (self.root / "main.py").write_text("print('ok')\n", encoding="utf-8")
        response = """
<tool name="read_file"><path>main.py</path></tool>
<tool name="edit_file">
<path>main.py</path>
<old_str>print('ok')</old_str>
<new_str>if True print('broken')</new_str>
</tool>
"""
        _, _, results, _ = self.run_worker(
            [response, "Готово."],
            user_message="проверь запуск через терминал",
            max_agent_steps=2,
        )

        compile_outputs = [
            item["output"] for item in results
            if item["name"] == "run_terminal" and "py_compile" in item["output"]
        ]
        self.assertTrue(compile_outputs)
        self.assertIn("[exit 1]", compile_outputs[-1])

    def test_cli_task_extracts_required_command_goals_from_history(self):
        worker = AgentWorker(
            agent_profile(),
            "Проверь запуск через терминал.",
            project_root=str(self.root),
            history=[
                (
                    "Сделай небольшой Python-проект task notes CLI. "
                    "add <text> добавляет, list показывает, done <id> отмечает, clear удаляет.",
                    "План выполнен.",
                )
            ],
        )

        self.assertEqual(
            worker._command_goals,
            [
                "python main.py add",
                "python main.py list",
                "python main.py done",
                "python main.py clear",
            ],
        )
        self.assertTrue(worker._requires_command_success)
        self.assertFalse(worker._command_tool_succeeded)
        self.assertEqual(worker.run_state.access_mode, "confirm_access")
        self.assertEqual(len(worker.run_state.task_graph), 4)

    def test_py_compile_does_not_satisfy_cli_command_goals(self):
        worker = AgentWorker(
            agent_profile(),
            "Проверь запуск через терминал.",
            project_root=str(self.root),
            history=[
                (
                    "Сделай task notes CLI: add <text>, list, done <id>, clear в main.py.",
                    "Готово.",
                )
            ],
        )
        call = ToolCall(
            name="run_terminal",
            args={"command": "python -m py_compile main.py"},
        )

        worker._record_command_result(
            call,
            ToolResult(ok=True, output="$ python -m py_compile main.py\n[exit 0]"),
        )

        self.assertEqual(worker._command_goals_done, set())
        self.assertFalse(worker._command_tool_succeeded)
        self.assertIn("python main.py add", worker._pending_command_goal_text())

    def test_cli_command_goals_mark_done_only_for_matching_commands(self):
        worker = AgentWorker(
            agent_profile(),
            "Проверь запуск через терминал.",
            project_root=str(self.root),
            history=[
                (
                    "Сделай notes CLI: add <text>, list, done <id>, clear. Файл main.py.",
                    "Готово.",
                )
            ],
        )

        for command in (
            'python main.py add "first note"',
            "python main.py list",
            "python main.py done 1",
            "python main.py clear",
        ):
            output = f"$ {command}\n[exit 0]"
            if command == "python main.py list":
                output = "$ python main.py list\n1. first note\n[exit 0]"
            worker._record_command_result(
                ToolCall(name="run_terminal", args={"command": command}),
                ToolResult(ok=True, output=output),
            )

        self.assertEqual(set(worker._command_goals), worker._command_goals_done)
        self.assertTrue(worker._command_tool_succeeded)
        self.assertEqual(len(worker.run_state.command_history), 4)

    def test_literal_cli_goal_requires_exact_argument(self):
        worker = AgentWorker(
            agent_profile(),
            'Проверь команды:\npython main.py add "first task"\npython main.py list',
            project_root=str(self.root),
        )

        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": 'python main.py add "first note"'}),
            ToolResult(ok=True, output='$ python main.py add "first note"\n[exit 0]'),
        )

        self.assertIn('python main.py add "first task"', worker._command_goals)
        self.assertNotIn('python main.py add "first task"', worker._command_goals_done)
        self.assertFalse(worker._command_tool_succeeded)

    def test_todo_cli_list_must_show_added_task_before_goal_closes(self):
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app.\npython main.py add "first task"\npython main.py list',
            project_root=str(self.root),
        )

        add_call = ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'})
        list_call = ToolCall(name="run_terminal", args={"command": "python main.py list"})

        worker._record_command_result(
            add_call,
            ToolResult(ok=True, output='$ python main.py add "first task"\nTask added: "first task"\n[exit 0]'),
        )
        self.assertIn('python main.py add "first task"', worker._command_goals_done)

        list_result = ToolResult(ok=True, output="$ python main.py list\n\n[exit 0]")
        worker._record_command_result(list_call, list_result)

        self.assertNotIn("python main.py list", worker._command_goals_done)
        self.assertNotIn('python main.py add "first task"', worker._command_goals_done)
        self.assertFalse(worker._command_tool_succeeded)
        self.assertIn("verification failed", list_result.output)
        self.assertIn("first task", worker._pending_command_goal_text())

    def test_todo_cli_list_with_added_task_closes_goal(self):
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app.\npython main.py add "first task"\npython main.py list',
            project_root=str(self.root),
        )

        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'}),
            ToolResult(ok=True, output='$ python main.py add "first task"\nTask added: "first task"\n[exit 0]'),
        )
        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": "python main.py list"}),
            ToolResult(ok=True, output="$ python main.py list\n1. first task\n[exit 0]"),
        )

        self.assertIn('python main.py add "first task"', worker._command_goals_done)
        self.assertIn("python main.py list", worker._command_goals_done)

    def test_add_requires_notes_json_side_effect_when_requested(self):
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app. Храни задачи в notes.json.\n'
            'python main.py add "first task"\npython main.py list',
            project_root=str(self.root),
        )
        result = ToolResult(ok=True, output='$ python main.py add "first task"\nAdded\n[exit 0]')

        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'}),
            result,
        )

        self.assertIn("missing_file_side_effect", result.output)
        self.assertNotIn('python main.py add "first task"', worker._command_goals_done)
        self.assertEqual(worker._active_repair.get("failure_type"), "missing_file_side_effect")
        self.assertFalse(worker._controller.final_guard().blocked is False)

    def test_functional_failure_creates_repair_ledger_item(self):
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app.\npython main.py add "first task"\npython main.py list',
            project_root=str(self.root),
        )
        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'}),
            ToolResult(ok=True, output='$ python main.py add "first task"\nAdded\n[exit 0]'),
        )
        result = ToolResult(ok=True, output="$ python main.py list\n\n[exit 0]")
        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": "python main.py list"}),
            result,
        )

        self.assertEqual(worker._active_repair.get("failure_type"), "state_not_persisted")
        self.assertTrue(any(item["type"] == "fix" for item in worker._ledger.to_summary()))
        self.assertTrue(worker._controller.final_guard().blocked)

    def test_done_and_clear_blocked_while_add_list_dependency_failed(self):
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app.\n'
            'python main.py add "first task"\npython main.py list\n'
            'python main.py done 1\npython main.py clear',
            project_root=str(self.root),
        )
        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'}),
            ToolResult(ok=True, output='$ python main.py add "first task"\nAdded\n[exit 0]'),
        )
        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": "python main.py list"}),
            ToolResult(ok=True, output="$ python main.py list\n\n[exit 0]"),
        )

        done = ToolCall(name="run_terminal", args={"command": "python main.py done 1"})
        clear = ToolCall(name="run_terminal", args={"command": "python main.py clear"})

        done_preflight = worker._preflight_tool_call(done, worker._tools["run_terminal"])
        clear_preflight = worker._preflight_tool_call(clear, worker._tools["run_terminal"])

        self.assertIsNotNone(done_preflight)
        self.assertIsNotNone(clear_preflight)
        self.assertIn("command_goal_dependency_blocked", done_preflight.output)
        self.assertIn('python main.py add "first task"', done_preflight.output)

    def test_repair_must_touch_relevant_file_before_rerun(self):
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app. Храни задачи в notes.json.\n'
            'python main.py add "first task"\npython main.py list',
            project_root=str(self.root),
        )
        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'}),
            ToolResult(ok=True, output='$ python main.py add "first task"\nAdded\n[exit 0]'),
        )
        rerun = ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'})

        preflight = worker._preflight_tool_call(rerun, worker._tools["run_terminal"])

        self.assertIsNotNone(preflight)
        self.assertIn("repair_required_before_rerun", preflight.output)

    def test_relevant_patch_allows_failed_command_rerun(self):
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app. Храни задачи в notes.json.\n'
            'python main.py add "first task"\npython main.py list',
            project_root=str(self.root),
        )
        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'}),
            ToolResult(ok=True, output='$ python main.py add "first task"\nAdded\n[exit 0]'),
        )
        worker._record_file_state(
            ToolCall(name="write_file", args={"path": "main.py", "content": "print('ok')\n"}),
            ToolResult(ok=True, output="wrote", meta={"path": "main.py"}),
        )
        rerun = ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'})

        self.assertIsNone(worker._preflight_tool_call(rerun, worker._tools["run_terminal"]))

    def test_repeated_repair_failure_sets_no_progress_blocker(self):
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app.\npython main.py add "first task"\npython main.py list',
            project_root=str(self.root),
        )
        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'}),
            ToolResult(ok=True, output='$ python main.py add "first task"\nAdded\n[exit 0]'),
        )
        list_call = ToolCall(name="run_terminal", args={"command": "python main.py list"})
        worker._record_command_result(list_call, ToolResult(ok=True, output="$ python main.py list\n\n[exit 0]"))
        worker._record_file_state(
            ToolCall(name="write_file", args={"path": "main.py", "content": "print('still wrong')\n"}),
            ToolResult(ok=True, output="wrote", meta={"path": "main.py"}),
        )
        worker._record_command_result(list_call, ToolResult(ok=True, output="$ python main.py list\n\n[exit 0]"))
        worker._record_command_result(list_call, ToolResult(ok=True, output="$ python main.py list\n\n[exit 0]"))

        preflight = worker._preflight_tool_call(
            ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'}),
            worker._tools["run_terminal"],
        )

        self.assertIsNotNone(preflight)
        self.assertIn("repair_no_progress_blocker", preflight.output)

    def test_repair_rerun_requires_clean_storage_before_add(self):
        (self.root / "notes.json").write_text('[{"task": "first task", "done": true}]', encoding="utf-8")
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app. Храни задачи в notes.json.\n'
            'python main.py add "first task"\npython main.py list',
            project_root=str(self.root),
        )
        worker._todo_cli_added_texts.add("first task")
        worker._active_repair = {"id": "repair-cli-command-goals", "touched_relevant": True}
        worker._repair_touched_relevant = True

        preflight = worker._preflight_tool_call(
            ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'}),
            worker._tools["run_terminal"],
        )

        self.assertIsNotNone(preflight)
        self.assertIn("clean_state_required_before_rerun", preflight.output)

    def test_auto_repair_rerun_resets_storage_and_closes_exact_goals(self):
        (self.root / "notes.json").write_text('[{"task": "first task", "done": true}]', encoding="utf-8")
        (self.root / "main.py").write_text(
            "\n".join([
                "import json, sys",
                "def load():",
                "    try:",
                "        return json.load(open('notes.json', 'r', encoding='utf-8'))",
                "    except Exception:",
                "        return {'tasks': []}",
                "def save(data):",
                "    with open('notes.json', 'w', encoding='utf-8') as f:",
                "        json.dump(data, f)",
                "cmd = sys.argv[1]",
                "data = load()",
                "tasks = data.setdefault('tasks', [])",
                "if cmd == 'add':",
                "    tasks.append({'task': ' '.join(sys.argv[2:]), 'done': False}); save(data)",
                "elif cmd == 'list':",
                "    [print(f\"{i}. {t['task']} - {'Done' if t['done'] else 'Pending'}\") for i, t in enumerate(tasks, 1)]",
                "elif cmd == 'done':",
                "    tasks[int(sys.argv[2]) - 1]['done'] = True; save(data)",
                "elif cmd == 'clear':",
                "    data['tasks'] = []; save(data)",
                "",
            ]),
            encoding="utf-8",
        )
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app. Храни задачи в notes.json.\n'
            'python main.py add "first task"\npython main.py list\n'
            'python main.py done 1\npython main.py clear',
            project_root=str(self.root),
            confirmation_policy="auto_confirm",
        )
        worker._active_repair = {
            "id": "repair-cli-command-goals",
            "description": "Fix clear storage behavior",
            "failure_type": "missing_file_side_effect",
        }
        worker._repair_touched_relevant = True
        worker._todo_cli_added_texts.add("first task")

        lines = worker._auto_rerun_repair_sequence()

        self.assertTrue(lines)
        self.assertTrue(worker._all_command_goals_done())
        self.assertFalse(worker._active_repair)
        self.assertNotIn("first task", (self.root / "notes.json").read_text(encoding="utf-8"))

    def test_todo_cli_rollback_resets_ledger_sequence_for_repair_rerun(self):
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app. Храни задачи в notes.json.\n'
            'python main.py add "first task"\npython main.py list\n'
            'python main.py done 1\npython main.py clear',
            project_root=str(self.root),
            confirmation_policy="auto_confirm",
        )
        worker._command_goals_done.update(
            {'python main.py add "first task"', "python main.py list", "python main.py done 1"}
        )
        for item in worker._ledger.items:
            if item.command in worker._command_goals_done:
                item.status = TaskStatus.DONE
            if item.command == "python main.py clear":
                item.status = TaskStatus.DOING

        worker._rollback_todo_cli_goals()
        worker._active_repair = {"id": "repair-cli-command-goals", "touched_relevant": True}
        worker._repair_touched_relevant = True

        self.assertEqual(worker._command_goals_done, set())
        self.assertEqual(worker._ledger.current_item().command, 'python main.py add "first task"')
        preflight = worker._preflight_tool_call(
            ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'}),
            worker._tools["run_terminal"],
        )
        self.assertIsNone(preflight)

    def test_done_invalid_output_is_functional_failure_even_with_exit_zero(self):
        (self.root / "notes.json").write_text('[{"task": "first task", "done": false}]', encoding="utf-8")
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app. Храни задачи в notes.json.\n'
            'python main.py add "first task"\npython main.py list\npython main.py done 1',
            project_root=str(self.root),
        )
        worker._command_goals_done.update({'python main.py add "first task"', "python main.py list"})
        result = ToolResult(ok=True, output="$ python main.py done 1\nInvalid task number.\n\n[exit 0]")

        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": "python main.py done 1"}),
            result,
        )

        self.assertIn("verification failed", result.output)
        self.assertNotIn("python main.py done 1", worker._command_goals_done)
        self.assertEqual(worker._active_repair.get("failure_type"), "missing_expected_stdout")

    def _traceback_output(self) -> str:
        main = self.root / "main.py"
        calc = self.root / "calculator.py"
        return (
            f"$ python main.py\n"
            "Traceback (most recent call last):\n"
            f"  File \"{main}\", line 3, in <module>\n"
            "    print(divide(10, 0))\n"
            f"  File \"{calc}\", line 2, in divide\n"
            "    return a / b\n"
            "           ~~^~~\n"
            "ZeroDivisionError: division by zero\n"
            "[exit 1]"
        )

    def test_traceback_parser_extracts_file_line_and_error_type(self):
        info = parse_traceback(self._traceback_output(), self.root)

        self.assertIsNotNone(info)
        self.assertEqual(info.error_type, "ZeroDivisionError")
        self.assertIn("division by zero", info.message)
        self.assertIn("main.py", info.relevant_files)
        self.assertIn("calculator.py", info.relevant_files)
        self.assertEqual(info.frames[-1].line, 2)

    def test_failed_traceback_run_creates_repair_ledger_item(self):
        worker = AgentWorker(
            agent_profile(),
            "Запусти проект, найди ошибку по traceback и исправь. После исправления снова запусти команду.",
            project_root=str(self.root),
            confirmation_policy="auto_confirm",
        )
        call = ToolCall(name="run_terminal", args={"command": "python main.py"})

        worker._record_command_result(call, ToolResult(ok=False, output=self._traceback_output()))

        self.assertIn("python main.py", worker._command_goals)
        self.assertEqual(worker._active_repair.get("failure_type"), "traceback_error")
        self.assertEqual(worker._active_repair.get("id"), "repair-traceback")
        self.assertIn("calculator.py", worker._active_repair.get("target_files"))
        self.assertTrue(any(item["id"] == "repair-traceback" for item in worker._ledger.to_summary()))
        self.assertFalse(worker._final_evaluation().allowed)

    def test_traceback_task_does_not_inherit_old_todo_command_goals_from_history(self):
        worker = AgentWorker(
            agent_profile(),
            "Запусти проект, найди ошибку по traceback и исправь. После исправления снова запусти команду.",
            history=[
                (
                    'Создай todo CLI.\npython main.py add "first task"\npython main.py list',
                    "Готово: add/list/done/clear проверены.",
                )
            ],
            project_root=str(self.root),
        )

        self.assertEqual(worker._command_goals, [])

        call = ToolCall(name="run_terminal", args={"command": "python main.py"})
        worker._record_command_result(call, ToolResult(ok=False, output=self._traceback_output()))

        self.assertEqual(worker._command_goals, ["python main.py"])
        self.assertNotIn('python main.py add "first task"', worker._command_goals)

    def test_new_traceback_action_omits_old_todo_history_from_prompt_context(self):
        worker = AgentWorker(
            agent_profile(),
            "Запусти проект, найди ошибку по traceback и исправь. После исправления снова запусти команду.",
            history=[
                (
                    'Создай todo CLI.\npython main.py add "first task"\npython main.py list',
                    "Готово: add/list/done/clear проверены.",
                )
            ],
            project_root=str(self.root),
        )

        context = worker._history_context_for_agent()

        self.assertIn("omitted", context)
        self.assertNotIn("first task", context)
        self.assertNotIn("python main.py add", context)

    def test_traceback_repair_requires_relevant_file_before_rerun(self):
        worker = AgentWorker(
            agent_profile(),
            "Запусти проект, найди ошибку по traceback и исправь. После исправления снова запусти команду.",
            project_root=str(self.root),
            confirmation_policy="auto_confirm",
        )
        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": "python main.py"}),
            ToolResult(ok=False, output=self._traceback_output()),
        )

        preflight = worker._preflight_tool_call(
            ToolCall(name="run_terminal", args={"command": "python main.py"}),
            worker._tools["run_terminal"],
        )

        self.assertIsNotNone(preflight)
        self.assertIn("repair_required_before_rerun", preflight.output)
        self.assertIn("calculator.py", preflight.output)

    def test_unrelated_patch_does_not_close_traceback_repair_item(self):
        worker = AgentWorker(
            agent_profile(),
            "Запусти проект, найди ошибку по traceback и исправь. После исправления снова запусти команду.",
            project_root=str(self.root),
            confirmation_policy="auto_confirm",
        )
        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": "python main.py"}),
            ToolResult(ok=False, output=self._traceback_output()),
        )

        worker._record_file_state(
            ToolCall(name="write_file", args={"path": "unrelated.py", "content": "print('x')\n"}),
            ToolResult(ok=True, output="wrote", meta={"path": "unrelated.py"}),
        )

        self.assertFalse(worker._repair_touched_relevant)
        self.assertTrue(worker._active_repair)
        preflight = worker._preflight_tool_call(
            ToolCall(name="run_terminal", args={"command": "python main.py"}),
            worker._tools["run_terminal"],
        )
        self.assertIsNotNone(preflight)

    def test_traceback_repair_rerun_after_relevant_patch_closes_goal(self):
        worker = AgentWorker(
            agent_profile(),
            "Запусти проект, найди ошибку по traceback и исправь. После исправления снова запусти команду.",
            project_root=str(self.root),
        )
        call = ToolCall(name="run_terminal", args={"command": "python main.py"})
        worker._record_command_result(call, ToolResult(ok=False, output=self._traceback_output()))
        worker._record_file_state(
            ToolCall(name="edit_file", args={"path": "calculator.py", "old_str": "return a / b", "new_str": "return None if b == 0 else a / b"}),
            ToolResult(ok=True, output="edited", meta={"path": "calculator.py"}),
        )

        self.assertIsNone(worker._preflight_tool_call(call, worker._tools["run_terminal"]))
        worker._record_command_result(call, ToolResult(ok=True, output="$ python main.py\nNone\n[exit 0]"))

        self.assertIn("python main.py", worker._command_goals_done)
        self.assertFalse(worker._active_repair)
        self.assertTrue(worker._final_evaluation().allowed)

    def test_failed_traceback_rerun_requires_new_relevant_patch(self):
        worker = AgentWorker(
            agent_profile(),
            "Запусти проект, найди ошибку по traceback и исправь. После исправления снова запусти команду.",
            project_root=str(self.root),
        )
        call = ToolCall(name="run_terminal", args={"command": "python main.py"})
        worker._record_command_result(call, ToolResult(ok=False, output=self._traceback_output()))
        worker._record_file_state(
            ToolCall(name="edit_file", args={"path": "calculator.py", "old_str": "return a / b", "new_str": "raise ValueError('division by zero')"}),
            ToolResult(ok=True, output="edited", meta={"path": "calculator.py"}),
        )

        worker._record_command_result(
            call,
            ToolResult(
                ok=False,
                output=(
                    "$ python main.py\nTraceback (most recent call last):\n"
                    f"  File \"{self.root / 'main.py'}\", line 3, in <module>\n"
                    "    print(divide(10, 0))\n"
                    f"  File \"{self.root / 'calculator.py'}\", line 3, in divide\n"
                    "    raise ValueError('division by zero')\n"
                    "ValueError: division by zero\n[exit 1]"
                ),
            ),
        )

        self.assertFalse(worker._repair_touched_relevant)
        preflight = worker._preflight_tool_call(call, worker._tools["run_terminal"])
        self.assertIsNotNone(preflight)
        self.assertIn("do not replace one exception with another", preflight.output)

    def test_idempotent_traceback_edit_is_not_repair_progress(self):
        worker = AgentWorker(
            agent_profile(),
            "Запусти проект, найди ошибку по traceback и исправь. После исправления снова запусти команду.",
            project_root=str(self.root),
        )
        worker._record_command_result(
            ToolCall(name="run_terminal", args={"command": "python main.py"}),
            ToolResult(ok=False, output=self._traceback_output()),
        )
        worker._record_file_state(
            ToolCall(name="edit_file", args={"path": "calculator.py"}),
            ToolResult(ok=True, output="already", meta={"path": "calculator.py", "idempotent": True}),
        )

        self.assertFalse(worker._repair_touched_relevant)

    def test_traceback_safe_patch_replaces_valueerror_with_exit_zero_return(self):
        (self.root / "calculator.py").write_text(
            "def divide(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n",
            encoding="utf-8",
        )
        worker = AgentWorker(
            agent_profile(),
            "Запусти проект, найди ошибку по traceback и исправь. После исправления снова запусти команду.",
            project_root=str(self.root),
            confirmation_policy="auto_confirm",
        )
        output = (
            "$ python main.py\nTraceback (most recent call last):\n"
            f"  File \"{self.root / 'main.py'}\", line 3, in <module>\n"
            "    print(divide(10, 0))\n"
            f"  File \"{self.root / 'calculator.py'}\", line 3, in divide\n"
            "    raise ValueError('division by zero')\n"
            "ValueError: division by zero\n[exit 1]"
        )

        lines = worker._attempt_traceback_safe_patch(output)

        self.assertTrue(lines)
        text = (self.root / "calculator.py").read_text(encoding="utf-8")
        self.assertIn('return "Error: Division by zero"', text)
        self.assertNotIn("raise ValueError", text)

    def test_traceback_safe_patch_adds_module_level_wrapper_for_class_method(self):
        (self.root / "app").mkdir()
        (self.root / "app" / "controller.py").write_text(
            "class Controller:\n"
            "    def run(self):\n"
            "        print('ok')\n",
            encoding="utf-8",
        )
        worker = AgentWorker(
            agent_profile(),
            "Создай проект и проверь запуск через терминал.",
            project_root=str(self.root),
            confirmation_policy="auto_confirm",
        )
        worker._active_repair = {"id": "repair-traceback", "target_files": ["main.py"]}
        output = (
            "$ python main.py\nTraceback (most recent call last):\n"
            f"  File \"{self.root / 'main.py'}\", line 4, in <module>\n"
            "    app.controller.run()\n"
            "AttributeError: module 'app.controller' has no attribute 'run'\n"
            "[exit 1]"
        )

        lines = worker._attempt_traceback_safe_patch(output)

        self.assertTrue(lines)
        text = (self.root / "app" / "controller.py").read_text(encoding="utf-8")
        self.assertIn("def run():", text)
        self.assertIn("Controller().run()", text)
        self.assertTrue(worker._repair_touched_relevant)
        self.assertIn("app/controller.py", worker._active_repair["target_files"])

    def test_repeated_traceback_run_without_patch_triggers_guard(self):
        worker = AgentWorker(
            agent_profile(),
            "Запусти проект, найди ошибку по traceback и исправь. После исправления снова запусти команду.",
            project_root=str(self.root),
        )
        call = ToolCall(name="run_terminal", args={"command": "python main.py"})
        worker._record_command_result(call, ToolResult(ok=False, output=self._traceback_output()))

        preflight = worker._preflight_tool_call(call, worker._tools["run_terminal"])

        self.assertIsNotNone(preflight)
        self.assertIn("repair_required_before_rerun", preflight.output)

    def test_failed_exact_command_interrupts_remaining_batch_for_repair(self):
        (self.root / "main.py").write_text(
            "from missing_module import TodoApp\n",
            encoding="utf-8",
        )
        calls = """
<tool name="run_terminal">
<command>python main.py add "first task"</command>
</tool>
<tool name="run_terminal">
<command>python main.py list</command>
</tool>
"""

        _, _, results, worker = self.run_worker(
            [calls, "Готово."],
            user_message='Проверь команды:\npython main.py add "first task"\npython main.py list',
            max_agent_steps=2,
        )

        commands = [
            item["meta"].get("command") or item["output"].splitlines()[0].removeprefix("$ ")
            for item in results
            if item["name"] == "run_terminal"
        ]
        self.assertEqual(commands, ['python main.py add "first task"'])
        self.assertNotIn('python main.py add "first task"', worker._command_goals_done)
        self.assertFalse(worker._command_tool_succeeded)

    def test_pending_exact_command_goal_can_rerun_after_repair(self):
        worker = AgentWorker(
            agent_profile(),
            'Проверь команды:\npython main.py add "first task"\npython main.py list',
            project_root=str(self.root),
        )
        call = ToolCall(name="run_terminal", args={"command": 'python main.py add "first task"'})
        tool = worker._tools["run_terminal"]
        worker._successful_tool_keys.add(worker._tool_key(call))

        self.assertTrue(worker._allow_duplicate_command_goal_rerun(tool, call))

        worker._command_goals_done.add('python main.py add "first task"')
        self.assertFalse(worker._allow_duplicate_command_goal_rerun(tool, call))

    def test_v3_command_goals_keep_exact_and_semantic_modes(self):
        worker = AgentWorker(
            agent_profile(),
            'Создай todo CLI.\npython main.py add "first task"\npython main.py list',
            project_root=str(self.root),
        )

        modes = {goal.raw: goal.mode for goal in worker._command_goal_specs}

        self.assertEqual(modes['python main.py add "first task"'], "exact")
        self.assertEqual(modes["python main.py list"], "exact")
        exact = next(goal for goal in worker._command_goal_specs if goal.raw == 'python main.py add "first task"')
        self.assertFalse(exact.matches('python main.py add "first note"'))
        self.assertNotIn("python main.py add", modes)

    def test_v3_final_guard_blocks_pending_ledger(self):
        ledger = TaskLedger.from_command_goals([
            CommandGoal(
                raw='python main.py add "first task"',
                normalized=normalize_command('python main.py add "first task"'),
                mode="exact",
            )
        ])

        decision = final_with_pending_goals_guard(ledger)

        self.assertTrue(decision.blocked)
        self.assertIn("pending", decision.reason)
        self.assertIn("XML tool", decision.corrective)

    def test_v3_evaluator_blocks_active_repair_even_without_pending_ledger(self):
        state = AgentRunStateV3(
            run_id="eval",
            user_message="создай todo",
            resolved_task="создай todo",
            command_goals=["python main.py list"],
            command_goals_done=["python main.py list"],
        )
        ledger = TaskLedger()

        result = evaluate_final_readiness(
            state,
            ledger,
            command_goals=["python main.py list"],
            command_goals_done={"python main.py list"},
            active_repair={"id": "repair-cli-command-goals", "failure_type": "state_not_persisted"},
        )

        self.assertFalse(result.allowed)
        self.assertIn("pending repair", result.reason)

    def test_agentworker_final_evaluation_uses_repair_state(self):
        worker = AgentWorker(
            agent_profile(),
            'Создай CLI todo app.\npython main.py list',
            project_root=str(self.root),
        )
        worker._ledger = TaskLedger()
        worker._command_goals = ["python main.py list"]
        worker._command_goals_done.add("python main.py list")
        worker._active_repair = {
            "id": "repair-cli-command-goals",
            "description": "Fix state_not_persisted for python main.py list",
        }

        result = worker._final_evaluation()

        self.assertFalse(result.allowed)
        self.assertIn("pending repair", result.reason)
        self.assertFalse(worker.run_state_v3.final_allowed)

    def test_v3_project_map_ignores_runtime_and_model_artifacts(self):
        (self.root / "main.py").write_text("print('ok')\n", encoding="utf-8")
        (self.root / "models").mkdir()
        (self.root / "models" / "model.gguf").write_text("x", encoding="utf-8")
        (self.root / "dist").mkdir()
        (self.root / "dist" / "ZenAI.exe").write_text("x", encoding="utf-8")
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__" / "main.pyc").write_text("x", encoding="utf-8")

        tree = build_project_map(self.root)

        self.assertIn("main.py", tree)
        self.assertNotIn("models", tree)
        self.assertNotIn(".gguf", tree)
        self.assertNotIn("dist", tree)
        self.assertNotIn("__pycache__", tree)

    def test_existing_large_file_write_without_read_is_blocked(self):
        path = self.root / "existing.py"
        path.write_text("\n".join(f"line_{i} = {i}" for i in range(100)), encoding="utf-8")
        worker = AgentWorker(agent_profile(), "измени existing.py", project_root=str(self.root))
        call = ToolCall(name="write_file", args={"path": "existing.py", "content": "print('rewrite')\n"})

        result = worker._preflight_tool_call(call, worker._tools["write_file"])

        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertIn("existing_file_overwrite_guard", result.output)

        worker._read_files.add("existing.py")
        self.assertIsNone(worker._preflight_tool_call(call, worker._tools["write_file"]))

    def prepare_existing_calculator_project(self):
        (self.root / "app").mkdir()
        (self.root / "app" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "app" / "calculator.py").write_text(
            "def add(a, b):\n"
            "    return a + b\n\n\n"
            "def subtract(a, b):\n"
            "    return a - b\n\n\n"
            "def multiply(a, b):\n"
            "    return a * b\n",
            encoding="utf-8",
        )
        (self.root / "app" / "cli.py").write_text(
            "import argparse\n\n"
            "from app.calculator import add, subtract, multiply\n\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser(description=\"Small calculator CLI\")\n"
            "    parser.add_argument(\"operation\", choices=[\"add\", \"subtract\", \"multiply\"])\n"
            "    parser.add_argument(\"a\", type=float)\n"
            "    parser.add_argument(\"b\", type=float)\n"
            "    args = parser.parse_args()\n\n"
            "    if args.operation == \"add\":\n"
            "        result = add(args.a, args.b)\n"
            "    elif args.operation == \"subtract\":\n"
            "        result = subtract(args.a, args.b)\n"
            "    else:\n"
            "        result = multiply(args.a, args.b)\n\n"
            "    print(result)\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    main()\n",
            encoding="utf-8",
        )
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_calculator.py").write_text(
            "from app.calculator import add, subtract, multiply\n\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n\n\n"
            "def test_subtract():\n"
            "    assert subtract(5, 2) == 3\n\n\n"
            "def test_multiply():\n"
            "    assert multiply(4, 3) == 12\n",
            encoding="utf-8",
        )
        (self.root / "README.md").write_text(
            "# Calculator CLI\n\n"
            "Commands:\n\n"
            "```bash\n"
            "python -m app.cli add 2 3\n"
            "python -m app.cli subtract 5 2\n"
            "python -m app.cli multiply 4 3\n"
            "```\n",
            encoding="utf-8",
        )

    def existing_calculator_task(self) -> str:
        return (
            "В существующий calculator CLI добавь операцию divide. Нужно:\n"
            "- добавить функцию divide(a, b);\n"
            "- обработать деление на ноль понятной ошибкой;\n"
            "- добавить CLI operation divide;\n"
            "- добавить тесты;\n"
            "- обновить README;\n"
            "- запустить проверки.\n"
            "Не переписывай существующие файлы целиком, внеси точечные изменения."
        )

    def test_existing_project_divide_infers_cli_regression_goals(self):
        worker = AgentWorker(agent_profile(), self.existing_calculator_task(), project_root=str(self.root))

        self.assertEqual(
            worker._command_goals,
            [
                "python -m app.cli add 2 3",
                "python -m app.cli subtract 5 2",
                "python -m app.cli multiply 4 3",
                "python -m app.cli divide 10 2",
                "python -m app.cli divide 10 0",
            ],
        )

    def test_existing_project_patch_requires_edit_not_write_file(self):
        self.prepare_existing_calculator_project()
        worker = AgentWorker(agent_profile(), self.existing_calculator_task(), project_root=str(self.root))
        worker._read_files.add("app/calculator.py")
        call = ToolCall(
            name="write_file",
            args={"path": "app/calculator.py", "content": "def divide(a, b):\n    return a / b\n"},
        )

        result = worker._preflight_tool_call(call, worker._tools["write_file"])

        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertIn("patch_required_for_existing_file", result.output)

    def test_existing_project_unrelated_file_edit_triggers_guard(self):
        self.prepare_existing_calculator_project()
        worker = AgentWorker(agent_profile(), self.existing_calculator_task(), project_root=str(self.root))
        worker._read_files.add("unrelated.py")
        call = ToolCall(
            name="edit_file",
            args={"path": "unrelated.py", "old_str": "x", "new_str": "y"},
        )

        result = worker._preflight_tool_call(call, worker._tools["edit_file"])

        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertIn("unrelated_file_guard", result.output)

    def test_existing_project_divide_fallback_patches_and_verifies(self):
        self.prepare_existing_calculator_project()
        worker = AgentWorker(agent_profile(), self.existing_calculator_task(), project_root=str(self.root))
        worker._needs_confirmation = lambda tool, call=None: False
        failed_call = ToolCall(
            name="edit_file",
            args={"path": "app/calculator.py", "old_str": "", "new_str": "def divide(a, b):\n    return a / b\n"},
        )
        failed_result = ToolResult.error("[error: missing old_str]")

        self.assertTrue(worker._should_attempt_existing_calculator_divide_fallback(failed_call, failed_result))
        lines = worker._attempt_existing_calculator_divide_patch()
        verification = worker._auto_run_pending_command_goals()

        self.assertTrue(lines)
        self.assertTrue(verification)
        self.assertTrue(worker._all_command_goals_done())
        self.assertIn("calculator tests passed", "\n".join(verification))
        self.assertIn("app/calculator.py", worker._changed_files)
        self.assertIn("app/cli.py", worker._changed_files)
        self.assertIn("tests/test_calculator.py", worker._changed_files)
        self.assertIn("README.md", worker._changed_files)
        self.assertIn("def divide(a, b):", (self.root / "app" / "calculator.py").read_text(encoding="utf-8"))
        self.assertIn("\"divide\"", (self.root / "app" / "cli.py").read_text(encoding="utf-8"))
        self.assertIn("test_divide", (self.root / "tests" / "test_calculator.py").read_text(encoding="utf-8"))
        checks = [
            (["python", "-m", "app.cli", "add", "2", "3"], "5"),
            (["python", "-m", "app.cli", "subtract", "5", "2"], "3"),
            (["python", "-m", "app.cli", "multiply", "4", "3"], "12"),
            (["python", "-m", "app.cli", "divide", "10", "2"], "5"),
            (["python", "-m", "app.cli", "divide", "10", "0"], "zero"),
        ]
        for command, expected in checks:
            completed = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(expected, (completed.stdout + completed.stderr).lower())

    def test_divide_command_functional_output_is_verified(self):
        worker = AgentWorker(agent_profile(), self.existing_calculator_task(), project_root=str(self.root))
        bad_call = ToolCall(name="run_terminal", args={"command": "python -m app.cli divide 10 2"})

        worker._record_command_result(
            bad_call,
            ToolResult(ok=True, output="$ python -m app.cli divide 10 2\n4.0\n[exit 0]"),
        )

        self.assertNotIn("python -m app.cli divide 10 2", worker._command_goals_done)
        self.assertTrue(worker._active_repair)

        worker._active_repair = {}
        worker._record_command_result(
            bad_call,
            ToolResult(ok=True, output="$ python -m app.cli divide 10 2\n5.0\n[exit 0]"),
        )

        self.assertIn("python -m app.cli divide 10 2", worker._command_goals_done)

    def test_divide_zero_traceback_creates_repair_ledger_item(self):
        worker = AgentWorker(agent_profile(), self.existing_calculator_task(), project_root=str(self.root))
        call = ToolCall(name="run_terminal", args={"command": "python -m app.cli divide 10 0"})

        worker._record_command_result(
            call,
            ToolResult(
                ok=False,
                output="$ python -m app.cli divide 10 0\nTraceback (most recent call last):\nZeroDivisionError: division by zero\n[exit 1]",
            ),
        )

        self.assertEqual(worker._active_repair.get("failure_type"), "traceback_error")
        self.assertNotIn("python -m app.cli divide 10 0", worker._command_goals_done)

    def test_final_blocked_after_existing_project_patch_until_commands_run(self):
        self.prepare_existing_calculator_project()
        worker = AgentWorker(agent_profile(), self.existing_calculator_task(), project_root=str(self.root))
        call = ToolCall(
            name="edit_file",
            args={"path": "app/calculator.py", "old_str": "def multiply(a, b):", "new_str": "def multiply(a, b):"},
        )
        worker._record_file_state(call, ToolResult(ok=True, output="changed", meta={"path": "app/calculator.py"}))

        result = worker._final_evaluation()

        self.assertFalse(result.allowed)
        self.assertIn("pending command goals", result.reason)
        self.assertIn("app/calculator.py", worker.run_state_v3.changed_files)

    def test_markdown_only_divide_implementation_is_rejected(self):
        markdown = """Вот код:
```python
def divide(a, b):
    return a / b
```
"""
        _, chunks, results, _ = self.run_worker(
            [markdown, "Готово."],
            user_message=self.existing_calculator_task(),
            max_agent_steps=2,
        )

        self.assertFalse(results)
        self.assertIn("текст/markdown не меняет файлы", "".join(chunks))

    def test_failed_command_goal_remains_pending_until_rerun_success(self):
        worker = AgentWorker(
            agent_profile(),
            'Проверь команду:\npython main.py list',
            project_root=str(self.root),
        )
        call = ToolCall(name="run_terminal", args={"command": "python main.py list"})

        worker._record_command_result(call, ToolResult(ok=False, output="$ python main.py list\nboom\n[exit 1]"))
        self.assertNotIn("python main.py list", worker._command_goals_done)
        self.assertFalse(worker._command_tool_succeeded)

        worker._record_command_result(call, ToolResult(ok=True, output="$ python main.py list\nok\n[exit 0]"))
        self.assertIn("python main.py list", worker._command_goals_done)
        self.assertTrue(worker._command_tool_succeeded)

    def test_final_response_blocked_until_exact_command_goal_done(self):
        (self.root / "main.py").write_text("print('ok')\n", encoding="utf-8")
        run = """
<tool name="run_terminal">
<command>python main.py list</command>
</tool>
"""
        model, chunks, results, worker = self.run_worker(
            ["Готово без проверки.", run, "Готово после проверки."],
            user_message="Проверь команду:\npython main.py list",
            max_agent_steps=4,
        )

        self.assertEqual([item["name"] for item in results], ["run_terminal"])
        self.assertIn("python main.py list", worker._command_goals_done)
        self.assertIn("System corrective instruction", model.prompts[1])
        self.assertIn("Готово после проверки", "".join(chunks))

    def test_dangerous_command_is_blocked_even_with_auto_confirm_policy(self):
        result = RunTerminalTool(str(self.root)).execute(
            ToolCall(name="run_terminal", args={"command": "python hello.py; del hello.py"})
        )

        self.assertFalse(result.ok)
        self.assertIn("blocked", result.output)

    def test_simple_create_edit_run_final_does_not_hit_max_steps(self):
        write = """
<tool name="write_file">
<path>hello.py</path>
<content>print("hi")
</content>
</tool>
"""
        edit = """
<tool name="edit_file">
<path>hello.py</path>
<old_str>print("hi")</old_str>
<new_str>print("hello world")</new_str>
</tool>
"""
        run = """
<tool name="run_terminal">
<command>python hello.py</command>
</tool>
"""

        _, chunks, results, worker = self.run_worker(
            [write, edit, run, "Готово: файл создан, изменён и запущен."],
            user_message="создай, измени и выполни python hello.py",
            max_agent_steps=4,
        )

        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), 'print("hello world")\n')
        self.assertIn("hello world", "\n".join(item["output"] for item in results))
        self.assertNotIn("лимит итераций", "".join(chunks))
        self.assertIsNone(worker.continuation_state)

    def test_file_goals_extract_multiple_required_paths(self):
        goals = extract_file_goals(
            "Создай main.py, app/model.py, app/controller.py, app/view.py и README.md"
        )

        self.assertEqual(
            [goal.path for goal in goals],
            ["main.py", "app/model.py", "app/controller.py", "app/view.py", "README.md"],
        )
        controller = next(goal for goal in goals if goal.path == "app/controller.py")
        view = next(goal for goal in goals if goal.path == "app/view.py")
        self.assertTrue(controller.dependency_ids)
        self.assertTrue(view.dependency_ids)

    def test_final_blocked_when_required_file_goal_missing(self):
        state = AgentRunStateV3(run_id="r", user_message="u", resolved_task="create app/view.py")
        state.file_goals = [FileGoal(path="app/view.py")]
        result = evaluate_final_readiness(state, TaskLedger())

        self.assertFalse(result.allowed)
        self.assertIn("missing required files", result.reason)

    def test_empty_file_does_not_close_file_goal(self):
        (self.root / "main.py").write_text("", encoding="utf-8")
        goal = verify_file_goal(FileGoal(path="main.py"), str(self.root))

        self.assertEqual(goal.status, FileGoalStatus.FAILED)
        self.assertIn("Empty required file", goal.failure_reason)

    def test_placeholder_file_does_not_close_file_goal(self):
        (self.root / "app.py").write_text("def run():\n    ...\n", encoding="utf-8")
        goal = verify_file_goal(FileGoal(path="app.py"), str(self.root))

        self.assertEqual(goal.status, FileGoalStatus.FAILED)
        self.assertIn("Placeholder code detected", goal.failure_reason)

    def test_python_file_goal_requires_ast_parse(self):
        (self.root / "bad.py").write_text("def broken(:\n", encoding="utf-8")
        goal = verify_file_goal(FileGoal(path="bad.py"), str(self.root))

        self.assertEqual(goal.status, FileGoalStatus.FAILED)
        self.assertIn("Python syntax error", goal.failure_reason)

    def test_expected_symbols_are_detected_for_file_goal(self):
        (self.root / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        goal = verify_file_goal(FileGoal(path="calculator.py", expected_symbols=["divide"]), str(self.root))

        self.assertEqual(goal.status, FileGoalStatus.FAILED)
        self.assertIn("Missing expected symbols", goal.failure_reason)

        (self.root / "calculator.py").write_text(
            "def add(a, b):\n    return a + b\n\n"
            "def divide(a, b):\n    return a / b\n",
            encoding="utf-8",
        )
        goal = verify_file_goal(FileGoal(path="calculator.py", expected_symbols=["divide"]), str(self.root))
        self.assertEqual(goal.status, FileGoalStatus.DONE)

    def test_evaluator_allows_final_when_file_and_command_goals_done(self):
        state = AgentRunStateV3(run_id="r", user_message="u", resolved_task="create main.py")
        state.file_goals = [FileGoal(path="main.py", status=FileGoalStatus.DONE)]
        state.command_goals = ["python main.py"]
        state.command_goals_done = ["python main.py"]
        result = evaluate_final_readiness(
            state,
            TaskLedger(),
            command_goals=["python main.py"],
            command_goals_done={"python main.py"},
        )

        self.assertTrue(result.allowed)

    def test_lazy_generation_guard_blocks_write_placeholder(self):
        worker = AgentWorker(agent_profile(), "создай main.py", project_root=str(self.root))
        call = ToolCall("write_file", {"path": "main.py", "content": "def main():\n    ...\n"}, "")

        result = worker._preflight_tool_call(call, worker._tools["write_file"])

        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertTrue(result.meta.get("lazy_generation"))

    def test_lazy_generation_guard_blocks_russian_placeholder(self):
        worker = AgentWorker(agent_profile(), "создай main.py", project_root=str(self.root))
        call = ToolCall("write_file", {"path": "main.py", "content": "# остальной код\n"}, "")

        result = worker._preflight_tool_call(call, worker._tools["write_file"])

        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertIn("остальной код", result.output)

    def test_lazy_generation_guard_blocks_edit_todo_implementation(self):
        (self.root / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")
        worker = AgentWorker(agent_profile(), "измени main.py", project_root=str(self.root))
        worker._read_files.add("main.py")
        call = ToolCall(
            "edit_file",
            {"path": "main.py", "old_str": "return 1", "new_str": "TODO implement"},
            "",
        )

        result = worker._preflight_tool_call(call, worker._tools["edit_file"])

        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertTrue(result.meta.get("lazy_generation"))

    def test_lazy_generation_guard_blocks_patch_notimplementederror(self):
        (self.root / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")
        worker = AgentWorker(agent_profile(), "измени main.py", project_root=str(self.root))
        worker._read_files.add("main.py")
        patch = (
            "--- a/main.py\n"
            "+++ b/main.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def main():\n"
            "-    return 1\n"
            "+    raise NotImplementedError\n"
        )
        call = ToolCall("apply_patch", {"patch": patch}, "")

        result = worker._preflight_tool_call(call, worker._tools["apply_patch"])

        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertTrue(result.meta.get("lazy_generation"))

    def test_empty_init_and_abstract_pass_are_allowed_by_lazy_guard(self):
        self.assertEqual(detect_lazy_placeholders("", path="app/__init__.py"), [])
        abstract = "from abc import abstractmethod\n\nclass Base:\n    @abstractmethod\n    def run(self):\n        pass\n"
        self.assertEqual(detect_lazy_placeholders(abstract, path="base.py", purpose="abstract interface"), [])

    def test_one_mutating_action_per_turn_blocks_batch(self):
        worker = AgentWorker(
            agent_profile(),
            "Создай main.py и app/model.py",
            project_root=str(self.root),
        )
        calls = [
            ToolCall("write_file", {"path": "main.py", "content": "print('ok')\n"}, ""),
            ToolCall("write_file", {"path": "app/model.py", "content": "VALUE = 1\n"}, ""),
        ]

        message = worker._one_mutating_file_action_batch_guard(calls)

        self.assertIn("Only one mutating file action", message)

    def test_batch_can_be_narrowed_to_current_file_goal(self):
        worker = AgentWorker(
            agent_profile(),
            "Создай main.py и app/model.py",
            project_root=str(self.root),
        )
        worker._file_goals = extract_file_goals("Создай main.py и app/model.py")
        worker._ledger = TaskLedger([
            TaskLedgerItem(
                id="file-1",
                description="create_file: main.py",
                type=TaskType.CREATE_FILE,
                status=TaskStatus.DONE,
                target_file="main.py",
                required_tool="write_file",
            ),
            TaskLedgerItem(
                id="file-2",
                description="create_file: app/model.py",
                type=TaskType.CREATE_FILE,
                status=TaskStatus.TODO,
                target_file="app/model.py",
                required_tool="write_file",
            ),
        ])
        worker._controller.ledger = worker._ledger
        calls = [
            ToolCall("write_file", {"path": "main.py", "content": "print('ok')\n"}, ""),
            ToolCall("write_file", {"path": "app/model.py", "content": "VALUE = 1\n"}, ""),
        ]

        narrowed = worker._current_file_goal_call_from_batch(calls)

        self.assertEqual(len(narrowed), 1)
        self.assertEqual(narrowed[0].args["path"], "app/model.py")

    def test_read_plus_one_write_is_allowed_by_batch_guard(self):
        worker = AgentWorker(agent_profile(), "создай main.py", project_root=str(self.root))
        calls = [
            ToolCall("read_file", {"path": "main.py"}, ""),
            ToolCall("write_file", {"path": "main.py", "content": "print('ok')\n"}, ""),
        ]

        self.assertEqual(worker._one_mutating_file_action_batch_guard(calls), "")

    def test_next_turn_can_write_second_file_and_file_goals_verify(self):
        first = '<tool name="write_file"><path>main.py</path><content>print("ok")\n</content></tool>'
        second = '<tool name="write_file"><path>app/model.py</path><content>VALUE = 1\n</content></tool>'
        _, _, results, worker = self.run_worker(
            [first, second, "Готово."],
            user_message="Создай main.py и app/model.py",
            max_agent_steps=5,
        )

        self.assertEqual([item["name"] for item in results], ["write_file", "read_file", "write_file", "read_file"])
        statuses = {goal.path: goal.status for goal in worker._file_goals}
        self.assertEqual(statuses["main.py"], FileGoalStatus.DONE)
        self.assertEqual(statuses["app/model.py"], FileGoalStatus.DONE)

    def test_batch_write_response_does_not_corrupt_file_goal_ledger(self):
        batch = """
<tool name="write_file"><path>main.py</path><content>print("ok")\n</content></tool>
<tool name="write_file"><path>app/model.py</path><content>VALUE = 1\n</content></tool>
"""
        single = '<tool name="write_file"><path>main.py</path><content>print("ok")\n</content></tool>'
        _, chunks, results, worker = self.run_worker(
            [batch, single, "Готово."],
            user_message="Создай main.py и app/model.py",
            max_agent_steps=4,
        )

        self.assertEqual([item["name"] for item in results[:2]], ["write_file", "read_file"])
        self.assertIn("текущий FileGoal", "".join(chunks))
        self.assertEqual((self.root / "main.py").read_text(encoding="utf-8"), 'print("ok")\n')
        self.assertFalse((self.root / "app" / "model.py").exists())
        self.assertIn("app/model.py", [goal.path for goal in worker._file_goals])

    def test_evaluator_reports_missing_files_after_partial_multifile_work(self):
        state = AgentRunStateV3(run_id="r", user_message="u", resolved_task="create files")
        state.file_goals = [
            FileGoal(path="main.py", status=FileGoalStatus.DONE),
            FileGoal(path="app/view.py", status=FileGoalStatus.PLANNED),
        ]
        result = evaluate_final_readiness(state, TaskLedger())

        self.assertFalse(result.allowed)
        self.assertIn("app/view.py", result.reason)

    def test_file_goals_multifile_integration_smoke_creates_all_files_and_runs(self):
        outputs = [
            '<tool name="write_file"><path>main.py</path><content>from app.controller import Controller\n\nif __name__ == "__main__":\n    print(Controller().run())\n</content></tool>',
            '<tool name="write_file"><path>app/model.py</path><content>class Model:\n    def message(self):\n        return "ok"\n</content></tool>',
            '<tool name="write_file"><path>app/controller.py</path><content>from app.model import Model\nfrom app.view import render\n\nclass Controller:\n    def run(self):\n        return render(Model().message())\n</content></tool>',
            '<tool name="write_file"><path>app/view.py</path><content>def render(text):\n    return f"View: {text}"\n</content></tool>',
            '<tool name="write_file"><path>README.md</path><content># Demo app\n\nRun with `python main.py`.\n</content></tool>',
            '<tool name="run_terminal"><command>python main.py</command></tool>',
            "Готово: все файлы созданы и запуск проверен.",
        ]
        _, chunks, results, worker = self.run_worker(
            outputs,
            user_message=(
                "Создай небольшой multi-file Python проект:\n"
                "- main.py\n- app/model.py\n- app/controller.py\n- app/view.py\n- README.md\n"
                "После создания проверь командой:\npython main.py"
            ),
            max_agent_steps=8,
        )

        for path in ["main.py", "app/model.py", "app/controller.py", "app/view.py", "README.md"]:
            content = (self.root / path).read_text(encoding="utf-8")
            self.assertTrue(content.strip(), path)
            self.assertFalse(detect_lazy_placeholders(content, path=path), path)
        names = [item["name"] for item in results]
        self.assertEqual(names.count("write_file"), 5)
        self.assertGreaterEqual(names.count("read_file"), 5)
        self.assertEqual(names[-1], "run_terminal")
        self.assertIn("View: ok", "\n".join(item["output"] for item in results))
        self.assertTrue(all(goal.status == FileGoalStatus.DONE for goal in worker._file_goals))
        self.assertTrue(worker.run_state_v3.final_allowed)
        self.assertIn("Готово", "".join(chunks))


if __name__ == "__main__":
    unittest.main()
