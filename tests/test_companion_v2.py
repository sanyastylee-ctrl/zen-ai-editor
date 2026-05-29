from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QCheckBox, QPushButton, QSlider

from ai import prompt_builder
from ai.worker import InferenceWorker
from core import app_data
from core.companion import (
    CompanionMemoryStore,
    build_companion_context,
    compact_companion_history,
    extract_explicit_memory,
    is_companion_echo_response,
    normalize_companion_echo_text,
    normalize_companion_reply_for_repeat,
)
from core.profiles import AIProfile, ChatTemplate, ProfileKind
from ui.persona_editor import PersonaEditor


class ScriptedTextModel:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def __call__(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        text = self.outputs.pop(0) if self.outputs else "ok"
        return iter([{"choices": [{"text": text}]}])


class ChunkedTextModel:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.prompts: list[str] = []

    def __call__(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        return iter({"choices": [{"text": chunk}]} for chunk in self.chunks)


class FakeManager:
    def __init__(self, model):
        self.model = model

    def on_load_start(self, cb): pass
    def on_load_finish(self, cb): pass
    def off_load_start(self, cb): pass
    def off_load_finish(self, cb): pass
    def get_model(self, **kwargs): return self.model


def companion_profile(**kwargs) -> AIProfile:
    data = {
        "id": "lera",
        "name": "Лера",
        "kind": ProfileKind.COMPANION,
        "chat_template": ChatTemplate.CHATML,
        "model_file": "lera.gguf",
        "system_prompt": "Ты {character_name}.",
        "persona": {"character_name": "Лера", "memory_enabled": "false"},
        "n_ctx": 8192,
        "max_tokens": 128,
    }
    data.update(kwargs)
    return AIProfile(**data)


class CompanionV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_companion_memory_store_handles_corrupt_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "companion_memory.json"
            path.write_text("{broken", encoding="utf-8")
            store = CompanionMemoryStore(path)

            self.assertEqual(store.load(), [])
            first = store.add("любит лаконичные ответы", category="preference")
            second = store.add("любит лаконичные ответы", category="preference")

            self.assertIsNotNone(first)
            self.assertEqual(first.id, second.id)
            self.assertEqual(len(store.load()), 1)

    def test_explicit_memory_extraction_does_not_store_every_message(self):
        self.assertEqual(extract_explicit_memory("Привет, как дела?"), "")
        self.assertEqual(
            extract_explicit_memory("Запомни: я люблю тёмный UI"),
            "я люблю тёмный UI",
        )

    def test_companion_context_contains_state_and_memories(self):
        with tempfile.TemporaryDirectory() as td:
            store = CompanionMemoryStore(Path(td) / "companion_memory.json")
            store.add("пользователь любит короткие ответы", category="preference")

            context = build_companion_context(
                {
                    "character_name": "Лера",
                    "current_mood": "игривое",
                    "companion_mode": "support",
                    "tenderness": "8",
                    "playfulness": "6",
                    "memory_enabled": "true",
                },
                memory_store=store,
            )

            self.assertIn("Companion State v2", context)
            self.assertIn("Mode: support", context)
            self.assertIn("Mood: игривое", context)
            self.assertIn("пользователь любит короткие ответы", context)

    def test_prompt_builder_injects_companion_state_context(self):
        profile = companion_profile(persona={"character_name": "Лера", "companion_mode": "ideas"})

        with mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ===\nMode: ideas"):
            built = prompt_builder.build(profile, "Привет")

        self.assertIn("Ты Лера.", built.system)
        self.assertIn("Mode: ideas", built.system)
        self.assertIn("Mode: ideas", built.formatted)
        self.assertIn("Never repeat the user's message", built.system)

    def test_prompt_roles_are_ordered_for_companion(self):
        profile = companion_profile()

        with mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            built = prompt_builder.build(
                profile,
                "latest-user",
                history=[("old-user", "old-assistant")],
            )

        self.assertLess(built.formatted.index("<|im_start|>system"), built.formatted.index("<|im_start|>user\nold-user"))
        self.assertLess(built.formatted.index("<|im_start|>user\nold-user"), built.formatted.index("<|im_start|>assistant\nold-assistant"))
        self.assertLess(built.formatted.index("<|im_start|>assistant\nold-assistant"), built.formatted.index("<|im_start|>user\nlatest-user"))
        self.assertTrue(built.formatted.rstrip().endswith("<|im_start|>assistant"))

    def test_echo_detector_catches_parenthesized_greeting(self):
        detected, similarity = is_companion_echo_response("Привет", "(Привет)")

        self.assertTrue(detected)
        self.assertGreaterEqual(similarity, 0.99)

    def test_echo_detector_catches_short_question(self):
        detected, _ = is_companion_echo_response("Тут?", "Тут?")

        self.assertTrue(detected)

    def test_echo_detector_normalizes_quotes_stars_and_punctuation(self):
        for response in ["“Привет”", "*Привет*", "'Привет!'", "«Привет»"]:
            with self.subTest(response=response):
                detected, _ = is_companion_echo_response("Привет", response)
                self.assertTrue(detected)
        self.assertEqual(normalize_companion_echo_text("(Тут?)"), normalize_companion_echo_text("Тут"))

    def test_companion_worker_retries_echo_once(self):
        profile = companion_profile()
        model = ScriptedTextModel(["(Привет)", "Привет, я здесь. Как ты?"])
        worker = InferenceWorker(profile, "Привет", history=[])
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)

        with mock.patch("ai.worker.LLAMA_AVAILABLE", True), \
             mock.patch("ai.worker.resolve_model_path", return_value="fake.gguf"), \
             mock.patch("ai.worker.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.worker.acceleration_warning", return_value=""), \
             mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            worker.run()

        self.assertEqual("".join(chunks), "Привет, я здесь. Как ты?")
        self.assertEqual(len(model.prompts), 2)
        self.assertTrue(worker.companion_response_valid)
        self.assertEqual(worker.validated_response_text, "Привет, я здесь. Как ты?")

    def test_companion_worker_blocks_second_echo(self):
        profile = companion_profile()
        model = ScriptedTextModel(["(Тут?)", "Тут?"])
        worker = InferenceWorker(profile, "Тут?", history=[])
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)

        with mock.patch("ai.worker.LLAMA_AVAILABLE", True), \
             mock.patch("ai.worker.resolve_model_path", return_value="fake.gguf"), \
             mock.patch("ai.worker.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.worker.acceleration_warning", return_value=""), \
             mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            worker.run()

        rendered = "".join(chunks)
        self.assertIn("эхо", rendered)
        self.assertFalse(worker.companion_response_valid)
        self.assertEqual(worker.companion_block_reason, "echo")
        self.assertEqual(worker.validated_response_text, "")
        self.assertNotIn("(Тут?)", rendered)

    def test_companion_worker_blocks_empty_second_response(self):
        profile = companion_profile()
        model = ScriptedTextModel(["", ""])
        worker = InferenceWorker(profile, "Привет", history=[])
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)

        with mock.patch("ai.worker.LLAMA_AVAILABLE", True), \
             mock.patch("ai.worker.resolve_model_path", return_value="fake.gguf"), \
             mock.patch("ai.worker.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.worker.acceleration_warning", return_value=""), \
             mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            worker.run()

        self.assertIn("пустой ответ", "".join(chunks))
        self.assertFalse(worker.companion_response_valid)
        self.assertEqual(worker.companion_block_reason, "empty_response")
        self.assertEqual(worker.validated_response_text, "")

    def test_60_turn_companion_prompts_always_include_latest_marker(self):
        profile = companion_profile()
        history: list[tuple[str, str]] = []
        with mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            for i in range(1, 61):
                marker = f"msg-{i:03d}"
                built = prompt_builder.build(profile, f"Ответь на {marker}", history=history)
                self.assertIn(marker, built.formatted)
                self.assertEqual(built.user, f"Ответь на {marker}")
                history.append((f"Ответь на {marker}", f"вижу {marker}"))

    def test_trim_under_small_context_keeps_latest_user_marker(self):
        profile = companion_profile(n_ctx=256, max_tokens=96)
        history = [
            (f"old-user-{i} " + "x" * 300, f"old-assistant-{i} " + "y" * 300)
            for i in range(20)
        ]

        with mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            built = prompt_builder.build(profile, "latest-marker-060", history=history)

        self.assertIn("latest-marker-060", built.formatted)
        self.assertLess(len(built.history), len(history))

    def test_companion_worker_retries_exact_previous_response(self):
        profile = companion_profile()
        model = ScriptedTextModel(["старый ответ", "новый ответ на msg-046"])
        worker = InferenceWorker(
            profile,
            "msg-046",
            history=[("msg-045", "старый ответ")],
        )
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)

        with mock.patch("ai.worker.LLAMA_AVAILABLE", True), \
             mock.patch("ai.worker.resolve_model_path", return_value="fake.gguf"), \
             mock.patch("ai.worker.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.worker.acceleration_warning", return_value=""), \
             mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            worker.run()

        self.assertEqual("".join(chunks), "новый ответ на msg-046")
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("msg-046", model.prompts[0])
        self.assertIn("msg-046", model.prompts[1])

    def test_companion_worker_errors_instead_of_saving_duplicate_response(self):
        profile = companion_profile()
        model = ScriptedTextModel(["старый ответ", "старый ответ"])
        worker = InferenceWorker(profile, "msg-046", history=[("msg-045", "старый ответ")])
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)

        with mock.patch("ai.worker.LLAMA_AVAILABLE", True), \
             mock.patch("ai.worker.resolve_model_path", return_value="fake.gguf"), \
             mock.patch("ai.worker.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.worker.acceleration_warning", return_value=""), \
             mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            worker.run()

        rendered = "".join(chunks)
        self.assertIn("модель повторила предыдущий ответ", rendered)
        self.assertNotEqual(rendered.strip(), "старый ответ")

    def test_compaction_marker_output_is_blocked(self):
        profile = companion_profile()
        model = ScriptedTextModel([
            "[previous repetitive companion reply omitted]",
            "[previous repetitive companion reply omitted]",
        ])
        worker = InferenceWorker(profile, "msg-047", history=[("msg-046", "живой ответ")])
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)

        with mock.patch("ai.worker.LLAMA_AVAILABLE", True), \
             mock.patch("ai.worker.resolve_model_path", return_value="fake.gguf"), \
             mock.patch("ai.worker.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.worker.acceleration_warning", return_value=""), \
             mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            worker.run()

        self.assertFalse(worker.companion_response_valid)
        self.assertEqual(worker.companion_block_reason, "repeat")
        self.assertNotEqual("".join(chunks).strip(), "[previous repetitive companion reply omitted]")

    def test_restored_history_next_user_message_enters_prompt(self):
        profile = companion_profile()
        restored_history = [("msg-059", "ответ на msg-059"), ("msg-060", "ответ на msg-060")]

        with mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            built = prompt_builder.build(profile, "msg-061", history=restored_history)

        self.assertIn("msg-061", built.formatted)
        self.assertIn("<|im_start|>user\nmsg-061", built.formatted)

    def test_memory_summary_does_not_remove_latest_user(self):
        profile = companion_profile(persona={"character_name": "Лера", "memory_enabled": "true"})

        with tempfile.TemporaryDirectory() as td:
            store = CompanionMemoryStore(Path(td) / "companion_memory.json")
            store.add("пользователь любит маркер msg-060", category="preference")
            with mock.patch("core.companion.CompanionMemoryStore", return_value=store):
                built = prompt_builder.build(profile, "live msg-060", history=[("old", "old answer")])

        self.assertIn("пользователь любит маркер msg-060", built.system)
        self.assertIn("live msg-060", built.formatted)

    def test_companion_history_compacts_repetitive_assistant_scripts(self):
        history = [
            (
                f"msg-{i:03d}: reply briefly",
                f"msg-{i:03d}: (Лера выходит из ресторана)\n\nМм... Как мне хочется, так?",
            )
            for i in range(1, 8)
        ]

        compacted = compact_companion_history(history, max_turns=10, repeat_threshold=2)
        assistant_text = "\n".join(assistant for _user, assistant in compacted)

        self.assertEqual(compacted, [])
        self.assertNotIn("[previous repetitive companion reply omitted]", assistant_text)
        self.assertNotIn("Лера выходит из ресторана", assistant_text)

    def test_near_repeat_with_only_marker_changed_triggers_retry(self):
        previous = "msg-045: (Лера выходит из ресторана)\n\nМм... Как мне хочется, так?"
        new = "msg-046: (Лера выходит из ресторана)\n\nМм... Как мне хочется, так?"

        self.assertEqual(
            normalize_companion_reply_for_repeat(previous),
            normalize_companion_reply_for_repeat(new),
        )

        profile = companion_profile()
        model = ScriptedTextModel([new, "msg-046: я с тобой, вижу именно msg-046"])
        worker = InferenceWorker(profile, "msg-046", history=[("msg-045", previous)])
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)

        with mock.patch("ai.worker.LLAMA_AVAILABLE", True), \
             mock.patch("ai.worker.resolve_model_path", return_value="fake.gguf"), \
             mock.patch("ai.worker.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.worker.acceleration_warning", return_value=""), \
             mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            worker.run()

        self.assertEqual("".join(chunks), "msg-046: я с тобой, вижу именно msg-046")
        self.assertEqual(len(model.prompts), 2)

    def test_companion_emits_long_response_after_validation_buffer(self):
        profile = companion_profile()
        model = ChunkedTextModel([
            "Лера отвечает на новое сообщение мягко, но без повтора старого текста. ",
            "Она добавляет вторую мысль, чтобы поток стал видимым в UI раньше финала. ",
            "И завершает коротко.",
        ])
        worker = InferenceWorker(profile, "новое сообщение", history=[("старое", "другой ответ")])
        chunks: list[str] = []
        worker.chunk_received.connect(chunks.append)

        with mock.patch("ai.worker.LLAMA_AVAILABLE", True), \
             mock.patch("ai.worker.resolve_model_path", return_value="fake.gguf"), \
             mock.patch("ai.worker.ModelManager.instance", return_value=FakeManager(model)), \
             mock.patch("ai.worker.acceleration_warning", return_value=""), \
             mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
            worker.run()

        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("поток стал видимым", "".join(chunks))

    def test_companion_logs_prompt_contains_latest_user(self):
        with tempfile.TemporaryDirectory() as td:
            old_logs = app_data.LOGS_DIR
            app_data.LOGS_DIR = Path(td) / "logs"
            try:
                profile = companion_profile()
                model = ScriptedTextModel(["ответ на msg-001"])
                worker = InferenceWorker(profile, "msg-001", history=[])
                with mock.patch("ai.worker.LLAMA_AVAILABLE", True), \
                     mock.patch("ai.worker.resolve_model_path", return_value="fake.gguf"), \
                     mock.patch("ai.worker.ModelManager.instance", return_value=FakeManager(model)), \
                     mock.patch("ai.worker.acceleration_warning", return_value=""), \
                     mock.patch("ai.prompt_builder.build_companion_context", return_value="=== Companion State v2 ==="):
                    worker.run()
                log = (app_data.LOGS_DIR / "zenai.log").read_text(encoding="utf-8")
            finally:
                app_data.LOGS_DIR = old_logs

        self.assertIn("[companion_prompt_ready]", log)
        self.assertIn('contains_latest_user="True"', log)

    def test_persona_editor_exposes_v2_controls_and_serializes_them(self):
        editor = PersonaEditor()
        editor.set_persona({
            "character_name": "Лера",
            "companion_mode": "roleplay",
            "tenderness": "9",
            "memory_enabled": "false",
            "boundaries": "не говорить сухо",
        })

        data = editor.get_persona()

        self.assertEqual(data["companion_mode"], "roleplay")
        self.assertEqual(data["tenderness"], "9")
        self.assertEqual(data["memory_enabled"], "false")
        self.assertEqual(data["boundaries"], "не говорить сухо")
        self.assertGreaterEqual(len(editor.findChildren(QSlider)), 6)
        self.assertTrue(any(isinstance(w, QCheckBox) for w in editor.findChildren(QCheckBox)))
        button_texts = {button.text() for button in editor.findChildren(QPushButton)}
        self.assertIn("Что помнишь?", button_texts)
        self.assertIn("Забыть всё", button_texts)


if __name__ == "__main__":
    unittest.main()
