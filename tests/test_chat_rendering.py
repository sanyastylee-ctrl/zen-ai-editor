from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QLabel, QPushButton

from core.profiles import AIProfile, ProfileKind
from ui.chat.code_block import CodeBlockWidget
from ui.chat.message_widget import MessageWidget
from ui.chat.styles import Palette
from ui.chat.tool_block import ToolBlockWidget


class MessageWidgetRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_single_chunk_finish_renders_final_text(self):
        record = {
            "role": "assistant",
            "sender": "Ассистент",
            "text": "",
            "profile_kind": "coder",
            "streaming": True,
        }
        widget = MessageWidget(record)

        record["text"] = "Одночастный ответ кодера"
        record["streaming"] = False
        widget.update_record(record)

        labels = [child.text() for child in widget.findChildren(QLabel)]

        self.assertTrue(any("Одночастный ответ кодера" in text for text in labels))

    def test_adjacent_text_blocks_merge_into_one_label(self):
        record = {
            "role": "user",
            "sender": "Ты",
            "text": "Первый абзац\n\nВторой абзац",
            "streaming": False,
        }
        widget = MessageWidget(record)

        content_labels = [
            child.text() for child in widget.findChildren(QLabel)
            if "Первый абзац" in child.text() or "Второй абзац" in child.text()
        ]

        self.assertEqual(len(content_labels), 1)
        self.assertIn("<br><br>", content_labels[0])

    def test_user_card_has_compact_maximum_width(self):
        widget = MessageWidget({
            "role": "user",
            "sender": "Ты",
            "text": "Короткое сообщение",
            "streaming": False,
        })

        self.assertEqual(widget._card.maximumWidth(), 660)

    def test_user_plain_text_label_is_transparent(self):
        widget = MessageWidget({
            "role": "user",
            "sender": "Ты",
            "text": "объясни, что сделал",
            "streaming": False,
        })

        labels = [
            child for child in widget.findChildren(QLabel)
            if "объясни" in child.text()
        ]

        self.assertEqual(len(labels), 1)
        self.assertIn("background:transparent", labels[0].styleSheet())

    def test_user_header_labels_are_transparent(self):
        widget = MessageWidget({
            "role": "user",
            "sender": "Ты",
            "text": "Алло?",
            "time": "2026-05-27T13:28:00",
            "streaming": False,
        })

        labels = {
            child.text(): child for child in widget.findChildren(QLabel)
            if child.text() in {"Ты", "13:28"}
        }

        self.assertIn("Ты", labels)
        self.assertIn("13:28", labels)
        self.assertIn("background:transparent", labels["Ты"].styleSheet())
        self.assertIn("background:transparent", labels["13:28"].styleSheet())

    def test_model_file_action_buttons_are_labeled(self):
        from ui.profile_editor import ProfileEditor

        editor = ProfileEditor(AIProfile(
            id="coder",
            name="Кодер",
            kind=ProfileKind.CODER,
        ))
        buttons = {
            btn.text(): btn for btn in editor.findChildren(QPushButton)
            if btn.objectName() == "secondaryCompact"
        }

        self.assertIn("Обновить", buttons)
        self.assertIn("Выбрать", buttons)
        self.assertTrue(buttons["Обновить"].text().strip())
        self.assertTrue(buttons["Выбрать"].text().strip())

    def test_settings_styles_use_chat_palette(self):
        from ui.settings_dialog import SettingsDialog
        from ui.profile_editor import ProfileEditor
        from ui.persona_editor import PersonaEditor

        combined = (
            SettingsDialog._stylesheet()
            + ProfileEditor._stylesheet()
            + PersonaEditor().styleSheet()
        )

        self.assertIn(Palette.ACCENT, combined)
        self.assertIn(Palette.BG_CODE, combined)
        self.assertIn("QComboBox::drop-down", combined)
        self.assertIn("width: 0", combined)
        self.assertNotIn("#0E639C", combined)

    def test_scintilla_factory_uses_chat_palette(self):
        from widgets.code_editor import QSCI_AVAILABLE, make_scintilla_editor

        if not QSCI_AVAILABLE:
            self.skipTest("QScintilla is not available")

        editor = make_scintilla_editor()
        self.assertEqual(editor.paper().name().lower(), Palette.BG_CODE.lower())
        self.assertIn(Palette.BG_CODE, editor.styleSheet())
        self.assertIn("167,139,250", editor.styleSheet())

    def test_large_code_block_height_is_bounded(self):
        code = "\n".join(f"print({i})" for i in range(120))
        widget = CodeBlockWidget(code, "python")
        widget._fit_height()

        self.assertLessEqual(widget._view.height(), 620)

    def test_long_success_tool_card_stays_compact(self):
        widget = ToolBlockWidget(
            "run_terminal",
            "$ python main.py list",
            "\n".join(f"line {i}" for i in range(20)) + "\n[exit 0]",
            ok=True,
        )

        self.assertTrue(widget._output_view.isHidden())
        self.assertFalse(widget._toggle_btn.isHidden())


if __name__ == "__main__":
    unittest.main()
