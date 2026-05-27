from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QLabel

from ui.chat.message_widget import MessageWidget


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


if __name__ == "__main__":
    unittest.main()
