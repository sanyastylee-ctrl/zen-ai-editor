import sys
import requests
import json
from PyQt6.QtWidgets import QMainWindow, QTextEdit, QLineEdit, QVBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal

class OllamaWorker(QThread):
    response_ready = pyqtSignal(str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "qwen2.5:14b",
                "prompt": self.prompt,
                "stream": False
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            text = response.json().get("response", "")
            self.response_ready.emit(text)
        except Exception as e:
            self.response_ready.emit(f"Error: {str(e)}")

class ZenEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zen AI Editor")
        self.resize(1400, 900)

        # Стилизация под современные темные IDE
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: 1px solid #333; padding:
            QLineEdit { background-color: #2D2D2D; color: #FFFFFF; border: 1px solid #444; border-radius: 5px; }
        """)

        self.chat_history = QTextEdit(self)
        self.chat_history.setReadOnly(True)
        self.input_field = QLineEdit(self)
        self.input_field.returnPressed.connect(self.send_message)

        layout = QVBoxLayout()
        layout.addWidget(self.chat_history)
        layout.addWidget(self.input_field)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_file(self, index):
        pass

    def send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return
        self.chat_history.append(f"<b>User:</b> {text}")
        self.input_field.clear()

        self.worker = OllamaWorker(text)
        self.worker.response_ready.connect(self.handle_ai_response)
        self.worker.start()

    def handle_ai_response(self, text):
        self.chat_history.append(f"<span style='color: green;'><b>Qwen:</b> {text}</span>")
