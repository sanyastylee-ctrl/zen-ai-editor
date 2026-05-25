import sys
import os
import requests
import json

from PyQt6.QtWidgets import (QApplication, QMainWindow, QSplitter, QWidget, 
                             QVBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                             QFrame, QHBoxLayout, QTreeView)
from PyQt6.QtGui import QFont, QFileSystemModel, QTextCursor
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Поток для связи с Ollama, чтобы не вешать GUI
class OllamaWorker(QThread):
    chunk_received = pyqtSignal(str)
    
    def __init__(self, prompt, code_context):
        super().__init__()
        self.prompt = prompt
        self.code_context = code_context

    def run(self):
        try:
            full_prompt = f"Контекст кода:\n{self.code_context}\n\nЗапрос: {self.prompt}"
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen2.5:14b", "prompt": full_prompt, "stream": True},
                stream=True
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        self.chunk_received.emit(data['response'])
        except Exception as e:
            self.chunk_received.emit(f"\nОшибка подключения к Ollama: {str(e)}")

class ZenEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zen AI Editor")
        self.resize(1200, 800)
        self.setMinimumSize(800, 600)
        
        # Стили
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: none; padding: 10px; font-size: 14px; }
            QLineEdit { background-color: #3C3C3C; color: #FFFFFF; border: 1px solid #555555; border-radius: 6px; padding: 10px; }
            QPushButton { background-color: #0E639C; color: white; border-radius: 6px; padding: 8px 15px; }
            QSplitter::handle { background-color: #333333; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)

        # Сайдбар
        self.sidebar = QFrame()
        sidebar_layout = QVBoxLayout(self.sidebar)
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(os.getcwd())
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(os.getcwd()))
        self.tree_view.hideColumn(1); self.tree_view.hideColumn(2); self.tree_view.hideColumn(3); self.tree_view.setHeaderHidden(True)
        self.tree_view.doubleClicked.connect(self.open_file)
        sidebar_layout.addWidget(self.tree_view)
        
        # Правая зона
        right_zone = QWidget()
        right_layout = QVBoxLayout(right_zone)
        
        self.toggle_btn = QPushButton("Проект")
        self.toggle_btn.clicked.connect(lambda: self.sidebar.setVisible(not self.sidebar.isVisible()))
        right_layout.addWidget(self.toggle_btn)

        self.work_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self.send_message)
        
        chat_col = QVBoxLayout()
        chat_col.addWidget(self.chat_history); chat_col.addWidget(self.chat_input)
        chat_widget = QWidget(); chat_widget.setLayout(chat_col)
        
        self.code_editor = QTextEdit()
        self.work_splitter.addWidget(chat_widget)
        self.work_splitter.addWidget(self.code_editor)
        right_layout.addWidget(self.work_splitter)

        self.main_splitter.addWidget(self.sidebar)
        self.main_splitter.addWidget(right_zone)

    def open_file(self, index):
        file_path = self.file_model.filePath(index)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.code_editor.setPlainText(f.read())

    def send_message(self):
        text = self.chat_input.text().strip()
        if text:
            self.chat_history.append(f"<b>Ты:</b> {text}")
            self.chat_input.clear()
            # Запускаем поток
            self.worker = OllamaWorker(text, self.code_editor.toPlainText())
            self.worker.chunk_received.connect(self.update_chat)
            self.worker.start()

    def update_chat(self, chunk):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        self.chat_history.setTextCursor(cursor)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ZenEditor()
    window.show()
    sys.exit(app.exec())
