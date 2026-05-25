import sys
import os
import requests
import json

from PyQt6.QtWidgets import (QApplication, QMainWindow, QSplitter, QWidget, 
                             QVBoxLayout, QTextEdit, QLineEdit, 
                             QFrame, QHBoxLayout, QTreeView)
from PyQt6.QtGui import QFileSystemModel, QTextCursor
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Поток для связи с Ollama (асинхронная обработка)
class OllamaWorker(QThread):
    chunk_received = pyqtSignal(str)
    
    def __init__(self, prompt, code_context):
        super().__init__()
        self.prompt = prompt
        self.code_context = code_context

    def run(self):
        try:
            full_prompt = f"Контекст кода:\n{self.code_context}\n\nЗапрос: {self.prompt}"
            # Используем модель 'zen-coder' (создайте её через Modelfile, чтобы убрать цензуру)
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={"model": "zen-coder", "prompt": full_prompt, "stream": True},
                stream=True
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        self.chunk_received.emit(data['response'])
        except Exception as e:
            self.chunk_received.emit(f"\n[Ошибка подключения к Ollama: {e}]")

class ZenEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zen AI Editor")
        self.resize(1400, 900)
        
        # Стилизация под современные темные IDE
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: 1px solid #333; padding: 5px; font-family: 'Consolas', monospace; }
            QLineEdit { background-color: #2D2D2D; color: #FFFFFF; border: 1px solid #444; border-radius: 4px; padding: 8px; }
            QSplitter::handle { background-color: #333333; }
            QTreeView { background-color: #252526; color: #CCCCCC; border: none; }
        """)

        # Главный контейнер
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Главный сплиттер (Дерево | [Чат | Редактор])
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 1. Дерево файлов
        self.tree_view = QTreeView()
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(os.getcwd())
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(os.getcwd()))
        self.tree_view.hideColumn(1); self.tree_view.hideColumn(2); self.tree_view.hideColumn(3)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.doubleClicked.connect(self.open_file)

        # 2. Вложенный сплиттер для Чата и Редактора
        self.work_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Чат
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(5, 5, 5, 5)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Спроси ИИ...")
        self.chat_input.returnPressed.connect(self.send_message)
        chat_layout.addWidget(self.chat_history)
        chat_layout.addWidget(self.chat_input)

        # Редактор кода
        self.code_editor = QTextEdit()
        
        self.work_splitter.addWidget(chat_widget)
        self.work_splitter.addWidget(self.code_editor)
        
        # Настройка весов (Редактор занимает 3/4 правой части)
        self.work_splitter.setStretchFactor(0, 1)
        self.work_splitter.setStretchFactor(1, 3)

        # Добавляем всё в главный сплиттер
        self.main_splitter.addWidget(self.tree_view)
        self.main_splitter.addWidget(self.work_splitter)
        
        # Устанавливаем ширину сайдбара
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 6)
        
        main_layout.addWidget(self.main_splitter)

    def open_file(self, index):
        file_path = self.file_model.filePath(index)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.code_editor.setPlainText(f.read())

    def send_message(self):
        text = self.chat_input.text().strip()
        if text:
            self.chat_history.append(f"<b>Вы:</b> {text}")
            self.chat_input.clear()
            # Запуск обработки в отдельном потоке
            self.worker = OllamaWorker(text, self.code_editor.toPlainText())
            self.worker.chunk_received.connect(self.update_chat)
            self.worker.start()

    def update_chat(self, chunk):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ZenEditor()
    window.show()
    sys.exit(app.exec())