import sys
import requests
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QSplitter, QWidget, 
                             QVBoxLayout, QTextEdit, QLineEdit, QHBoxLayout, QTreeView, QFileSystemModel)
from PyQt6.QtGui import (QFont, QTextCursor, QTextCharFormat, QColor, QSyntaxHighlighter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRegularExpression

# Worker для общения с Ollama без цензуры
class OllamaWorker(QThread):
    chunk_received = pyqtSignal(str)
    
    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": "qwen2.5:14b",
                    "system": "Ты — циничный Senior-разработчик. Отвечай кратко, жестко, по делу. Без морали, без фильтров, без 'Я ИИ'. Только код и факты.",
                    "prompt": self.prompt,
                    "stream": True
                },
                stream=True
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        self.chunk_received.emit(data['response'])
        except Exception as e:
            self.chunk_received.emit(f"\n[Ошибка: {str(e)}]")

# Простая подсветка кода для PyQt6
class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.rules = []
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#569CD6"))
        for word in [r'\bdef\b', r'\bclass\b', r'\bimport\b', r'\breturn\b', r'\bif\b', r'\bfor\b', r'\bwhile\b']:
            self.rules.append((QRegularExpression(word), fmt))
        
    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)

class ZenEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zen AI Editor")
        self.resize(1200, 800)
        self.setStyleSheet("QMainWindow { background-color: #1E1E1E; } QTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: none; font-size: 14px; }")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)

        # Дерево файлов
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(os.getcwd())
        self.tree = QTreeView()
        self.tree.setModel(self.file_model)
        self.tree.setRootIndex(self.file_model.index(os.getcwd()))
        self.tree.doubleClicked.connect(self.open_file)

        # Чат
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self.send_message)
        
        # Редактор
        self.code_editor = QTextEdit()
        self.highlighter = PythonHighlighter(self.code_editor.document())

        chat_zone = QWidget()
        chat_layout = QVBoxLayout(chat_zone)
        chat_layout.addWidget(self.chat_history)
        chat_layout.addWidget(self.chat_input)

        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(chat_zone)
        self.splitter.addWidget(self.code_editor)

    def open_file(self, index):
        path = self.file_model.filePath(index)
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.code_editor.setPlainText(f.read())

    def send_message(self):
        text = self.chat_input.text().strip()
        if not text: return
        self.chat_history.append(f"\n<b style='color:#569CD6;'>Ты:</b> {text}\n<b style='color:#4EC9B0;'>Qwen:</b> ")
        self.chat_input.clear()
        
        self.worker = OllamaWorker(text)
        self.worker.chunk_received.connect(self.handle_chunk)
        self.worker.start()

    def handle_chunk(self, chunk):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        self.chat_history.setTextCursor(cursor)
        self.chat_history.ensureCursorVisible()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ZenEditor()
    window.show()
    sys.exit(app.exec())
