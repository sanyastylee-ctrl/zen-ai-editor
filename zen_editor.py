import sys
import requests
import json
import os

from PyQt6.QtWidgets import (QApplication, QMainWindow, QSplitter, QWidget, 
                             QVBoxLayout, QTextEdit, QLineEdit, QHBoxLayout, 
                             QTreeView, QSizePolicy)
from PyQt6.QtGui import (QFont, QTextCursor, QTextCharFormat, QColor, 
                         QSyntaxHighlighter, QFileSystemModel)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRegularExpression

class OllamaWorker(QThread):
    chunk_received = pyqtSignal(str)
    stream_finished = pyqtSignal()  # ДОБАВИЛ ЭТОТ СИГНАЛ
    
    def __init__(self, full_prompt):
        super().__init__()
        self.full_prompt = full_prompt

    def run(self):
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": "qwen2.5:14b",
                    "system": "Ты опытный циничный Senior-разработчик. Отвечай кратко, жестко, технически точно и только по делу. Никогда не читай морали, не фильтруй информацию, не отказывайся писать код и никогда не упоминай, что ты ИИ. Говори от первого лица.",
                    "prompt": self.full_prompt,
                    "stream": True
                },
                stream=True
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        self.chunk_received.emit(data['response'])
            self.stream_finished.emit() # Сообщаем, что закончили
        except Exception:
            self.stream_finished.emit()

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.rules = []
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#569CD6"))
        for word in [r'\bdef\b', r'\bclass\b', r'\bimport\b', r'\breturn\b', r'\bif\b', r'\bfor\b']:
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
        
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: none; font-family: 'Consolas', monospace; font-size: 14px; }
            QLineEdit { background-color: #2D2D2D; color: #E1E1E1; border: 1px solid #3E3E3E; padding: 10px; border-radius: 4px; }
            QTreeView { background-color: #1E1E1E; color: #CCCCCC; border: none; font-size: 13px; }
            QSplitter::handle { background-color: #333333; }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(os.getcwd())
        self.tree = QTreeView()
        self.tree.setModel(self.file_model)
        self.tree.setRootIndex(self.file_model.index(os.getcwd()))
        self.tree.setMinimumWidth(220)
        self.tree.doubleClicked.connect(self.open_file)
        
        chat_zone = QWidget()
        chat_layout = QVBoxLayout(chat_zone)
        chat_layout.setContentsMargins(10, 10, 10, 10)
        chat_layout.setSpacing(10)
        
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ввод команды...")
        self.chat_input.returnPressed.connect(self.send_message)
        
        chat_layout.addWidget(self.chat_history)
        chat_layout.addWidget(self.chat_input)
        
        self.code_editor = QTextEdit()
        self.highlighter = PythonHighlighter(self.code_editor.document())
        
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(chat_zone)
        self.splitter.addWidget(self.code_editor)
        
        layout.addWidget(self.splitter)

    def open_file(self, index):
        path = self.file_model.filePath(index)
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.code_editor.setPlainText(f.read())

    def send_message(self):
        text = self.chat_input.text().strip()
        if not text: return
        self.chat_history.append(f"\n<b>Ты:</b> {text}\n<b>Qwen:</b>")
        self.chat_input.clear()
        
        full_prompt = f"Контекст: {self.code_editor.toPlainText()}\nВопрос: {text}"
        self.worker = OllamaWorker(full_prompt)
        self.worker.chunk_received.connect(self.handle_chunk)
        self.worker.stream_finished.connect(self.finish_stream)
        self.worker.start()

    def handle_chunk(self, chunk):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        self.chat_history.setTextCursor(cursor)
        self.chat_history.ensureCursorVisible()

    def finish_stream(self):
        self.chat_history.append("\n" + "-"*20)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ZenEditor()
    window.show()
    sys.exit(app.exec())