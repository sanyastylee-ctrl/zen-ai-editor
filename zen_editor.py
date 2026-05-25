import sys
import os
import requests
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QSplitter, QWidget, 
                             QVBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                             QFrame, QHBoxLayout, QTreeView)
from PyQt6.QtGui import QFont, QFileSystemModel
from PyQt6.QtCore import Qt, QThread, pyqtSignal

class OllamaWorker(QThread):
    chunk_received = pyqtSignal(str)
    stream_finished = pyqtSignal()

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": "qwen2.5:14b",
                "prompt": self.prompt,
                "stream": True
            }, stream=True)
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        self.chunk_received.emit(data['response'])
        except Exception as e:
            self.chunk_received.emit(f"Ошибка: {str(e)}")
        
        self.stream_finished.emit()

class ZenEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zen AI Editor")
        self.resize(1200, 800)
        self.setMinimumSize(800, 600)

        # Премиальная темная тема
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QFrame { background-color: #252526; border: none; }
            QTextEdit { 
                background-color: #1E1E1E; color: #D4D4D4; 
                border: none; padding: 10px; font-size: 14px;
            }
            QLineEdit { 
                background-color: #3C3C3C; color: #FFFFFF; 
                border: 1px solid #555555; border-radius: 6px; 
                padding: 10px; font-size: 14px;
            }
            QPushButton { 
                background-color: #0E639C; color: white; 
                border-radius: 6px; padding: 8px 15px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1177BB; }
            QSplitter::handle { background-color: #333333; width: 2px; }
            QTreeView { 
                background-color: #252526; color: #CCCCCC; 
                border: none; font-size: 14px;
            }
            QTreeView::item:hover { background-color: #2A2D2E; }
            QTreeView::item:selected { background-color: #37373D; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)

        # === 1. ЛЕВАЯ ШТОРКА (Сайдбар с деревом файлов) ===
        self.sidebar = QFrame()
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(os.getcwd())
        
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(os.getcwd()))
        
        # Скрываем лишние колонки и заголовок
        self.tree_view.hideColumn(1)
        self.tree_view.hideColumn(2)
        self.tree_view.hideColumn(3)
        self.tree_view.setHeaderHidden(True) 
        
        self.tree_view.doubleClicked.connect(self.open_file)
        sidebar_layout.addWidget(self.tree_view)
        
        self.sidebar.hide()

        # === 2. ПРАВАЯ ЗОНА (Рабочая) ===
        right_zone = QWidget()
        right_layout = QVBoxLayout(right_zone)
        right_layout.setContentsMargins(10, 10, 10, 10)

        top_bar = QHBoxLayout()
        self.toggle_btn = QPushButton("☰ Проект")
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        top_bar.addWidget(self.toggle_btn)
        top_bar.addStretch()
        right_layout.addLayout(top_bar)

        self.work_splitter = QSplitter(Qt.Orientation.Horizontal)
        right_layout.addWidget(self.work_splitter)

        # -- Зона Чата --
        chat_zone = QWidget()
        chat_layout = QVBoxLayout(chat_zone)
        chat_layout.setContentsMargins(0, 0, 10, 0)
        
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setPlaceholderText("Здесь будет диалог с Qwen...")
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Спроси ИИ или дай задачу... (Enter для отправки)")
        self.chat_input.returnPressed.connect(self.send_message)
        
        chat_layout.addWidget(self.chat_history)
        chat_layout.addWidget(self.chat_input)

        # -- Зона Редактора --
        self.code_editor = QTextEdit()
        self.code_editor.setFont(QFont("Consolas", 11))
        self.code_editor.setPlaceholderText("# Чистый холст для кода...")

        self.work_splitter.addWidget(chat_zone)
        self.work_splitter.addWidget(self.code_editor)
        self.work_splitter.setSizes([400, 800])

        self.main_splitter.addWidget(self.sidebar)
        self.main_splitter.addWidget(right_zone)
        self.main_splitter.setSizes([250, 950])

    def toggle_sidebar(self):
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def open_file(self, index):
        file_path = self.file_model.filePath(index)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.code_editor.setPlainText(content)
            except Exception as e:
                self.chat_history.append(f"<b style='color:red;'>Ошибка открытия:</b> {str(e)}")

    def send_message(self):
        text = self.chat_input.text().strip()
        if text:
            current_code = self.code_editor.toPlainText().strip()

            if current_code:
                full_prompt = f"Контекст (текущий код в редакторе):\n{current_code}\n\nВопрос: {text}"
            else:
                full_prompt = text

            self.chat_history.append(f"<b style='color:#569CD6;'>Ты:</b> {text}")
            self.chat_input.clear()

            self.worker = OllamaWorker(full_prompt)
            self.worker.chunk_received.connect(self.handle_chunk)
            self.worker.stream_finished.connect(self.finish_stream)
            self.worker.start()
    
    def handle_chunk(self, chunk):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(chunk)
        self.chat_history.setTextCursor(cursor)

    def finish_stream(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ZenEditor()
    window.show()
    sys.exit(app.exec())
