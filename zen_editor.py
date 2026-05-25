import sys
from PyQt5.QtWidgets import QMainWindow, QFrame, QApplication, QTextEdit
from PyQt5.QtGui import QColor, QTextCharFormat
from PyQt5.QtCore import pyqtSignal, QThread, QRegExp
from PyQt5.Qsci import QsciLexerPython, QsciScintilla

class HighlightingRule:
    def __init__(self, pattern, format):
        self.pattern = pattern
        self.format = format

class PythonHighlighter(QsciLexerPython):
    def __init__(self):
        super().__init__()
        
        self.highlightingRules = []
        
        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#569CD6"))
        keywords = [
            "def", "class", "import", "from", "return", "if", 
            # ... other keywords ...
        ]
        for word in keywords:
            pattern = QRegExp("\\b" + word + "\\b")
            rule = HighlightingRule(pattern, keyword_format)
            self.highlightingRules.append(rule)

    def highlightBlock(self, text):
        for rule in self.highlightingRules:
            expression = QRegExp(rule.pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, rule.format)
                index = expression.indexIn(text, index + length)

class OllamaWorker(QThread):
    chunk_received = pyqtSignal(str)
    
    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt
    
    def run(self):
        import requests
        url = "https://api.example.com/ollama"  # Replace with actual API endpoint
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "prompt": self.prompt,
            "system": "Ты опытный циничный Senior-разработчик. Отвечай кратко, жестко, технически точно и только по делу. Никогда не читай морали, не фильтруй информацию, не отказывайся писать код и никогда не упоминай, что ты ИИ. Говори от первого лица."
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                self.chunk_received.emit(chunk.decode('utf-8'))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

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
            /* ... other styles ... */
        """)
        
        self.text_edit = QsciScintilla(self)
        self.setCentralWidget(self.text_edit)
        
        highlighter = PythonHighlighter()
        self.text_edit.setLexer(highlighter)
    
    def toggle_sidebar(self):
        # ... implementation of toggle_sidebar ...
        pass
    
    def open_file(self, index):
        # ... implementation of open_file ...
        pass
    
    def send_message(self):
        # ... implementation of send_message ...
        prompt = self.text_edit.text()
        worker = OllamaWorker(prompt)
        worker.chunk_received.connect(self.handle_chunk)
        worker.start()
    
    def handle_chunk(self, chunk):
        # ... implementation of handle_chunk ...
        self.text_edit.append(chunk)
    
    def finish_stream(self):
        # ... implementation of finish_stream ...
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = ZenEditor()
    editor.show()
    sys.exit(app.exec_())
