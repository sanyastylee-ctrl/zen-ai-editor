import sys
import os
import re
import threading
import subprocess
import difflib
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QSplitter, QWidget,
                             QVBoxLayout, QTextEdit, QLineEdit, QPushButton,
                             QFrame, QHBoxLayout, QTreeView, QComboBox,
                             QDialog, QFormLayout, QDoubleSpinBox, QSpinBox,
                             QLabel, QFileDialog, QTabWidget, QMessageBox,
                             QProgressBar, QScrollArea, QCheckBox, QSizePolicy)
from PyQt6.QtGui import (QFont, QFileSystemModel, QTextCursor,
                         QSyntaxHighlighter, QTextCharFormat, QColor,
                         QShortcut, QKeySequence)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRegularExpression, QTimer

try:
    from PyQt6.Qsci import (QsciScintilla, QsciLexerPython, QsciAPIs,
                             QsciLexerCPP, QsciLexerJavaScript,
                             QsciLexerHTML, QsciLexerCSS, QsciLexerJSON)
    QSCI_AVAILABLE = True
except ImportError:
    QSCI_AVAILABLE = False

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# RAG: попробуем faiss + sentence_transformers
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


# ============================================================
# МЕНЕДЖЕР МОДЕЛИ
# ============================================================
class ModelManager:
    _lock = threading.Lock()
    _model = None
    _model_path = ""

    @classmethod
    def get_model(cls, path):
        with cls._lock:
            if cls._model_path != path or cls._model is None:
                cls._model = None
                cls._model = Llama(
                    model_path=path,
                    n_gpu_layers=-1,
                    n_ctx=4096,
                    verbose=False
                )
                cls._model_path = path
            return cls._model


# ============================================================
# TOKEN BUDGET MANAGER
# ============================================================
class TokenBudgetManager:
    """Обрезает контекст по реальному n_ctx модели, а не по символам."""

    CHARS_PER_TOKEN = 3.5

    def __init__(self, n_ctx=4096, system_reserve=512, response_reserve=1024):
        self.n_ctx = n_ctx
        self.system_reserve = system_reserve
        self.response_reserve = response_reserve

    @property
    def context_budget(self):
        return self.n_ctx - self.system_reserve - self.response_reserve

    def estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / self.CHARS_PER_TOKEN))

    def trim_context(self, code_context: str, prompt: str, system_prompt: str) -> tuple[str, bool]:
        sys_tokens = self.estimate_tokens(system_prompt)
        prompt_tokens = self.estimate_tokens(prompt)
        available = self.context_budget - sys_tokens - prompt_tokens

        if available <= 0:
            return "", True

        context_tokens = self.estimate_tokens(code_context)
        if context_tokens <= available:
            return code_context, False

        max_chars = int(available * self.CHARS_PER_TOKEN)
        trimmed = code_context[:max_chars]
        return trimmed, True

    def get_usage_pct(self, code_context: str, prompt: str, system_prompt: str) -> int:
        total = (self.estimate_tokens(system_prompt) +
                 self.estimate_tokens(prompt) +
                 self.estimate_tokens(code_context))
        return min(100, int(total / self.n_ctx * 100))


# ============================================================
# RAG — Project Context Indexing с сохранением на диск
# ============================================================
class ProjectRAG:
    def __init__(self):
        self._model = None
        self._index = None
        self._chunks = []
        self._chunk_meta = []

    def get_storage_dir(self):
        d = os.path.join(os.getcwd(), '.zen_ai')
        os.makedirs(d, exist_ok=True)
        return d

    def _ensure_model(self):
        if self._model is None and RAG_AVAILABLE:
            self._model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_index(self) -> int:
        if not RAG_AVAILABLE: return 0
        d = self.get_storage_dir()
        idx_path = os.path.join(d, 'project.faiss')
        meta_path = os.path.join(d, 'meta.json')
        chunks_path = os.path.join(d, 'chunks.json')
        
        if os.path.exists(idx_path) and os.path.exists(meta_path) and os.path.exists(chunks_path):
            try:
                self._index = faiss.read_index(idx_path)
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self._chunk_meta = json.load(f)
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self._chunks = json.load(f)
                return len(self._chunks)
            except Exception:
                pass
        return 0

    def save_index(self):
        if not RAG_AVAILABLE or not self._index: return
        d = self.get_storage_dir()
        faiss.write_index(self._index, os.path.join(d, 'project.faiss'))
        with open(os.path.join(d, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(self._chunk_meta, f)
        with open(os.path.join(d, 'chunks.json'), 'w', encoding='utf-8') as f:
            json.dump(self._chunks, f)

    def index_project(self, root_dir: str, extensions=('.py', '.js', '.ts', '.md', '.txt'),
                      chunk_size=40, overlap=10) -> int:
        if not RAG_AVAILABLE:
            return 0
        self._ensure_model()
        self._chunks = []
        self._chunk_meta = []

        for dirpath, _, files in os.walk(root_dir):
            if any(skip in dirpath for skip in ['.git', '__pycache__', 'node_modules', '.venv', '.zen_ai']):
                continue
            for fname in files:
                if not any(fname.endswith(ext) for ext in extensions):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    for i in range(0, len(lines), chunk_size - overlap):
                        chunk = ''.join(lines[i:i + chunk_size])
                        if chunk.strip():
                            self._chunks.append(chunk)
                            self._chunk_meta.append((fpath, i + 1))
                except Exception:
                    pass

        if not self._chunks:
            return 0

        embeddings = self._model.encode(self._chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype='float32')
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings)
        
        self.save_index()
        return len(self._chunks)

    def search(self, query: str, top_k=5) -> str:
        if not RAG_AVAILABLE or self._index is None or not self._chunks:
            return ""
        self._ensure_model()
        q_emb = self._model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype='float32')
        distances, indices = self._index.search(q_emb, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._chunks):
                fpath, line = self._chunk_meta[idx]
                rel = os.path.relpath(fpath)
                results.append(f"# {rel} (строка {line})\n{self._chunks[idx]}")
        return "\n\n---\n\n".join(results)


# ============================================================
# SANDBOX — встроенный терминал
# ============================================================
class SandboxWorker(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, command: str, cwd: str):
        super().__init__()
        self.command = command
        self.cwd = cwd
        self._proc = None
        self._stop = False

    def stop(self):
        self._stop = True
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def run(self):
        try:
            self._proc = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                text=True,
                bufsize=1
            )
            for line in self._proc.stdout:
                if self._stop:
                    break
                self.output_signal.emit(line)
            self._proc.wait()
            self.finished_signal.emit(self._proc.returncode)
        except Exception as e:
            self.output_signal.emit(f"[Ошибка запуска]: {str(e)}\n")
            self.finished_signal.emit(-1)


class SandboxWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 10))
        self.output.setStyleSheet(
            "background-color:#0D0D0D; color:#00FF88; border:none; padding:6px;"
        )
        layout.addWidget(self.output)

        cmd_row = QHBoxLayout()
        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText("$ команда... (Enter)")
        self.cmd_input.setFont(QFont("Consolas", 11))
        self.cmd_input.returnPressed.connect(self.run_command)
        cmd_row.addWidget(self.cmd_input)

        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setFixedWidth(70)
        self.run_btn.clicked.connect(self.run_command)
        cmd_row.addWidget(self.run_btn)

        self.kill_btn = QPushButton("✕ Kill")
        self.kill_btn.setFixedWidth(70)
        self.kill_btn.setObjectName("stop_btn")
        self.kill_btn.setEnabled(False)
        self.kill_btn.clicked.connect(self.kill_process)
        cmd_row.addWidget(self.kill_btn)

        self.clear_btn = QPushButton("🗑")
        self.clear_btn.setFixedWidth(40)
        self.clear_btn.setObjectName("secondary")
        self.clear_btn.clicked.connect(self.output.clear)
        cmd_row.addWidget(self.clear_btn)

        layout.addLayout(cmd_row)
        self._worker = None
        self._cwd = os.getcwd()

    def set_cwd(self, path: str):
        self._cwd = path

    def run_command(self):
        cmd = self.cmd_input.text().strip()
        if not cmd:
            return
        self.output.append(f"<span style='color:#569CD6;'>$ {cmd}</span>")
        self.cmd_input.clear()

        if self._worker and self._worker.isRunning():
            self._worker.stop()

        self._worker = SandboxWorker(cmd, self._cwd)
        self._worker.output_signal.connect(self._on_output)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()
        self.run_btn.setEnabled(False)
        self.kill_btn.setEnabled(True)

    def kill_process(self):
        if self._worker:
            self._worker.stop()

    def _on_output(self, text):
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(text)
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())

    def _on_finished(self, code):
        color = "#888888" if code == 0 else "#CE9178"
        self.output.append(f"<span style='color:{color};'>[exit {code}]</span><br>")
        self.run_btn.setEnabled(True)
        self.kill_btn.setEnabled(False)

    def run_code_file(self, file_path: str):
        cmd = f"{sys.executable} \"{file_path}\""
        self.cmd_input.setText(cmd)
        self.run_command()


# ============================================================
# DIFF APPLY DIALOG
# ============================================================
class DiffApplyDialog(QDialog):
    def __init__(self, old_code: str, new_code: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diff — предпросмотр изменений")
        self.resize(900, 600)
        self.setStyleSheet("""
            QDialog { background-color: #1E1E1E; color: #D4D4D4; }
            QTextEdit { background-color: #0D1117; color: #E6EDF3;
                        border: none; font-family: Consolas; font-size: 12px; }
            QPushButton {
                background-color: #238636; color: white;
                border-radius: 4px; padding: 6px 18px; font-weight: bold; border: none;
            }
            QPushButton:hover { background-color: #2EA043; }
            QPushButton#reject_btn {
                background-color: #3C3C3C; color: #D4D4D4;
                border: 1px solid #555555;
            }
            QPushButton#reject_btn:hover { background-color: #4A4A4A; }
        """)
        self.accepted_code = new_code

        layout = QVBoxLayout(self)
        label = QLabel("Зелёный — добавлено, красный — удалено. Принять изменения?")
        label.setStyleSheet("color:#888; font-size:12px; padding:4px;")
        layout.addWidget(label)

        self.diff_view = QTextEdit()
        self.diff_view.setReadOnly(True)
        layout.addWidget(self.diff_view)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.apply_btn = QPushButton("✔ Применить")
        self.apply_btn.clicked.connect(self.accept)
        reject_btn = QPushButton("✕ Отменить")
        reject_btn.setObjectName("reject_btn")
        reject_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(reject_btn)
        layout.addLayout(btn_row)

        self._render_diff(old_code, new_code)

    def _render_diff(self, old: str, new: str):
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = list(difflib.unified_diff(old_lines, new_lines,
                                         fromfile="текущий", tofile="из чата", lineterm=""))
        html_lines = []
        for line in diff:
            line_esc = (line.replace('&', '&amp;')
                           .replace('<', '&lt;')
                           .replace('>', '&gt;'))
            if line.startswith('+') and not line.startswith('+++'):
                html_lines.append(f"<span style='background-color:#0D2D0D; color:#3FB950;'>{line_esc}</span>")
            elif line.startswith('-') and not line.startswith('---'):
                html_lines.append(f"<span style='background-color:#2D0D0D; color:#F85149;'>{line_esc}</span>")
            elif line.startswith('@@'):
                html_lines.append(f"<span style='color:#58A6FF;'>{line_esc}</span>")
            else:
                html_lines.append(f"<span style='color:#8B949E;'>{line_esc}</span>")
        self.diff_view.setHtml("<pre style='margin:0;'>" + "".join(html_lines) + "</pre>")


# ============================================================
# QScintilla-редактор
# ============================================================
def make_scintilla_editor() -> "QsciScintilla":
    editor = QsciScintilla()
    editor.setFont(QFont("Consolas", 11))
    editor.setMarginsFont(QFont("Consolas", 9))
    editor.setMarginWidth(0, "0000")
    editor.setMarginLineNumbers(0, True)
    editor.setMarginsBackgroundColor(QColor("#252526"))
    editor.setMarginsForegroundColor(QColor("#858585"))
    editor.setFolding(QsciScintilla.FoldStyle.BoxedTreeFoldStyle, 2)
    editor.setFoldMarginColors(QColor("#252526"), QColor("#252526"))
    editor.setCaretLineVisible(True)
    editor.setCaretLineBackgroundColor(QColor("#2A2D2E"))
    editor.setCaretForegroundColor(QColor("#AEAFAD"))
    editor.setPaper(QColor("#1E1E1E"))
    editor.setColor(QColor("#D4D4D4"))
    editor.setIndentationsUseTabs(False)
    editor.setTabWidth(4)
    editor.setAutoIndent(True)
    editor.setIndentationGuides(True)
    editor.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
    editor.setMatchedBraceBackgroundColor(QColor("#3B514D"))
    editor.setMatchedBraceForegroundColor(QColor("#4DC9B0"))
    editor.setWrapMode(QsciScintilla.WrapMode.WrapNone)
    editor.setAutoCompletionSource(QsciScintilla.AutoCompletionSource.AcsAll)
    editor.setAutoCompletionThreshold(2)
    editor.setAutoCompletionCaseSensitivity(False)
    return editor

def set_lexer_for_file(editor: "QsciScintilla", file_path: str):
    if not QSCI_AVAILABLE:
        return
    ext = os.path.splitext(file_path)[1].lower()
    lexer_map = {
        '.py': QsciLexerPython, '.js': QsciLexerJavaScript, '.ts': QsciLexerJavaScript,
        '.cpp': QsciLexerCPP, '.c': QsciLexerCPP, '.h': QsciLexerCPP,
        '.html': QsciLexerHTML, '.htm': QsciLexerHTML, '.css': QsciLexerCSS, '.json': QsciLexerJSON,
    }
    lexer_cls = lexer_map.get(ext)
    if lexer_cls:
        lexer = lexer_cls(editor)
        lexer.setFont(QFont("Consolas", 11))
        lexer.setDefaultPaper(QColor("#1E1E1E"))
        lexer.setDefaultColor(QColor("#D4D4D4"))
        editor.setLexer(lexer)
    else:
        editor.setLexer(None)


class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.highlightingRules = []
        kw = QTextCharFormat(); kw.setForeground(QColor("#569CD6"))
        cl = QTextCharFormat(); cl.setForeground(QColor("#4EC9B0"))
        fn = QTextCharFormat(); fn.setForeground(QColor("#DCDCAA"))
        st = QTextCharFormat(); st.setForeground(QColor("#CE9178"))
        nm = QTextCharFormat(); nm.setForeground(QColor("#B5CEA8"))
        cm = QTextCharFormat(); cm.setForeground(QColor("#6A9955"))
        for w in ["\\bdef\\b","\\bclass\\b","\\bimport\\b","\\bfrom\\b","\\breturn\\b",
                  "\\bif\\b","\\belif\\b","\\belse\\b","\\btry\\b","\\bexcept\\b",
                  "\\bfor\\b","\\bwhile\\b","\\bin\\b","\\bis\\b","\\band\\b",
                  "\\bor\\b","\\bnot\\b","\\bwith\\b","\\bas\\b","\\bpass\\b",
                  "\\bTrue\\b","\\bFalse\\b","\\bNone\\b","\\braise\\b","\\byield\\b",
                  "\\blambda\\b","\\bglobal\\b","\\bnonlocal\\b","\\bdel\\b",
                  "\\bassert\\b","\\bfinally\\b","\\bcontinue\\b","\\bbreak\\b"]:
            self.highlightingRules.append((QRegularExpression(w), kw))
        self.highlightingRules += [
            (QRegularExpression("\\b[0-9]+\\.?[0-9]*\\b"), nm),
            (QRegularExpression('"[^"\\n]*"'), st),
            (QRegularExpression("'[^'\\n]*'"), st),
            (QRegularExpression("#[^\n]*"), cm),
            (QRegularExpression("\\bclass\\b\\s+([A-Za-z0-9_]+)"), cl),
            (QRegularExpression("\\bdef\\b\\s+([A-Za-z0-9_]+)"), fn),
            (QRegularExpression("\\b[A-Za-z0-9_]+\\s*(?=\\()"), fn),
        ]

    def highlightBlock(self, text):
        for pattern, fmt in self.highlightingRules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                if m.lastCapturedIndex() == 1:
                    self.setFormat(m.capturedStart(1), m.capturedLength(1), fmt)
                else:
                    self.setFormat(m.capturedStart(), m.capturedLength(), fmt)


# ============================================================
# НАСТРОЙКИ
# ============================================================
class SettingsDialog(QDialog):
    def __init__(self, parent=None, current_settings=None, available_models=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки ИИ")
        self.setMinimumWidth(480)
        self.setStyleSheet("""
            QDialog { background-color: #252526; color: #D4D4D4; }
            QLabel { color: #D4D4D4; font-size: 13px; }
            QComboBox, QTextEdit, QDoubleSpinBox, QSpinBox {
                background-color: #3C3C3C; color: #FFFFFF;
                border: 1px solid #555555; border-radius: 4px; padding: 5px;
            }
            QPushButton {
                background-color: #0E639C; color: white;
                border-radius: 4px; padding: 6px 15px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1177BB; }
            QCheckBox { color: #D4D4D4; }
        """)
        self.settings = current_settings or {}
        layout = QFormLayout(self)
        layout.setSpacing(10)

        self.coder_combo = QComboBox()
        self.assistant_combo = QComboBox()
        if available_models:
            self.coder_combo.addItems(available_models)
            self.assistant_combo.addItems(available_models)
            if self.settings.get('coder_model') in available_models:
                self.coder_combo.setCurrentText(self.settings['coder_model'])
            if self.settings.get('assistant_model') in available_models:
                self.assistant_combo.setCurrentText(self.settings['assistant_model'])

        self.sys_prompt_edit = QTextEdit()
        self.sys_prompt_edit.setMaximumHeight(70)
        self.sys_prompt_edit.setPlainText(self.settings.get('system_prompt', 'Ты — полезный ИИ-ассистент. Отвечай по существу.'))

        self.coder_prompt_edit = QTextEdit()
        self.coder_prompt_edit.setMaximumHeight(70)
        self.coder_prompt_edit.setPlainText(self.settings.get('coder_system_prompt', 'Ты — опытный программист. Пиши чистый рабочий код.'))

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(self.settings.get('temperature', 0.7))

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(256, 8192)
        self.max_tokens_spin.setSingleStep(256)
        self.max_tokens_spin.setValue(self.settings.get('max_tokens', 2048))

        self.n_ctx_spin = QSpinBox()
        self.n_ctx_spin.setRange(512, 32768)
        self.n_ctx_spin.setSingleStep(512)
        self.n_ctx_spin.setValue(self.settings.get('n_ctx', 4096))
        self.n_ctx_spin.setToolTip("Размер контекста модели (n_ctx). Должен совпадать с реальным ctx модели.")

        self.diff_check = QCheckBox("Показывать diff перед вставкой кода")
        self.diff_check.setChecked(self.settings.get('diff_before_apply', True))

        self.rag_check = QCheckBox("Использовать RAG при запросах (если проиндексирован)")
        self.rag_check.setChecked(self.settings.get('use_rag', False))

        layout.addRow("Модель Кодера:", self.coder_combo)
        layout.addRow("Модель Ассистента:", self.assistant_combo)
        layout.addRow("Промпт Ассистента:", self.sys_prompt_edit)
        layout.addRow("Промпт Кодера:", self.coder_prompt_edit)
        layout.addRow("Температура:", self.temp_spin)
        layout.addRow("Макс. токенов:", self.max_tokens_spin)
        layout.addRow("n_ctx модели:", self.n_ctx_spin)
        layout.addRow("", self.diff_check)
        layout.addRow("", self.rag_check)

        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self.accept)
        layout.addRow("", save_btn)

    def get_settings(self):
        return {
            'coder_model': self.coder_combo.currentText(),
            'assistant_model': self.assistant_combo.currentText(),
            'system_prompt': self.sys_prompt_edit.toPlainText(),
            'coder_system_prompt': self.coder_prompt_edit.toPlainText(),
            'temperature': self.temp_spin.value(),
            'max_tokens': self.max_tokens_spin.value(),
            'n_ctx': self.n_ctx_spin.value(),
            'diff_before_apply': self.diff_check.isChecked(),
            'use_rag': self.rag_check.isChecked(),
        }


# ============================================================
# ВОРКЕР ГЕНЕРАЦИИ (С динамическими шаблонами)
# ============================================================
class LlamaCppWorker(QThread):
    chunk_received = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, prompt, code_context, model_path,
                 system_prompt="", coder_system_prompt="",
                 temperature=0.7, max_tokens=2048, mode="coder"):
        super().__init__()
        self.prompt = prompt
        self.code_context = code_context
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.coder_system_prompt = coder_system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = mode
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        if Llama is None:
            self.chunk_received.emit("\n[Ошибка: llama-cpp-python не установлен]\n")
            self.finished_signal.emit()
            return
        try:
            model = ModelManager.get_model(self.model_path)
            model_name_lower = self.model_path.lower()

            # Собираем части сообщения
            if self.mode == "coder":
                sys_p = self.coder_system_prompt or "Ты — опытный программист. Пиши чистый рабочий код."
                user_msg = f"Контекст кода:\n{self.code_context}\n\nЗапрос: {self.prompt}"
            else:
                sys_p = self.system_prompt
                user_msg = self.prompt
                if self.code_context:
                    user_msg += f"\n\nПрикреплённые данные:\n{self.code_context}"

            # Динамические шаблоны промптов
            if "llama-3" in model_name_lower or "llama3" in model_name_lower:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_p}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
                formatted_prompt = f"<s>[INST] {sys_p}\n\n{user_msg} [/INST]"
            elif "gemma" in model_name_lower:
                formatted_prompt = (
                    f"<start_of_turn>user\n{sys_p}\n\n{user_msg}<end_of_turn>\n"
                    f"<start_of_turn>model\n"
                )
            else:
                # ChatML (Qwen, Dolphin и др.)
                formatted_prompt = (
                    f"<|im_start|>system\n{sys_p}<|im_end|>\n"
                    f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

            output = model(
                formatted_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            
            for chunk in output:
                if self._stop_requested:
                    self.chunk_received.emit("\n<i style='color:#888888;'>[Генерация остановлена]</i>")
                    break
                text = chunk['choices'][0]['text']
                if text:
                    self.chunk_received.emit(text)
        except Exception as e:
            self.chunk_received.emit(f"\n[Критическая ошибка: {str(e)}]\n")
        finally:
            self.finished_signal.emit()


# ============================================================
# ВОРКЕР ИНДЕКСАЦИИ RAG
# ============================================================
class RagIndexWorker(QThread):
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, rag: ProjectRAG, root_dir: str):
        super().__init__()
        self.rag = rag
        self.root_dir = root_dir

    def run(self):
        try:
            count = self.rag.index_project(self.root_dir)
            self.finished_signal.emit(count)
        except Exception as e:
            self.error_signal.emit(str(e))


# ============================================================
# ГЛАВНОЕ ОКНО
# ============================================================
class ZenEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zen AI Editor")
        self.resize(1400, 900)
        self.setMinimumSize(900, 600)

        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QFrame { background-color: #252526; border: none; }
            QTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: none; padding: 10px; font-size: 14px; }
            QLineEdit { background-color: #3C3C3C; color: #FFFFFF; border: 1px solid #555555; border-radius: 6px; padding: 10px; font-size: 14px; }
            QPushButton { background-color: #0E639C; color: white; border-radius: 6px; padding: 8px 15px; font-weight: bold; font-size: 13px; border: none; }
            QPushButton:hover { background-color: #1177BB; }
            QPushButton:disabled { background-color: #3A3A3A; color: #666666; }
            QPushButton#secondary { background-color: #3C3C3C; color: #D4D4D4; border: 1px solid #555555; }
            QPushButton#secondary:hover { background-color: #4A4A4A; }
            QPushButton#secondary:disabled { background-color: #2E2E2E; color: #555555; }
            QPushButton#stop_btn:enabled { background-color: #8B2020; color: white; }
            QPushButton#stop_btn:enabled:hover { background-color: #A52828; }
            QPushButton#stop_btn:disabled { background-color: #3A3A3A; color: #666666; }
            QPushButton#green_btn { background-color: #1B4F1B; color: #4EC9B0; border: 1px solid #2A7A2A; }
            QPushButton#green_btn:hover { background-color: #236B23; }
            QComboBox { background-color: #3C3C3C; color: #FFFFFF; border: 1px solid #555555; border-radius: 6px; padding: 5px 10px; font-size: 13px; min-width: 150px; }
            QComboBox QAbstractItemView { background-color: #2D2D2D; color: #FFFFFF; selection-background-color: #0E639C; }
            QSplitter::handle { background-color: #333333; width: 2px; }
            QTreeView { background-color: #252526; color: #CCCCCC; border: none; font-size: 13px; }
            QTreeView::item:hover { background-color: #2A2D2E; }
            QTreeView::item:selected { background-color: #37373D; }
            QTabWidget::pane { border: none; background: #1E1E1E; }
            QTabBar::tab { background: #2D2D2D; color: #888888; padding: 6px 16px; border: none; }
            QTabBar::tab:selected { background: #1E1E1E; color: #D4D4D4; border-bottom: 2px solid #0E639C; }
            QTabBar::tab:hover { background: #3C3C3C; color: #D4D4D4; }
            QProgressBar { background: #3C3C3C; border-radius: 3px; height: 6px; text-align: right; font-size: 10px; color: #888888; }
            QProgressBar::chunk { background: #0E639C; border-radius: 3px; }
            QLabel#token_label { color: #888888; font-size: 11px; }
            QLabel#token_warn { color: #CE9178; font-size: 11px; }
        """)

        self.app_settings = {
            'coder_model': '', 'assistant_model': '',
            'system_prompt': 'Ты — полезный ИИ-ассистент. Отвечай по существу.',
            'coder_system_prompt': 'Ты — опытный программист. Пиши чистый рабочий код. Отвечай кратко.',
            'temperature': 0.7, 'max_tokens': 2048, 'n_ctx': 4096,
            'diff_before_apply': True, 'use_rag': False,
        }
        self.attached_files_content = ""
        self.available_models = []
        self.current_file_path = ""
        self.worker = None
        self._chat_raw_log = []
        self._current_ai_response = ""

        self.token_budget = TokenBudgetManager(n_ctx=4096, response_reserve=2048)
        self.rag = ProjectRAG()
        self._rag_indexed = False
        self._rag_worker = None

        self._build_ui()
        self.scan_local_models()
        
        # Загрузка RAG из кэша
        loaded = self.rag.load_index()
        if loaded > 0:
            self._rag_indexed = True
            self.rag_status_label.setText(f"RAG: {loaded} чанков (кэш)")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)

        # Сайдбар
        self.sidebar = QFrame()
        sb_layout = QVBoxLayout(self.sidebar)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.setSpacing(4)
        rag_row = QHBoxLayout()
        self.index_btn = QPushButton("⟳ Индекс RAG")
        self.index_btn.setObjectName("green_btn")
        self.index_btn.setToolTip("Проиндексировать проект (faiss + sentence-transformers)")
        self.index_btn.clicked.connect(self.start_rag_indexing)
        rag_row.addWidget(self.index_btn)
        self.rag_status_label = QLabel("RAG: –")
        self.rag_status_label.setObjectName("token_label")
        rag_row.addWidget(self.rag_status_label)
        sb_layout.addLayout(rag_row)

        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(os.getcwd())
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(os.getcwd()))
        self.tree_view.hideColumn(1); self.tree_view.hideColumn(2); self.tree_view.hideColumn(3)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.doubleClicked.connect(self.open_file)
        sb_layout.addWidget(self.tree_view)
        self.sidebar.hide()

        # Правая зона
        right_zone = QWidget()
        right_layout = QVBoxLayout(right_zone)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(6)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)
        self.toggle_btn = QPushButton("☰ Проект")
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        top_bar.addWidget(self.toggle_btn)
        self.save_btn = QPushButton("💾 Сохранить")
        self.save_btn.clicked.connect(self.save_file)
        top_bar.addWidget(self.save_btn)
        self.run_file_btn = QPushButton("▶ Запустить")
        self.run_file_btn.setObjectName("green_btn")
        self.run_file_btn.clicked.connect(self.run_current_file)
        top_bar.addWidget(self.run_file_btn)
        top_bar.addStretch()
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Режим: Кодер", "Режим: Ассистент"])
        top_bar.addWidget(self.mode_selector)
        self.settings_btn = QPushButton("⚙ Настройки")
        self.settings_btn.clicked.connect(self.open_settings)
        top_bar.addWidget(self.settings_btn)
        right_layout.addLayout(top_bar)

        budget_row = QHBoxLayout()
        self.token_label = QLabel("Токены: 0 / 4096")
        self.token_label.setObjectName("token_label")
        budget_row.addWidget(self.token_label)
        self.token_bar = QProgressBar()
        self.token_bar.setRange(0, 100)
        self.token_bar.setValue(0)
        self.token_bar.setFormat("%p%")
        budget_row.addWidget(self.token_bar)
        right_layout.addLayout(budget_row)

        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_file)

        self.work_splitter = QSplitter(Qt.Orientation.Horizontal)
        right_layout.addWidget(self.work_splitter)

        # Чат
        chat_zone = QWidget()
        chat_layout = QVBoxLayout(chat_zone)
        chat_layout.setContentsMargins(0, 0, 10, 0)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setPlaceholderText("Здесь будет диалог с ИИ...")

        input_layout = QHBoxLayout()
        input_layout.setSpacing(6)
        self.attach_btn = QPushButton("📎")
        self.attach_btn.setObjectName("secondary")
        self.attach_btn.setFixedWidth(40)
        self.attach_btn.clicked.connect(self.attach_file)
        input_layout.addWidget(self.attach_btn)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Спроси ИИ или дай задачу... (Enter)")
        self.chat_input.returnPressed.connect(self.send_message)
        self.chat_input.textChanged.connect(self._update_token_bar)
        input_layout.addWidget(self.chat_input)
        self.stop_btn = QPushButton("⏹ Стоп")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)
        input_layout.addWidget(self.stop_btn)
        
        self.apply_btn = QPushButton("↙ Вставить код")
        self.apply_btn.setObjectName("secondary")
        self.apply_btn.setToolTip("Если в редакторе выделен текст, заменит его. Если нет — заменит весь файл.")
        self.apply_btn.clicked.connect(self.apply_code_from_chat)
        input_layout.addWidget(self.apply_btn)

        chat_layout.addWidget(self.chat_history)
        chat_layout.addLayout(input_layout)

        # Редактор + Терминал
        right_panel = QTabWidget()
        right_panel.setTabPosition(QTabWidget.TabPosition.South)

        if QSCI_AVAILABLE:
            self.code_editor = make_scintilla_editor()
            self.code_editor.textChanged.connect(self._update_token_bar)
            editor_widget = self.code_editor
        else:
            self.code_editor = QTextEdit()
            self.code_editor.setFont(QFont("Consolas", 11))
            self.code_editor.textChanged.connect(self._update_token_bar)
            self._highlighter = PythonHighlighter(self.code_editor.document())
            editor_widget = self.code_editor

        right_panel.addTab(editor_widget, "📝 Редактор")
        self.sandbox = SandboxWidget()
        right_panel.addTab(self.sandbox, "⚡ Терминал")

        self.work_splitter.addWidget(chat_zone)
        self.work_splitter.addWidget(right_panel)
        self.work_splitter.setSizes([420, 880])
        self.main_splitter.addWidget(self.sidebar)
        self.main_splitter.addWidget(right_zone)
        self.main_splitter.setSizes([250, 1150])

    def _get_editor_text(self) -> str:
        if QSCI_AVAILABLE and isinstance(self.code_editor, QsciScintilla):
            return self.code_editor.text()
        return self.code_editor.toPlainText()

    def _set_editor_text(self, text: str):
        if QSCI_AVAILABLE and isinstance(self.code_editor, QsciScintilla):
            self.code_editor.setText(text)
        else:
            self.code_editor.setPlainText(text)

    def _replace_selection(self, new_text: str):
        if QSCI_AVAILABLE and isinstance(self.code_editor, QsciScintilla):
            self.code_editor.replaceSelectedText(new_text)
        else:
            cursor = self.code_editor.textCursor()
            cursor.insertText(new_text)

    def _update_token_bar(self):
        prompt = self.chat_input.text()
        code_ctx = self._get_editor_text()
        current_mode = "coder" if "Кодер" in self.mode_selector.currentText() else "assistant"
        sys_p = self.app_settings['coder_system_prompt'] if current_mode == "coder" else self.app_settings['system_prompt']

        self.token_budget = TokenBudgetManager(
            n_ctx=self.app_settings.get('n_ctx', 4096),
            response_reserve=self.app_settings.get('max_tokens', 2048)
        )
        pct = self.token_budget.get_usage_pct(code_ctx, prompt, sys_p)
        used = self.token_budget.estimate_tokens(sys_p + prompt + code_ctx)
        n_ctx = self.app_settings.get('n_ctx', 4096)

        self.token_bar.setValue(pct)
        self.token_label.setText(f"Токены: ~{used} / {n_ctx}")

        if pct >= 90:
            self.token_bar.setStyleSheet("QProgressBar::chunk { background: #CE4040; border-radius: 3px; }")
            self.token_label.setObjectName("token_warn")
        elif pct >= 70:
            self.token_bar.setStyleSheet("QProgressBar::chunk { background: #CDA040; border-radius: 3px; }")
            self.token_label.setObjectName("token_label")
        else:
            self.token_bar.setStyleSheet("")
            self.token_label.setObjectName("token_label")
        self.token_label.style().unpolish(self.token_label)
        self.token_label.style().polish(self.token_label)

    def start_rag_indexing(self):
        if not RAG_AVAILABLE:
            self.chat_history.append("<b style='color:#CE9178;'>RAG недоступен: установите faiss-cpu и sentence-transformers.</b><br>")
            return
        self.index_btn.setEnabled(False)
        self.rag_status_label.setText("RAG: индексация...")
        self._rag_worker = RagIndexWorker(self.rag, os.getcwd())
        self._rag_worker.finished_signal.connect(self._on_rag_indexed)
        self._rag_worker.error_signal.connect(self._on_rag_error)
        self._rag_worker.start()

    def _on_rag_indexed(self, count: int):
        self._rag_indexed = count > 0
        self.rag_status_label.setText(f"RAG: {count} чанков")
        self.index_btn.setEnabled(True)
        self.chat_history.append(f"<i style='color:#4EC9B0;'>✓ Проект проиндексирован и сохранен: {count} фрагментов.</i><br>")

    def _on_rag_error(self, err: str):
        self.index_btn.setEnabled(True)
        self.rag_status_label.setText("RAG: ошибка")
        self.chat_history.append(f"<b style='color:#CE9178;'>RAG ошибка: {err}</b><br>")

    def run_current_file(self):
        if self.current_file_path and self.current_file_path.endswith('.py'):
            tab_widget = self.work_splitter.widget(1)
            if isinstance(tab_widget, QTabWidget):
                tab_widget.setCurrentIndex(1)
            self.sandbox.set_cwd(os.path.dirname(self.current_file_path))
            self.sandbox.run_code_file(self.current_file_path)
        elif self.current_file_path:
            self.chat_history.append("<i style='color:#888;'>Запуск поддерживается только для .py файлов.</i><br>")
        else:
            self.chat_history.append("<i style='color:#888;'>Сначала сохраните файл (Ctrl+S).</i><br>")

    def scan_local_models(self):
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_dir): os.makedirs(models_dir)
        if Llama is None:
            self.chat_history.append("<b style='color:#CE9178;'>Внимание: llama-cpp-python не установлена.</b><br>")
            return
        self.available_models = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]
        if self.available_models:
            if not self.app_settings['coder_model']: self.app_settings['coder_model'] = self.available_models[0]
            if not self.app_settings['assistant_model']: self.app_settings['assistant_model'] = self.available_models[0]
        else:
            self.chat_history.append("<b style='color:#CE9178;'>Положите файлы .gguf в папку /models и перезапустите.</b><br>")

    def open_settings(self):
        dialog = SettingsDialog(self, self.app_settings, self.available_models)
        if dialog.exec():
            self.app_settings = dialog.get_settings()
            self.token_budget = TokenBudgetManager(n_ctx=self.app_settings['n_ctx'], response_reserve=self.app_settings['max_tokens'])
            self._update_token_bar()
            self.chat_history.append("<i style='color:#888888;'>Настройки сохранены.</i><br>")

    def attach_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Прикрепить файл")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.attached_files_content += f"\n--- Файл: {os.path.basename(file_path)} ---\n"
                    self.attached_files_content += f.read()[:5000]
                self.chat_history.append(f"<i style='color:#888888;'>📎 {os.path.basename(file_path)} загружен.</i><br>")
            except Exception as e:
                self.chat_history.append(f"<b style='color:#CE9178;'>Ошибка чтения:</b> {str(e)}<br>")

    def toggle_sidebar(self):
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def stop_generation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def save_file(self):
        if self.current_file_path:
            try:
                with open(self.current_file_path, 'w', encoding='utf-8') as f:
                    f.write(self._get_editor_text())
                name = os.path.basename(self.current_file_path)
                self.chat_history.append(f"<i style='color:#888888;'>Сохранено: {name}</i><br>")
                self.setWindowTitle(f"Zen AI Editor — {name}")
            except Exception as e:
                self.chat_history.append(f"<b style='color:#CE9178;'>Ошибка сохранения:</b> {str(e)}<br>")
        else:
            self.save_file_as()

    def save_file_as(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", os.getcwd(), "Python Files (*.py);;All Files (*)")
        if file_path:
            self.current_file_path = file_path
            self.save_file()

    def open_file(self, index):
        file_path = self.file_model.filePath(index)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._set_editor_text(content)
                self.current_file_path = file_path
                self.setWindowTitle(f"Zen AI Editor — {os.path.basename(file_path)}")
                self.sandbox.set_cwd(os.path.dirname(file_path))
                if QSCI_AVAILABLE and isinstance(self.code_editor, QsciScintilla):
                    set_lexer_for_file(self.code_editor, file_path)
            except Exception as e:
                self.chat_history.append(f"<b style='color:#CE9178;'>Ошибка:</b> {str(e)}")

    def send_message(self):
        if self.worker and self.worker.isRunning(): return
        text = self.chat_input.text().strip()
        if not text: return

        current_mode = "coder" if "Кодер" in self.mode_selector.currentText() else "assistant"
        model_key = 'coder_model' if current_mode == "coder" else 'assistant_model'
        selected_model_file = self.app_settings[model_key]

        if not selected_model_file:
            self.chat_history.append("<b style='color:#CE9178;'>Не выбрана модель в настройках.</b><br>")
            return

        code_ctx_raw = self._get_editor_text() if current_mode == "coder" else self.attached_files_content
        sys_p = self.app_settings['coder_system_prompt'] if current_mode == "coder" else self.app_settings['system_prompt']

        self.token_budget = TokenBudgetManager(n_ctx=self.app_settings.get('n_ctx', 4096), response_reserve=self.app_settings.get('max_tokens', 2048))
        code_context, was_trimmed = self.token_budget.trim_context(code_ctx_raw, text, sys_p)
        if was_trimmed:
            self.chat_history.append("<i style='color:#CDA040;'>⚠ Контекст обрезан — превышен бюджет токенов n_ctx.</i><br>")

        if self.app_settings.get('use_rag') and self._rag_indexed:
            rag_context = self.rag.search(text, top_k=3)
            if rag_context:
                code_context = f"[RAG — релевантные фрагменты проекта]\n{rag_context}\n\n[Текущий файл]\n{code_context}"
                self.chat_history.append("<i style='color:#888;'>🔍 RAG: найден релевантный контекст проекта.</i><br>")

        self.chat_history.append(f"<b style='color:#569CD6;'>Ты:</b> {text}<br>")
        self._chat_raw_log.append(f"USER: {text}")
        self.chat_input.clear()
        self.chat_input.setDisabled(True)
        self.mode_selector.setDisabled(True)
        self.stop_btn.setEnabled(True)

        if current_mode == "assistant":
            self.attached_files_content = ""

        model_path = os.path.join(os.getcwd(), "models", selected_model_file)
        self.chat_history.append(f"<b style='color:#4EC9B0;'>[{current_mode} | {selected_model_file}]:</b> ")
        self._current_ai_response = ""

        self.worker = LlamaCppWorker(
            prompt=text, code_context=code_context, model_path=model_path,
            system_prompt=self.app_settings['system_prompt'],
            coder_system_prompt=self.app_settings['coder_system_prompt'],
            temperature=self.app_settings['temperature'], max_tokens=self.app_settings['max_tokens'], mode=current_mode
        )
        self.worker.chunk_received.connect(self.update_chat)
        self.worker.status_signal.connect(self.add_status_msg)
        self.worker.finished_signal.connect(self.unlock_ui)
        self.worker.start()

    def unlock_ui(self):
        self.chat_input.setDisabled(False)
        self.mode_selector.setDisabled(False)
        self.stop_btn.setEnabled(False)
        self.chat_input.setFocus()
        self.chat_history.append("<br>")
        self._chat_raw_log.append(f"AI: {self._current_ai_response}")
        self._update_token_bar()

    def add_status_msg(self, msg):
        self.chat_history.append(msg)

    def update_chat(self, chunk):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        self.chat_history.setTextCursor(cursor)
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())
        self._current_ai_response += chunk

    def apply_code_from_chat(self):
        full_log = "\n".join(self._chat_raw_log)
        blocks = re.findall(r'```[a-zA-Z]*\n(.*?)```', full_log, re.DOTALL)

        if not blocks:
            self.chat_history.append("<br><i style='color:#888888;'>[Блоки кода не найдены]</i><br>")
            self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())
            return

        new_code = blocks[-1].strip()
        
        # ПРОВЕРКА НА ВЫДЕЛЕНИЕ В РЕДАКТОРЕ
        has_selection = False
        old_selected_text = ""
        
        if QSCI_AVAILABLE and isinstance(self.code_editor, QsciScintilla):
            has_selection = self.code_editor.hasSelectedText()
            if has_selection:
                old_selected_text = self.code_editor.selectedText()
        else:
            cursor = self.code_editor.textCursor()
            has_selection = cursor.hasSelection()
            if has_selection:
                old_selected_text = cursor.selectedText()

        # ЕСЛИ ЕСТЬ ВЫДЕЛЕНИЕ — Заменяем только его
        if has_selection:
            if self.app_settings.get('diff_before_apply', True):
                dialog = DiffApplyDialog(old_selected_text, new_code, parent=self)
                if dialog.exec():
                    self._replace_selection(dialog.accepted_code)
                    self.chat_history.append("<br><i style='color:#888888;'>[Выделенный код заменен с diff]</i><br>")
                else:
                    self.chat_history.append("<br><i style='color:#888888;'>[Применение отменено]</i><br>")
            else:
                self._replace_selection(new_code)
                self.chat_history.append("<br><i style='color:#888888;'>[Выделенный код заменен]</i><br>")
        
        # ЕСЛИ НЕТ ВЫДЕЛЕНИЯ — Заменяем весь файл
        else:
            old_code = self._get_editor_text()
            if self.app_settings.get('diff_before_apply', True) and old_code.strip():
                dialog = DiffApplyDialog(old_code, new_code, parent=self)
                if dialog.exec():
                    self._set_editor_text(dialog.accepted_code)
                    self.chat_history.append("<br><i style='color:#888888;'>[Код применён ко всему файлу с diff]</i><br>")
                else:
                    self.chat_history.append("<br><i style='color:#888888;'>[Применение отменено]</i><br>")
            else:
                self._set_editor_text(new_code)
                self.chat_history.append("<br><i style='color:#888888;'>[Код перенесён в редактор]</i><br>")

        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ZenEditor()
    window.show()
    sys.exit(app.exec())