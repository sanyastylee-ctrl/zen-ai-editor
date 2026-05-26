"""
CodeBlockWidget — карточка с подсвечиваемым кодом.

В заголовке — язык и кнопки [Copy] [↙ В редактор].
Подсветка через QScintilla с лексером по языку.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont, QGuiApplication
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
)

from .styles import Palette, mono_font, Spacing

try:
    from PyQt6.Qsci import (
        QsciScintilla, QsciLexerPython, QsciLexerCPP, QsciLexerJavaScript,
        QsciLexerHTML, QsciLexerCSS, QsciLexerJSON, QsciLexerBash,
        QsciLexerMarkdown, QsciLexerYAML, QsciLexerXML, QsciLexerSQL,
    )
    QSCI_AVAILABLE = True
except ImportError:
    QsciScintilla = None  # type: ignore
    QSCI_AVAILABLE = False


_LEXER_MAP: dict[str, str] = {
    "python": "python", "py": "python",
    "javascript": "js", "js": "js", "ts": "js", "typescript": "js", "jsx": "js", "tsx": "js",
    "c": "cpp", "cpp": "cpp", "c++": "cpp", "h": "cpp", "hpp": "cpp",
    "html": "html", "htm": "html",
    "css": "css", "scss": "css",
    "json": "json",
    "bash": "bash", "sh": "bash", "shell": "bash", "zsh": "bash", "fish": "bash",
    "powershell": "bash", "ps1": "bash",
    "markdown": "md", "md": "md",
    "yaml": "yaml", "yml": "yaml",
    "xml": "xml",
    "sql": "sql",
}


def _build_lexer(parent, lang: str):
    if not QSCI_AVAILABLE:
        return None
    kind = _LEXER_MAP.get(lang.lower(), "")
    cls = {
        "python": QsciLexerPython,
        "js": QsciLexerJavaScript,
        "cpp": QsciLexerCPP,
        "html": QsciLexerHTML,
        "css": QsciLexerCSS,
        "json": QsciLexerJSON,
        "bash": QsciLexerBash,
        "md": QsciLexerMarkdown,
        "yaml": QsciLexerYAML,
        "xml": QsciLexerXML,
        "sql": QsciLexerSQL,
    }.get(kind)
    if not cls:
        return None
    lexer = cls(parent)
    f = mono_font(11)
    lexer.setFont(f)
    lexer.setDefaultPaper(QColor(Palette.BG_CODE))
    lexer.setDefaultColor(QColor(Palette.TEXT_PRIMARY))
    return lexer


class CodeBlockWidget(QFrame):
    """Карточка с кодом. Сигнал insert_requested(code) для кнопки 'В редактор'."""

    insert_requested = pyqtSignal(str)

    def __init__(self, code: str, lang: str = "", parent=None) -> None:
        super().__init__(parent)
        self._code = code
        self._lang = (lang or "").strip()

        self.setObjectName("code_card")
        self.setStyleSheet(f"""
            QFrame#code_card {{
                background: {Palette.BG_CODE};
                border: 1px solid {Palette.BORDER};
                border-radius: {Spacing.CODE_RADIUS}px;
            }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ===== HEADER =====
        header = QFrame()
        header.setObjectName("code_header")
        header.setStyleSheet(f"""
            QFrame#code_header {{
                background: {Palette.BG_CODE_HEADER};
                border-top-left-radius: {Spacing.CODE_RADIUS}px;
                border-top-right-radius: {Spacing.CODE_RADIUS}px;
                border-bottom: 1px solid {Palette.BORDER};
            }}
            QLabel {{ color: {Palette.TEXT_SECONDARY}; font-size: 11px; }}
            QPushButton {{
                background: transparent;
                color: {Palette.TEXT_SECONDARY};
                border: none;
                padding: 4px 10px;
                font-size: 11px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                color: {Palette.TEXT_PRIMARY};
                background: rgba(255,255,255,0.05);
            }}
        """)
        h = QHBoxLayout(header)
        h.setContentsMargins(12, 6, 6, 6)
        h.setSpacing(4)

        lang_label = QLabel(self._lang or "plain")
        h.addWidget(lang_label)
        h.addStretch()

        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.clicked.connect(self._on_copy)
        h.addWidget(self._copy_btn)

        insert_btn = QPushButton("↙ В редактор")
        insert_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        insert_btn.clicked.connect(lambda: self.insert_requested.emit(self._code))
        h.addWidget(insert_btn)

        outer.addWidget(header)

        # ===== БЛОК С КОДОМ =====
        if QSCI_AVAILABLE:
            self._view = self._make_scintilla()
        else:
            self._view = self._make_plaintext()
        outer.addWidget(self._view)

    # ---------- API ----------

    def set_code(self, code: str) -> None:
        """Обновить содержимое (для стрима)."""
        self._code = code
        if QSCI_AVAILABLE and isinstance(self._view, QsciScintilla):
            self._view.setText(code)
        else:
            self._view.setPlainText(code)
        self._fit_height()

    # ---------- внутренности ----------

    def _make_scintilla(self):
        ed = QsciScintilla()
        ed.setReadOnly(True)
        ed.setUtf8(True)
        ed.setMarginWidth(0, 0)
        ed.setMarginWidth(1, 0)
        ed.setMarginsBackgroundColor(QColor(Palette.BG_CODE))
        ed.setIndentationGuides(False)
        ed.setCaretLineVisible(False)
        ed.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        ed.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ed.setPaper(QColor(Palette.BG_CODE))
        ed.setColor(QColor(Palette.TEXT_PRIMARY))
        ed.setFont(mono_font(11))
        ed.setFrameStyle(0)
        ed.setWrapMode(QsciScintilla.WrapMode.WrapNone)

        lexer = _build_lexer(ed, self._lang)
        if lexer is not None:
            ed.setLexer(lexer)

        ed.setText(self._code)
        self._scintilla_ref = ed  # чтобы Qt не удалил лексер
        # высота под содержимое
        QTimer.singleShot(0, self._fit_height)
        return ed

    def _make_plaintext(self):
        ed = QPlainTextEdit()
        ed.setReadOnly(True)
        ed.setFont(mono_font(11))
        ed.setStyleSheet(
            f"QPlainTextEdit {{ background: {Palette.BG_CODE};"
            f" color: {Palette.TEXT_PRIMARY}; border: none; padding: 8px 12px; }}"
        )
        ed.setPlainText(self._code)
        ed.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        ed.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        QTimer.singleShot(0, self._fit_height)
        return ed

    def _fit_height(self) -> None:
        """Подгоняем высоту под количество строк (кода обычно мало)."""
        lines = max(1, self._code.count("\n") + 1)
        # ограничение 25 строк по высоте, дальше скролл
        visible_lines = min(lines, 25)
        fm = self._view.fontMetrics()
        line_h = fm.lineSpacing()
        pad = 14
        h = line_h * visible_lines + pad
        self._view.setFixedHeight(h)
        if lines > 25:
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def _on_copy(self) -> None:
        cb = QGuiApplication.clipboard()
        cb.setText(self._code)
        # лёгкая анимация: меняем текст на "Скопировано", возвращаем через 1.2s
        self._copy_btn.setText("✓ Скопировано")
        QTimer.singleShot(1200, lambda: self._copy_btn.setText("Copy"))
