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

    bg = QColor(Palette.BG_CODE)
    lexer = cls(parent)
    f = mono_font(11)
    lexer.setFont(f)
    lexer.setDefaultPaper(bg)
    lexer.setDefaultColor(QColor(Palette.TEXT_PRIMARY))

    # Все стили — тёмный фон (иначе некоторые токены белые)
    for i in range(128):
        try:
            lexer.setPaper(bg, i)
            lexer.setFont(f, i)
        except Exception:
            pass

    # Раскраска токенов по типам
    def c(style, color):
        try:
            lexer.setColor(QColor(color), style)
        except Exception:
            pass

    KW  = Palette.ACCENT        # ключевые слова  — лавандовый
    STR = "#A5D6A7"              # строки          — зелёный
    NUM = "#F78C6C"              # числа           — оранжевый
    CMT = Palette.TEXT_DIM      # комментарии     — серый
    FN  = "#82AAFF"              # функции/методы  — голубой
    CLS = "#4EC9B0"              # классы/типы     — циан
    DEC = Palette.ACCENT_AMBER  # декораторы      — янтарный
    OP  = Palette.TEXT_SECONDARY # операторы
    TXT = Palette.TEXT_PRIMARY

    if cls is QsciLexerPython:
        c(QsciLexerPython.Default,                  TXT)
        c(QsciLexerPython.Comment,                  CMT)
        c(QsciLexerPython.CommentBlock,             CMT)
        c(QsciLexerPython.Number,                   NUM)
        c(QsciLexerPython.DoubleQuotedString,       STR)
        c(QsciLexerPython.SingleQuotedString,       STR)
        c(QsciLexerPython.TripleDoubleQuotedString, STR)
        c(QsciLexerPython.TripleSingleQuotedString, STR)
        c(QsciLexerPython.Keyword,                  KW)
        c(QsciLexerPython.FunctionMethodName,       FN)
        c(QsciLexerPython.ClassName,                CLS)
        c(QsciLexerPython.Operator,                 OP)
        c(QsciLexerPython.Identifier,               TXT)
        c(QsciLexerPython.Decorator,                DEC)

    elif cls is QsciLexerJavaScript:
        c(QsciLexerJavaScript.Default,              TXT)
        c(QsciLexerJavaScript.Comment,              CMT)
        c(QsciLexerJavaScript.CommentLine,          CMT)
        c(QsciLexerJavaScript.CommentDoc,           CMT)
        c(QsciLexerJavaScript.Number,               NUM)
        c(QsciLexerJavaScript.Keyword,              KW)
        c(QsciLexerJavaScript.DoubleQuotedString,   STR)
        c(QsciLexerJavaScript.SingleQuotedString,   STR)
        c(QsciLexerJavaScript.Operator,             OP)
        c(QsciLexerJavaScript.Identifier,           TXT)
        c(QsciLexerJavaScript.Regex,                STR)

    elif cls is QsciLexerCPP:
        c(QsciLexerCPP.Default,                     TXT)
        c(QsciLexerCPP.Comment,                     CMT)
        c(QsciLexerCPP.CommentLine,                 CMT)
        c(QsciLexerCPP.Number,                      NUM)
        c(QsciLexerCPP.Keyword,                     KW)
        c(QsciLexerCPP.DoubleQuotedString,          STR)
        c(QsciLexerCPP.SingleQuotedString,          STR)
        c(QsciLexerCPP.Operator,                    OP)
        c(QsciLexerCPP.PreProcessor,                DEC)

    elif cls is QsciLexerJSON:
        c(QsciLexerJSON.Default,                    TXT)
        c(QsciLexerJSON.String,                     STR)
        c(QsciLexerJSON.Number,                     NUM)
        c(QsciLexerJSON.Keyword,                    KW)
        c(QsciLexerJSON.Operator,                   OP)
        c(QsciLexerJSON.Error,                      Palette.ACCENT_RED)
        try:
            c(QsciLexerJSON.Property,               FN)
        except Exception:
            pass

    elif cls is QsciLexerCSS:
        c(QsciLexerCSS.Default,                     TXT)
        c(QsciLexerCSS.Comment,                     CMT)
        c(QsciLexerCSS.Tag,                         KW)
        c(QsciLexerCSS.ClassSelector,               CLS)
        c(QsciLexerCSS.IDSelector,                  DEC)
        c(QsciLexerCSS.Attribute,                   FN)
        c(QsciLexerCSS.Value,                       STR)
        c(QsciLexerCSS.Operator,                    OP)

    elif cls is QsciLexerHTML:
        c(QsciLexerHTML.Default,                    TXT)
        c(QsciLexerHTML.Tag,                        KW)
        c(QsciLexerHTML.Attribute,                  FN)
        c(QsciLexerHTML.HTMLDoubleQuotedString,     STR)
        c(QsciLexerHTML.HTMLSingleQuotedString,     STR)
        c(QsciLexerHTML.HTMLComment,                CMT)
        c(QsciLexerHTML.Entity,                     DEC)

    elif cls is QsciLexerBash:
        c(QsciLexerBash.Default,                    TXT)
        c(QsciLexerBash.Comment,                    CMT)
        c(QsciLexerBash.Number,                     NUM)
        c(QsciLexerBash.Keyword,                    KW)
        c(QsciLexerBash.DoubleQuotedString,         STR)
        c(QsciLexerBash.SingleQuotedString,         STR)
        c(QsciLexerBash.Operator,                   OP)
        c(QsciLexerBash.Identifier,                 TXT)
        try:
            c(QsciLexerBash.HereDocumentDelimiter,  DEC)
        except Exception:
            pass

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
        # Убираем ВСЕ margin'ы (включая symbol/fold) — иначе чёрные полосы слева.
        for m in range(5):
            ed.setMarginWidth(m, 0)
            ed.setMarginType(m, QsciScintilla.MarginType.SymbolMargin)
        ed.setMarginsBackgroundColor(QColor(Palette.BG_CODE))
        ed.setMarginWidth(0, 6)  # маленький левый отступ-воздух
        ed.setMarginsBackgroundColor(QColor(Palette.BG_CODE))
        ed.setIndentationGuides(False)
        ed.setCaretLineVisible(False)
        ed.setCaretWidth(0)
        # перенос длинных строк — код не режется справа и нет горизонт. скролла
        ed.setWrapMode(QsciScintilla.WrapMode.WrapWord)
        ed.setWrapIndentMode(QsciScintilla.WrapIndentMode.WrapIndentSame)
        ed.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ed.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ed.setPaper(QColor(Palette.BG_CODE))
        ed.setColor(QColor(Palette.TEXT_PRIMARY))
        ed.setFont(mono_font(11))
        ed.setFrameStyle(0)
        ed.setStyleSheet(f"""
            QsciScintilla {{
                border: none;
                background: {Palette.BG_CODE};
                border-bottom-left-radius: {Spacing.CODE_RADIUS}px;
                border-bottom-right-radius: {Spacing.CODE_RADIUS}px;
            }}
        """)

        lexer = _build_lexer(ed, self._lang)
        if lexer is not None:
            ed.setLexer(lexer)

        ed.setText(self._code)
        self._scintilla_ref = ed  # чтобы Qt не удалил лексер
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
        """Подгоняем высоту под содержимое с учётом переноса строк."""
        fm = self._view.fontMetrics()
        line_h = fm.lineSpacing()
        pad = 16

        # Для Scintilla с word-wrap логических строк мало, а визуальных больше.
        # SCI_LINESONSCREEN не годится (виджет ещё не показан) — оцениваем по
        # ширине: делим длину каждой логической строки на вместимость.
        try:
            from PyQt6.Qsci import QsciScintilla as _Q
            if isinstance(self._view, _Q):
                width_px = max(self._view.viewport().width(), 400)
                char_w = max(fm.horizontalAdvance("m"), 7)
                cols = max(20, (width_px - 16) // char_w)
                visual = 0
                for ln in self._code.split("\n"):
                    visual += max(1, -(-len(ln) // cols))  # ceil div
                visible = min(visual, 28)
                self._view.setFixedHeight(int(line_h * visible + pad))
                return
        except Exception:
            pass

        # plaintext fallback
        lines = max(1, self._code.count("\n") + 1)
        visible = min(lines, 28)
        self._view.setFixedHeight(int(line_h * visible + pad))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # ширина изменилась → wrap пересчитался → подгоняем высоту
        QTimer.singleShot(0, self._fit_height)

    def _on_copy(self) -> None:
        cb = QGuiApplication.clipboard()
        cb.setText(self._code)
        # лёгкая анимация: меняем текст на "Скопировано", возвращаем через 1.2s
        self._copy_btn.setText("✓ Скопировано")
        QTimer.singleShot(1200, lambda: self._copy_btn.setText("Copy"))
