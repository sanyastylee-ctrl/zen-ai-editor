"""
QScintilla code editor factory + Diff dialog.
Логика вынесена из исходного zen_editor.py.
"""

from __future__ import annotations

import difflib
import os

from PyQt6.QtCore import Qt, QRegularExpression
from PyQt6.QtGui import (
    QColor, QFont, QSyntaxHighlighter, QTextCharFormat,
)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
)

from ui.chat.styles import Palette, mono_font, form_controls_qss

try:
    from PyQt6.Qsci import (
        QsciScintilla, QsciLexerPython, QsciLexerCPP, QsciLexerJavaScript,
        QsciLexerHTML, QsciLexerCSS, QsciLexerJSON,
    )
    QSCI_AVAILABLE = True
except ImportError:
    QsciScintilla = None  # type: ignore
    QSCI_AVAILABLE = False


# ============================================================
# SCINTILLA FACTORY
# ============================================================

def make_scintilla_editor() -> "QsciScintilla":
    editor = QsciScintilla()
    code_font = mono_font(11)
    margin_font = mono_font(9)
    editor.setFont(code_font)
    editor.setMarginsFont(margin_font)

    # Margin 0 — номера строк
    editor.setMarginWidth(0, "0000 ")
    editor.setMarginLineNumbers(0, True)
    # Margin 1 — символы (не используем, убираем полностью)
    editor.setMarginWidth(1, 0)
    # Margin 2 — fold (сворачивание)
    editor.setMarginWidth(2, 12)
    editor.setFolding(QsciScintilla.FoldStyle.PlainFoldStyle, 2)

    # Все margins ОДНОГО цвета с фоном кода — никакого белого столба
    bg = QColor(Palette.BG_CODE)
    editor.setMarginsBackgroundColor(bg)
    editor.setMarginsForegroundColor(QColor(Palette.TEXT_DIM))
    editor.setFoldMarginColors(bg, bg)

    # Убираем белые полосы от маркеров отладчика (margins 3-4)
    editor.setMarginWidth(3, 0)
    editor.setMarginWidth(4, 0)

    editor.setCaretLineVisible(True)
    editor.setCaretLineBackgroundColor(QColor(167, 139, 250, 15))
    editor.setCaretForegroundColor(QColor(Palette.ACCENT))
    editor.setPaper(bg)
    editor.setColor(QColor(Palette.TEXT_PRIMARY))
    editor.setSelectionBackgroundColor(QColor(167, 139, 250, 60))
    editor.setSelectionForegroundColor(QColor(Palette.TEXT_PRIMARY))
    editor.setIndentationsUseTabs(False)
    editor.setTabWidth(4)
    editor.setAutoIndent(True)
    editor.setIndentationGuides(True)
    editor.setIndentationGuidesBackgroundColor(bg)
    editor.setIndentationGuidesForegroundColor(QColor(Palette.BORDER))
    editor.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
    editor.setMatchedBraceBackgroundColor(QColor(167, 139, 250, 45))
    editor.setMatchedBraceForegroundColor(QColor(Palette.ACCENT_HOVER))
    editor.setWrapMode(QsciScintilla.WrapMode.WrapNone)
    editor.setAutoCompletionSource(QsciScintilla.AutoCompletionSource.AcsAll)
    editor.setAutoCompletionThreshold(2)
    editor.setAutoCompletionCaseSensitivity(False)
    editor.setStyleSheet(f"""
        QsciScintilla {{
            background: {Palette.BG_CODE};
            color: {Palette.TEXT_PRIMARY};
            border: none;
        }}
        QScrollBar:vertical, QScrollBar:horizontal {{
            background: transparent; border: none;
        }}
        QScrollBar:vertical {{ width: 10px; }}
        QScrollBar:horizontal {{ height: 10px; }}
        QScrollBar::handle {{
            background: rgba(167,139,250,0.35);
            border-radius: 5px;
            min-height: 28px; min-width: 28px;
        }}
        QScrollBar::handle:hover {{ background: rgba(184,163,255,0.55); }}
        QScrollBar::add-line, QScrollBar::sub-line,
        QScrollBar::add-page, QScrollBar::sub-page {{
            background: transparent; border: none; width: 0; height: 0;
        }}
    """)
    return editor


def set_lexer_for_file(editor, file_path: str) -> None:
    if not QSCI_AVAILABLE:
        return
    ext = os.path.splitext(file_path)[1].lower()
    lexer_map = {
        ".py": QsciLexerPython,
        ".js": QsciLexerJavaScript, ".ts": QsciLexerJavaScript,
        ".cpp": QsciLexerCPP, ".c": QsciLexerCPP, ".h": QsciLexerCPP,
        ".html": QsciLexerHTML, ".htm": QsciLexerHTML,
        ".css": QsciLexerCSS,
        ".json": QsciLexerJSON,
    }
    cls = lexer_map.get(ext)
    if not cls:
        editor.setLexer(None)
        return

    lexer = cls(editor)
    lexer.setFont(mono_font(11))
    bg = QColor(Palette.BG_CODE)
    lexer.setDefaultPaper(bg)
    lexer.setDefaultColor(QColor(Palette.TEXT_PRIMARY))

    # Все стили получают тёмный фон — иначе некоторые токены остаются белыми
    for i in range(128):
        try:
            lexer.setPaper(bg, i)
            lexer.setFont(mono_font(11), i)
        except Exception:
            pass

    editor.setLexer(lexer)

    # Раскраска по типам токенов
    _apply_colors(lexer, cls)


def _c(hex_color: str) -> QColor:
    return QColor(hex_color)


def _apply_colors(lexer, cls) -> None:
    """Красит токены лексера в нашу палитру."""

    KW   = Palette.ACCENT           # ключевые слова    — лавандовый
    STR  = "#A5D6A7"                 # строки            — зелёный
    NUM  = "#F78C6C"                 # числа             — оранжевый
    CMT  = Palette.TEXT_DIM          # комментарии       — серый
    FN   = "#82AAFF"                 # функции/методы    — голубой
    CLS  = "#4EC9B0"                 # классы/типы       — циан
    DEC  = Palette.ACCENT_AMBER      # декораторы        — янтарный
    OP   = Palette.TEXT_SECONDARY    # операторы         — светло-серый
    TXT  = Palette.TEXT_PRIMARY      # обычный текст

    def c(style, color):
        try:
            lexer.setColor(_c(color), style)
        except Exception:
            pass

    if cls is QsciLexerPython:
        c(QsciLexerPython.Default,             TXT)
        c(QsciLexerPython.Comment,             CMT)
        c(QsciLexerPython.CommentBlock,        CMT)
        c(QsciLexerPython.Number,              NUM)
        c(QsciLexerPython.DoubleQuotedString,  STR)
        c(QsciLexerPython.SingleQuotedString,  STR)
        c(QsciLexerPython.TripleDoubleQuotedString, STR)
        c(QsciLexerPython.TripleSingleQuotedString, STR)
        c(QsciLexerPython.Keyword,             KW)
        c(QsciLexerPython.FunctionMethodName,  FN)
        c(QsciLexerPython.ClassName,           CLS)
        c(QsciLexerPython.Operator,            OP)
        c(QsciLexerPython.Identifier,          TXT)
        c(QsciLexerPython.Decorator,           DEC)

    elif cls is QsciLexerJavaScript:
        c(QsciLexerJavaScript.Default,         TXT)
        c(QsciLexerJavaScript.Comment,         CMT)
        c(QsciLexerJavaScript.CommentLine,     CMT)
        c(QsciLexerJavaScript.CommentDoc,      CMT)
        c(QsciLexerJavaScript.Number,          NUM)
        c(QsciLexerJavaScript.Keyword,         KW)
        c(QsciLexerJavaScript.DoubleQuotedString, STR)
        c(QsciLexerJavaScript.SingleQuotedString, STR)
        c(QsciLexerJavaScript.Operator,        OP)
        c(QsciLexerJavaScript.Identifier,      TXT)
        c(QsciLexerJavaScript.Regex,           STR)

    elif cls is QsciLexerCPP:
        c(QsciLexerCPP.Default,                TXT)
        c(QsciLexerCPP.Comment,                CMT)
        c(QsciLexerCPP.CommentLine,            CMT)
        c(QsciLexerCPP.CommentDoc,             CMT)
        c(QsciLexerCPP.Number,                 NUM)
        c(QsciLexerCPP.Keyword,                KW)
        c(QsciLexerCPP.DoubleQuotedString,     STR)
        c(QsciLexerCPP.SingleQuotedString,     STR)
        c(QsciLexerCPP.Operator,               OP)
        c(QsciLexerCPP.Identifier,             TXT)
        c(QsciLexerCPP.PreProcessor,           DEC)
        c(QsciLexerCPP.UUID,                   STR)

    elif cls is QsciLexerJSON:
        c(QsciLexerJSON.Default,               TXT)
        c(QsciLexerJSON.String,                STR)
        c(QsciLexerJSON.Number,                NUM)
        c(QsciLexerJSON.Keyword,               KW)
        c(QsciLexerJSON.Operator,              OP)
        c(QsciLexerJSON.Error,                 Palette.ACCENT_RED)
        try:
            c(QsciLexerJSON.Property,          FN)
        except Exception:
            pass

    elif cls is QsciLexerCSS:
        c(QsciLexerCSS.Default,                TXT)
        c(QsciLexerCSS.Comment,                CMT)
        c(QsciLexerCSS.Tag,                    KW)
        c(QsciLexerCSS.ClassSelector,          CLS)
        c(QsciLexerCSS.IDSelector,             DEC)
        c(QsciLexerCSS.Attribute,              FN)
        c(QsciLexerCSS.Value,                  STR)
        c(QsciLexerCSS.Operator,               OP)
        c(QsciLexerCSS.PseudoClass,            KW)

    elif cls is QsciLexerHTML:
        c(QsciLexerHTML.Default,               TXT)
        c(QsciLexerHTML.Tag,                   KW)
        c(QsciLexerHTML.Attribute,             FN)
        c(QsciLexerHTML.HTMLDoubleQuotedString, STR)
        c(QsciLexerHTML.HTMLSingleQuotedString, STR)
        c(QsciLexerHTML.HTMLComment,           CMT)
        c(QsciLexerHTML.Entity,                DEC)


# ============================================================
# FALLBACK PYTHON HIGHLIGHTER (если нет QScintilla)
# ============================================================

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document) -> None:
        super().__init__(document)
        self.rules = []

        kw = QTextCharFormat(); kw.setForeground(QColor("#569CD6"))
        cl = QTextCharFormat(); cl.setForeground(QColor("#4EC9B0"))
        fn = QTextCharFormat(); fn.setForeground(QColor("#DCDCAA"))
        st = QTextCharFormat(); st.setForeground(QColor("#CE9178"))
        nm = QTextCharFormat(); nm.setForeground(QColor("#B5CEA8"))
        cm = QTextCharFormat(); cm.setForeground(QColor("#6A9955"))

        keywords = [
            "def", "class", "import", "from", "return", "if", "elif", "else",
            "try", "except", "for", "while", "in", "is", "and", "or", "not",
            "with", "as", "pass", "True", "False", "None", "raise", "yield",
            "lambda", "global", "nonlocal", "del", "assert", "finally",
            "continue", "break",
        ]
        for w in keywords:
            self.rules.append((QRegularExpression(f"\\b{w}\\b"), kw))
        self.rules += [
            (QRegularExpression(r"\b[0-9]+\.?[0-9]*\b"), nm),
            (QRegularExpression(r'"[^"\n]*"'), st),
            (QRegularExpression(r"'[^'\n]*'"), st),
            (QRegularExpression(r"#[^\n]*"), cm),
            (QRegularExpression(r"\bclass\b\s+([A-Za-z0-9_]+)"), cl),
            (QRegularExpression(r"\bdef\b\s+([A-Za-z0-9_]+)"), fn),
            (QRegularExpression(r"\b[A-Za-z0-9_]+\s*(?=\()"), fn),
        ]

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self.rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                if m.lastCapturedIndex() == 1:
                    self.setFormat(m.capturedStart(1), m.capturedLength(1), fmt)
                else:
                    self.setFormat(m.capturedStart(), m.capturedLength(), fmt)


# ============================================================
# DIFF DIALOG
# ============================================================

class DiffApplyDialog(QDialog):
    def __init__(self, old_code: str, new_code: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Diff — предпросмотр изменений")
        self.resize(900, 600)
        self.setStyleSheet(form_controls_qss() + f"""
            QDialog {{ background-color: {Palette.BG_APP}; color: {Palette.TEXT_PRIMARY}; }}
            QTextEdit {{ background-color: {Palette.BG_CODE}; color: {Palette.TEXT_PRIMARY};
                        border: 1px solid {Palette.BORDER}; font-family: "{mono_font(12).family()}"; font-size: 12px; }}
            QPushButton {{
                background-color: {Palette.ACCENT}; color: white;
            }}
            QPushButton:hover {{ background-color: {Palette.ACCENT_HOVER}; }}
            QPushButton#reject {{
                background-color: transparent; color: {Palette.TEXT_SECONDARY};
                border: 1px solid {Palette.BORDER};
            }}
            QPushButton#reject:hover {{ background-color: rgba(167,139,250,0.06); }}
        """)

        self.accepted_code = new_code

        layout = QVBoxLayout(self)
        label = QLabel("Зелёный — добавлено, красный — удалено. Принять?")
        label.setStyleSheet(f"color:{Palette.TEXT_SECONDARY}; font-size:12px; padding:4px;")
        layout.addWidget(label)

        self.diff_view = QTextEdit()
        self.diff_view.setReadOnly(True)
        layout.addWidget(self.diff_view)

        btns = QHBoxLayout()
        btns.addStretch()
        apply_btn = QPushButton("✔ Применить")
        apply_btn.clicked.connect(self.accept)
        reject_btn = QPushButton("✕ Отменить")
        reject_btn.setObjectName("reject")
        reject_btn.clicked.connect(self.reject)
        btns.addWidget(apply_btn)
        btns.addWidget(reject_btn)
        layout.addLayout(btns)

        self._render_diff(old_code, new_code)

    def _render_diff(self, old: str, new: str) -> None:
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile="текущий", tofile="из чата", lineterm="",
        ))

        html_lines = []
        for line in diff:
            esc = (line.replace("&", "&amp;")
                       .replace("<", "&lt;").replace(">", "&gt;"))
            if line.startswith("+") and not line.startswith("+++"):
                html_lines.append(f"<span style='background-color:#0D2D0D; color:#3FB950;'>{esc}</span>")
            elif line.startswith("-") and not line.startswith("---"):
                html_lines.append(f"<span style='background-color:#2D0D0D; color:#F85149;'>{esc}</span>")
            elif line.startswith("@@"):
                html_lines.append(f"<span style='color:#58A6FF;'>{esc}</span>")
            else:
                html_lines.append(f"<span style='color:#8B949E;'>{esc}</span>")
        self.diff_view.setHtml("<pre style='margin:0;'>" + "".join(html_lines) + "</pre>")
