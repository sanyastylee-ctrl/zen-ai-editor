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
    if cls:
        lexer = cls(editor)
        lexer.setFont(QFont("Consolas", 11))
        lexer.setDefaultPaper(QColor("#1E1E1E"))
        lexer.setDefaultColor(QColor("#D4D4D4"))
        editor.setLexer(lexer)
    else:
        editor.setLexer(None)


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
        self.setStyleSheet("""
            QDialog { background-color: #1E1E1E; color: #D4D4D4; }
            QTextEdit { background-color: #0D1117; color: #E6EDF3;
                        border: none; font-family: Consolas; font-size: 12px; }
            QPushButton {
                background-color: #238636; color: white;
                border-radius: 4px; padding: 6px 18px;
                font-weight: bold; border: none;
            }
            QPushButton:hover { background-color: #2EA043; }
            QPushButton#reject {
                background-color: #3C3C3C; color: #D4D4D4;
                border: 1px solid #555555;
            }
            QPushButton#reject:hover { background-color: #4A4A4A; }
        """)

        self.accepted_code = new_code

        layout = QVBoxLayout(self)
        label = QLabel("Зелёный — добавлено, красный — удалено. Принять?")
        label.setStyleSheet("color:#888; font-size:12px; padding:4px;")
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
