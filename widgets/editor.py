import os
from PyQt6.QtGui import QFont, QColor, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtCore import QRegularExpression

try:
    from PyQt6.Qsci import (QsciScintilla, QsciLexerPython, QsciLexerCPP,
                             QsciLexerJavaScript, QsciLexerHTML, QsciLexerCSS, QsciLexerJSON)
    QSCI_AVAILABLE = True
except ImportError:
    QSCI_AVAILABLE = False

def make_scintilla_editor() -> "QsciScintilla":
    editor = QsciScintilla()
    editor.setFont(QFont("Consolas", 11))
    editor.setMarginsFont(QFont("Consolas", 9))
    editor.setMarginType(0, QsciScintilla.MarginType.NumberMargin)
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
    if not QSCI_AVAILABLE: return
    ext = os.path.splitext(file_path)[1].lower()
    lexer_map = {
        '.py': QsciLexerPython, '.js': QsciLexerJavaScript, '.ts': QsciLexerJavaScript,
        '.cpp': QsciLexerCPP, '.c': QsciLexerCPP, '.h': QsciLexerCPP,
        '.html': QsciLexerHTML, '.htm': QsciLexerHTML,
        '.css': QsciLexerCSS, '.json': QsciLexerJSON,
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

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.rules = []
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
            self.rules.append((QRegularExpression(w), kw))
        self.rules += [
            (QRegularExpression("\\b[0-9]+\\.?[0-9]*\\b"), nm),
            (QRegularExpression('"[^"\\n]*"'), st),
            (QRegularExpression("'[^'\\n]*'"), st),
            (QRegularExpression("#[^\n]*"), cm),
            (QRegularExpression("\\bclass\\b\\s+([A-Za-z0-9_]+)"), cl),
            (QRegularExpression("\\bdef\\b\\s+([A-Za-z0-9_]+)"), fn),
            (QRegularExpression("\\b[A-Za-z0-9_]+\\s*(?=\\()"), fn),
        ]

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                if m.lastCapturedIndex() == 1:
                    self.setFormat(m.capturedStart(1), m.capturedLength(1), fmt)
                else:
                    self.setFormat(m.capturedStart(), m.capturedLength(), fmt)