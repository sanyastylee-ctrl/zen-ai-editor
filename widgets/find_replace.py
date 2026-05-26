import re
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QCheckBox
from PyQt6.QtGui import QShortcut, QKeySequence, QTextCursor, QTextDocument
from PyQt6.QtCore import Qt
from widgets.editor import QSCI_AVAILABLE

try:
    from PyQt6.Qsci import QsciScintilla
except ImportError:
    pass

class FindReplaceBar(QWidget):
    def __init__(self, editor_tabs, parent=None):
        super().__init__(parent)
        self.editor_tabs = editor_tabs
        self._replace_mode = False
        self._build_ui()
        self.hide()

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget { background: #2D2D2D; border-top: 1px solid #444; }
            QLineEdit { background:#3C3C3C; color:#FFF; border:1px solid #555;
                        border-radius:4px; padding:4px 8px; font-size:13px; }
            QLineEdit:focus { border-color: #0E639C; }
            QPushButton { background:#3C3C3C; color:#D4D4D4; border:1px solid #555;
                          border-radius:4px; padding:4px 10px; font-size:12px; }
            QPushButton:hover { background:#4A4A4A; }
            QPushButton#apply_btn { background:#0E639C; color:#fff; border:none; }
            QPushButton#apply_btn:hover { background:#1177BB; }
            QLabel { color:#888; font-size:11px; }
            QCheckBox { color:#D4D4D4; font-size:12px; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        find_row = QHBoxLayout(); find_row.setSpacing(4)
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Найти... (Enter / Shift+Enter)")
        self.find_edit.returnPressed.connect(self.find_next)
        self.find_edit.textChanged.connect(self._on_find_text_changed)
        find_row.addWidget(self.find_edit)

        self.case_chk = QCheckBox("Aa")
        self.case_chk.setToolTip("Учитывать регистр")
        find_row.addWidget(self.case_chk)

        self.word_chk = QCheckBox("\\b")
        self.word_chk.setToolTip("Слово целиком")
        find_row.addWidget(self.word_chk)

        self.regex_chk = QCheckBox(".*")
        self.regex_chk.setToolTip("Регулярное выражение")
        find_row.addWidget(self.regex_chk)

        self.prev_btn = QPushButton("↑")
        self.prev_btn.setFixedWidth(28)
        self.prev_btn.setToolTip("Предыдущее (Shift+Enter)")
        self.prev_btn.clicked.connect(self.find_prev)
        find_row.addWidget(self.prev_btn)

        self.next_btn = QPushButton("↓")
        self.next_btn.setFixedWidth(28)
        self.next_btn.setToolTip("Следующее (Enter)")
        self.next_btn.clicked.connect(self.find_next)
        find_row.addWidget(self.next_btn)

        self.result_lbl = QLabel("")
        self.result_lbl.setFixedWidth(80)
        find_row.addWidget(self.result_lbl)

        close_btn = QPushButton("✕")
        close_btn.setFixedWidth(24)
        close_btn.clicked.connect(self.hide)
        find_row.addWidget(close_btn)
        layout.addLayout(find_row)

        self.replace_widget = QWidget()
        repl_row = QHBoxLayout(self.replace_widget)
        repl_row.setContentsMargins(0, 0, 0, 0); repl_row.setSpacing(4)

        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText("Заменить на...")
        repl_row.addWidget(self.replace_edit)

        self.repl_one_btn = QPushButton("Заменить")
        self.repl_one_btn.setObjectName("apply_btn")
        self.repl_one_btn.clicked.connect(self.replace_one)
        repl_row.addWidget(self.repl_one_btn)

        self.repl_all_btn = QPushButton("Заменить всё")
        self.repl_all_btn.setObjectName("apply_btn")
        self.repl_all_btn.clicked.connect(self.replace_all)
        repl_row.addWidget(self.repl_all_btn)

        layout.addWidget(self.replace_widget)
        self.replace_widget.hide()

        QShortcut(QKeySequence("Escape"), self).activated.connect(self.hide)

    def toggle(self, replace_mode: bool = False):
        if self.isVisible() and self._replace_mode == replace_mode:
            self.hide()
            return
        self._replace_mode = replace_mode
        self.replace_widget.setVisible(replace_mode)
        self.show()
        self.find_edit.setFocus()
        self.find_edit.selectAll()

    def _params(self):
        return dict(regexp=self.regex_chk.isChecked(), cs=self.case_chk.isChecked(), 
                    wo=self.word_chk.isChecked(), wrap=True, forward=True)

    def _current_qsci(self):
        ed = self.editor_tabs.current_editor()
        if QSCI_AVAILABLE and isinstance(ed, QsciScintilla): return ed
        return None

    def _on_find_text_changed(self, text: str):
        self.result_lbl.setText("")
        if text: self.find_next(silent=True)

    def find_next(self, silent: bool = False):
        text = self.find_edit.text()
        if not text: return
        ed = self._current_qsci()
        if ed:
            p = self._params()
            found = ed.findFirst(text, p['regexp'], p['cs'], p['wo'], p['wrap'], forward=True)
            if not silent:
                self.result_lbl.setText("✓" if found else "✗ не найдено")
                self.result_lbl.setStyleSheet("color:#4EC9B0" if found else "color:#CE9178")
        else:
            ed_qt = self.editor_tabs.current_editor()
            if ed_qt:
                flags = QTextDocument.FindFlag(0)
                if self.case_chk.isChecked(): flags |= QTextDocument.FindFlag.FindCaseSensitively
                if self.word_chk.isChecked(): flags |= QTextDocument.FindFlag.FindWholeWords
                found = ed_qt.find(text, flags)
                if not found:
                    cursor = ed_qt.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.Start)
                    ed_qt.setTextCursor(cursor)
                    ed_qt.find(text, flags)

    def find_prev(self):
        text = self.find_edit.text()
        if not text: return
        ed = self._current_qsci()
        if ed:
            p = self._params()
            ed.findFirst(text, p['regexp'], p['cs'], p['wo'], p['wrap'], forward=False)

    def replace_one(self):
        text = self.find_edit.text()
        repl = self.replace_edit.text()
        if not text: return
        ed = self._current_qsci()
        if ed:
            p = self._params()
            if ed.hasSelectedText() and ed.selectedText() == text:
                ed.replace(repl)
            else:
                found = ed.findFirst(text, p['regexp'], p['cs'], p['wo'], p['wrap'], forward=True)
                if found: ed.replace(repl)
            self.find_next()

    def replace_all(self):
        text = self.find_edit.text()
        repl = self.replace_edit.text()
        if not text: return
        ed = self._current_qsci()
        if ed:
            p = self._params()
            ed.beginUndoAction()
            count = 0
            found = ed.findFirst(text, p['regexp'], p['cs'], p['wo'], False, forward=True)
            while found:
                ed.replace(repl)
                count += 1
                found = ed.findNext()
            ed.endUndoAction()
            self.result_lbl.setText(f"{count} замен")
            self.result_lbl.setStyleSheet("color:#4EC9B0")
        else:
            ed_qt = self.editor_tabs.current_editor()
            if ed_qt:
                old_text = ed_qt.toPlainText()
                flags = re.IGNORECASE if not self.case_chk.isChecked() else 0
                new_text, count = re.subn(re.escape(text), repl, old_text, flags=flags)
                if count:
                    cursor = ed_qt.textCursor()
                    cursor.beginEditBlock()
                    cursor.select(QTextCursor.SelectionType.Document)
                    cursor.insertText(new_text)
                    cursor.endEditBlock()
                self.result_lbl.setText(f"{count} замен")