import os
from PyQt6.QtWidgets import QTabWidget, QPushButton, QMessageBox, QTextEdit
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtCore import pyqtSignal, Qt

from widgets.editor import make_scintilla_editor, set_lexer_for_file, PythonHighlighter, QSCI_AVAILABLE

try:
    from PyQt6.Qsci import QsciScintilla
except ImportError:
    pass

class EditorTabWidget(QTabWidget):
    """
    QTabWidget где каждая вкладка содержит независимый редактор кода.
    Хранит: editor (QsciScintilla или QTextEdit), file_path, modified.
    """
    context_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setDocumentMode(True)
        self._tabs: list[dict] = []  

        new_tab_btn = QPushButton("+")
        new_tab_btn.setFixedSize(26, 26)
        new_tab_btn.setStyleSheet(
            "QPushButton{background:#2D2D2D;color:#888;border:none;border-radius:4px;font-size:16px;}"
            "QPushButton:hover{background:#3C3C3C;color:#FFF;}"
        )
        new_tab_btn.setToolTip("Новая вкладка (Ctrl+T)")
        new_tab_btn.clicked.connect(lambda: self.new_tab())
        self.setCornerWidget(new_tab_btn, Qt.Corner.TopRightCorner)

        self.tabCloseRequested.connect(self._close_tab)
        self.currentChanged.connect(self.context_changed)
        self.new_tab()

    def _make_editor(self):
        if QSCI_AVAILABLE:
            ed = make_scintilla_editor()
            set_lexer_for_file(ed, ".py")
            ed.textChanged.connect(lambda: self._mark_modified(ed))
            return ed, None
        else:
            ed = QTextEdit()
            ed.setFont(QFont("Consolas", 11))
            hl = PythonHighlighter(ed.document())
            ed.textChanged.connect(lambda: self._mark_modified(ed))
            return ed, hl

    def _mark_modified(self, editor):
        for i, td in enumerate(self._tabs):
            if td["editor"] is editor:
                if not td["modified"]:
                    td["modified"] = True
                    title = self.tabText(i)
                    if not title.endswith(" *"):
                        self.setTabText(i, title + " *")
                break
        self.context_changed.emit()

    def new_tab(self, file_path: str = "", content: str = "") -> int:
        ed, hl = self._make_editor()
        td = {"editor": ed, "file_path": file_path, "modified": False, "highlighter": hl}
        self._tabs.append(td)
        title = os.path.basename(file_path) if file_path else "Новый файл"
        idx = self.addTab(ed, title)
        self.setCurrentIndex(idx)
        if content:
            self._set_text_raw(ed, content)
            td["modified"] = False
            self.setTabText(idx, title) 
        if file_path and QSCI_AVAILABLE and isinstance(ed, QsciScintilla):
            set_lexer_for_file(ed, file_path)
        return idx

    def open_file_tab(self, file_path: str, content: str):
        existing = self._find_tab_by_path(file_path)
        if existing >= 0:
            self.setCurrentIndex(existing)
            return

        cur = self.currentIndex()
        td  = self._tabs[cur] if 0 <= cur < len(self._tabs) else None
        if td and not td["file_path"] and not td["modified"]:
            self._set_text_raw(td["editor"], content)
            td["file_path"] = file_path
            td["modified"]  = False
            title = os.path.basename(file_path)
            self.setTabText(cur, title)
            if QSCI_AVAILABLE and isinstance(td["editor"], QsciScintilla):
                set_lexer_for_file(td["editor"], file_path)
        else:
            self.new_tab(file_path, content)

    def save_current(self) -> tuple[bool, str]:
        td = self._current_td()
        if not td: return False, ""
        if not td["file_path"]:
            return False, ""  
        try:
            with open(td["file_path"], 'w', encoding='utf-8') as f:
                f.write(self.get_text())
            td["modified"] = False
            self.setTabText(self.currentIndex(), os.path.basename(td["file_path"]))
            return True, td["file_path"]
        except Exception as e:
            return False, str(e)

    def set_current_path(self, path: str):
        td = self._current_td()
        if td:
            td["file_path"] = path
            td["modified"]  = False
            self.setTabText(self.currentIndex(), os.path.basename(path))
            if QSCI_AVAILABLE and isinstance(td["editor"], QsciScintilla):
                set_lexer_for_file(td["editor"], path)

    def current_file_path(self) -> str:
        td = self._current_td()
        return td["file_path"] if td else ""

    def current_editor(self):
        td = self._current_td()
        return td["editor"] if td else None

    def get_text(self) -> str:
        ed = self.current_editor()
        if ed is None: return ""
        if QSCI_AVAILABLE and isinstance(ed, QsciScintilla): return ed.text()
        return ed.toPlainText()

    def set_text(self, text: str, undo_safe: bool = False):
        ed = self.current_editor()
        if ed is None: return
        if undo_safe and QSCI_AVAILABLE and isinstance(ed, QsciScintilla):
            ed.beginUndoAction()
            ed.selectAll()
            ed.replaceSelectedText(text)
            ed.endUndoAction()
        elif undo_safe and not (QSCI_AVAILABLE and isinstance(ed, QsciScintilla)):
            cursor = ed.textCursor()
            cursor.beginEditBlock()
            cursor.select(QTextCursor.SelectionType.Document)
            cursor.insertText(text)
            cursor.endEditBlock()
        else:
            self._set_text_raw(ed, text)
        td = self._current_td()
        if td:
            td["modified"] = False
            title = os.path.basename(td["file_path"]) if td["file_path"] else "Новый файл"
            self.setTabText(self.currentIndex(), title)

    def replace_selection(self, text: str):
        ed = self.current_editor()
        if ed is None: return
        if QSCI_AVAILABLE and isinstance(ed, QsciScintilla):
            ed.replaceSelectedText(text)
        else:
            ed.textCursor().insertText(text)

    def has_selection(self) -> bool:
        ed = self.current_editor()
        if ed is None: return False
        if QSCI_AVAILABLE and isinstance(ed, QsciScintilla): return ed.hasSelectedText()
        return ed.textCursor().hasSelection()

    def selected_text(self) -> str:
        ed = self.current_editor()
        if ed is None: return ""
        if QSCI_AVAILABLE and isinstance(ed, QsciScintilla): return ed.selectedText()
        return ed.textCursor().selectedText()

    def _set_text_raw(self, ed, text: str):
        if QSCI_AVAILABLE and isinstance(ed, QsciScintilla): ed.setText(text)
        else: ed.setPlainText(text)

    def _current_td(self) -> dict | None:
        idx = self.currentIndex()
        if 0 <= idx < len(self._tabs): return self._tabs[idx]
        return None

    def _find_tab_by_path(self, path: str) -> int:
        for i, td in enumerate(self._tabs):
            if td["file_path"] == path: return i
        return -1

    def _close_tab(self, idx: int):
        if idx < 0 or idx >= len(self._tabs): return
        td = self._tabs[idx]
        if td["modified"] and td["file_path"]:
            res = QMessageBox.question(
                self, "Сохранить?",
                f"Файл «{os.path.basename(td['file_path'])}» изменён. Сохранить?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            if res == QMessageBox.StandardButton.Cancel: return
            if res == QMessageBox.StandardButton.Save:
                with open(td["file_path"], 'w', encoding='utf-8') as f:
                    f.write(self._get_text_for_editor(td["editor"]))

        if len(self._tabs) == 1:
            td["file_path"] = ""
            td["modified"]  = False
            self._set_text_raw(td["editor"], "")
            self.setTabText(0, "Новый файл")
            return

        self._tabs.pop(idx)
        self.removeTab(idx)

    def _get_text_for_editor(self, ed) -> str:
        if QSCI_AVAILABLE and isinstance(ed, QsciScintilla): return ed.text()
        return ed.toPlainText()