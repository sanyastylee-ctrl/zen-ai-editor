from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtCore import Qt

class CheatSheetDialog(QDialog):
    SHORTCUTS = [
        ("Enter",          "Отправить сообщение"),
        ("Ctrl+Enter",     "Отправить сообщение (альтернатива)"),
        ("Ctrl+L",         "Очистить строку ввода"),
        ("Ctrl+1",         "Переключить в Режим: Кодер"),
        ("Ctrl+2",         "Переключить в Режим: Ассистент"),
        ("",               ""),
        ("Ctrl+S",         "Сохранить файл"),
        ("Ctrl+T",         "Новая вкладка редактора"),
        ("Ctrl+W",         "Закрыть вкладку"),
        ("Ctrl+R",         "Запустить текущий .py файл"),
        ("",               ""),
        ("Ctrl+F",         "Поиск в редакторе"),
        ("Ctrl+H",         "Поиск и замена"),
        ("Esc",            "Закрыть панель поиска"),
        ("Enter (поиск)",  "Следующее совпадение"),
        ("↑ / ↓ (поиск)", "Пред. / след. совпадение"),
        ("",               ""),
        ("Ctrl+Z",         "Отменить (в редакторе)"),
        ("Ctrl+Y",         "Повторить (в редакторе)"),
        ("",               ""),
        ("F1 / ?",         "Этот справочник"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Горячие клавиши — Zen AI Editor")
        self.setMinimumWidth(460)
        self.setStyleSheet("""
            QDialog { background:#1E1E1E; color:#D4D4D4; }
            QLabel  { color:#D4D4D4; font-size:13px; }
            QPushButton { background:#0E639C; color:#fff; border-radius:4px;
                          padding:6px 20px; font-weight:bold; border:none; }
            QPushButton:hover { background:#1177BB; }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("⌨ Горячие клавиши")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#4EC9B0; margin-bottom:8px;")
        layout.addWidget(title)

        for keys, desc in self.SHORTCUTS:
            if not keys and not desc:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("border:none; background:#333; margin:4px 0;")
                sep.setFixedHeight(1)
                layout.addWidget(sep)
                continue
            row = QHBoxLayout(); row.setSpacing(0)
            key_lbl = QLabel(keys)
            key_lbl.setFixedWidth(170)
            key_lbl.setStyleSheet(
                "font-family:Consolas; font-size:12px; color:#DCDCAA;"
                "background:#2D2D2D; border-radius:3px; padding:2px 6px;"
            )
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet("color:#CCCCCC; font-size:12px; padding-left:10px;")
            row.addWidget(key_lbl)
            row.addWidget(desc_lbl)
            row.addStretch()
            layout.addLayout(row)

        layout.addSpacing(10)
        ok = QPushButton("Закрыть")
        ok.clicked.connect(self.accept)
        layout.addWidget(ok, alignment=Qt.AlignmentFlag.AlignRight)