import difflib
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton

class DiffApplyDialog(QDialog):
    def __init__(self, old_code: str, new_code: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diff — предпросмотр изменений")
        self.resize(900, 600)
        self.setStyleSheet("""
            QDialog { background-color: #1E1E1E; color: #D4D4D4; }
            QTextEdit { background-color: #0D1117; color: #E6EDF3;
                        border: none; font-family: Consolas; font-size: 12px; }
            QPushButton { background-color: #238636; color: white;
                border-radius: 4px; padding: 6px 18px; font-weight: bold; border: none; }
            QPushButton:hover { background-color: #2EA043; }
            QPushButton#reject_btn { background-color: #3C3C3C; color: #D4D4D4;
                border: 1px solid #555555; }
            QPushButton#reject_btn:hover { background-color: #4A4A4A; }
        """)
        self.accepted_code = new_code
        layout = QVBoxLayout(self)
        lbl = QLabel("Зелёный — добавлено, красный — удалено. Принять изменения?")
        lbl.setStyleSheet("color:#888; font-size:12px; padding:4px;")
        layout.addWidget(lbl)
        self.diff_view = QTextEdit()
        self.diff_view.setReadOnly(True)
        layout.addWidget(self.diff_view)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        apply_btn = QPushButton("✔ Применить")
        apply_btn.clicked.connect(self.accept)
        reject_btn = QPushButton("✕ Отменить")
        reject_btn.setObjectName("reject_btn")
        reject_btn.clicked.connect(self.reject)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(reject_btn)
        layout.addLayout(btn_row)
        self._render_diff(old_code, new_code)

    def _render_diff(self, old: str, new: str):
        diff = list(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile="текущий", tofile="из чата", lineterm=""
        ))
        html_lines = []
        for line in diff:
            esc = (line.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'))
            if   line.startswith('+') and not line.startswith('+++'):
                html_lines.append(f"<span style='background:#0D2D0D;color:#3FB950;'>{esc}</span>")
            elif line.startswith('-') and not line.startswith('---'):
                html_lines.append(f"<span style='background:#2D0D0D;color:#F85149;'>{esc}</span>")
            elif line.startswith('@@'):
                html_lines.append(f"<span style='color:#58A6FF;'>{esc}</span>")
            else:
                html_lines.append(f"<span style='color:#8B949E;'>{esc}</span>")
        self.diff_view.setHtml("<pre style='margin:0;'>" + "".join(html_lines) + "</pre>")