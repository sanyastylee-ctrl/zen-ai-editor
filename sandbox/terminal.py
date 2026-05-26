"""
Встроенный терминал. Запускает команды в подпроцессе и стримит вывод.

Главные правки против старой версии:
- UTF-8 на Windows (через encoding="utf-8" и PYTHONIOENCODING) — больше нет крокозябр
- Тёмный фон без неоново-зелёного, шрифт JetBrains Mono / Consolas
- Команды юзера отделены от output блоками с цветным prompt'ом
- ANSI-цвета конвертируются в HTML
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel,
)

try:
    from ui.chat.styles import Palette, mono_font
except Exception:
    class Palette:
        BG_CHAT = "#1B1B23"
        BG_CODE = "#0F0F14"
        TEXT_PRIMARY = "#E6E6EC"
        TEXT_SECONDARY = "#9C9CAB"
        TEXT_DIM = "#6D6D7A"
        BORDER = "#2D2D38"
        ACCENT = "#A78BFA"
        ACCENT_GREEN = "#4DD49F"
        ACCENT_RED = "#F87171"
        ACCENT_BLUE = "#7DD3FC"

    def mono_font(size: int = 11) -> QFont:
        f = QFont("Consolas", size)
        f.setStyleHint(QFont.StyleHint.TypeWriter)
        return f


_ANSI_RE = re.compile(r"\x1B\[([0-9;]*)m")
_ANSI_FG = {
    "30": "#5C5C66", "31": Palette.ACCENT_RED,   "32": Palette.ACCENT_GREEN,
    "33": "#FBBF24", "34": Palette.ACCENT_BLUE,  "35": Palette.ACCENT,
    "36": "#7DD3FC", "37": Palette.TEXT_PRIMARY,
    "90": Palette.TEXT_DIM, "91": Palette.ACCENT_RED, "92": Palette.ACCENT_GREEN,
    "93": "#FBBF24", "94": Palette.ACCENT_BLUE, "95": Palette.ACCENT,
    "96": "#7DD3FC", "97": "#FFFFFF",
}


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("\n", "<br>"))


def ansi_to_html(text: str) -> str:
    out: list[str] = []
    open_span = False
    pos = 0
    for m in _ANSI_RE.finditer(text):
        chunk = text[pos:m.start()]
        if chunk:
            out.append(_html_escape(chunk))
        codes = m.group(1).split(";") if m.group(1) else ["0"]
        for c in codes:
            if c in ("0", ""):
                if open_span:
                    out.append("</span>")
                    open_span = False
            elif c in _ANSI_FG:
                if open_span:
                    out.append("</span>")
                out.append(f"<span style='color:{_ANSI_FG[c]};'>")
                open_span = True
            elif c == "1":
                if open_span:
                    out.append("</span>")
                out.append("<span style='font-weight:700;'>")
                open_span = True
        pos = m.end()
    if pos < len(text):
        out.append(_html_escape(text[pos:]))
    if open_span:
        out.append("</span>")
    return "".join(out)


class SandboxWorker(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, command: str, cwd: str) -> None:
        super().__init__()
        self.command = command
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._stop = False

    def stop(self) -> None:
        self._stop = True
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def run(self) -> None:
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            self._proc = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                env=env,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if self._stop:
                    break
                self.output_signal.emit(line)
            self._proc.wait()
            self.finished_signal.emit(self._proc.returncode)
        except Exception as e:
            self.output_signal.emit(f"[Ошибка запуска]: {e}\n")
            self.finished_signal.emit(-1)


class SandboxWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background: {Palette.BG_CHAT};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # === область вывода ===
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(mono_font(11))
        self.output.setStyleSheet(f"""
            QTextEdit {{
                background: {Palette.BG_CODE};
                color: {Palette.TEXT_PRIMARY};
                border: none;
                padding: 14px 18px;
                selection-background-color: {Palette.ACCENT}44;
            }}
            QScrollBar:vertical {{
                background: transparent; width: 10px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {Palette.BORDER}; border-radius: 5px; min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {Palette.TEXT_DIM}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0; background: transparent;
            }}
        """)
        self.output.setHtml(
            f"<div style='color:{Palette.TEXT_DIM}; font-size:11px; padding-bottom:8px;'>"
            f"Terminal — выполнение команд в папке проекта."
            f"</div>"
        )
        layout.addWidget(self.output)

        # === строка ввода ===
        input_bar = QWidget()
        input_bar.setStyleSheet(f"""
            background: {Palette.BG_CHAT};
            border-top: 1px solid {Palette.BORDER};
        """)
        cmd_row = QHBoxLayout(input_bar)
        cmd_row.setContentsMargins(12, 8, 12, 8)
        cmd_row.setSpacing(8)

        prompt_label = QLabel("$")
        prompt_label.setStyleSheet(
            f"color:{Palette.ACCENT}; font-family: Consolas, monospace;"
            f"font-size: 14px; font-weight: 700;"
        )
        cmd_row.addWidget(prompt_label)

        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText("команда... (Enter)")
        self.cmd_input.setFont(mono_font(11))
        self.cmd_input.setStyleSheet(f"""
            QLineEdit {{
                background: {Palette.BG_CODE};
                color: {Palette.TEXT_PRIMARY};
                border: 1px solid {Palette.BORDER};
                border-radius: 6px;
                padding: 6px 10px;
                selection-background-color: {Palette.ACCENT}44;
            }}
            QLineEdit:focus {{ border: 1px solid {Palette.ACCENT}; }}
        """)
        self.cmd_input.returnPressed.connect(self.run_command)
        cmd_row.addWidget(self.cmd_input)

        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setFixedWidth(76)
        self.run_btn.setStyleSheet(self._btn_style(primary=True))
        self.run_btn.clicked.connect(self.run_command)
        cmd_row.addWidget(self.run_btn)

        self.kill_btn = QPushButton("✕ Kill")
        self.kill_btn.setFixedWidth(70)
        self.kill_btn.setEnabled(False)
        self.kill_btn.setStyleSheet(self._btn_style(danger=True))
        self.kill_btn.clicked.connect(self.kill_process)
        cmd_row.addWidget(self.kill_btn)

        self.clear_btn = QPushButton("⎚")
        self.clear_btn.setFixedWidth(40)
        self.clear_btn.setToolTip("Очистить вывод")
        self.clear_btn.setStyleSheet(self._btn_style())
        self.clear_btn.clicked.connect(self.clear_output)
        cmd_row.addWidget(self.clear_btn)

        layout.addWidget(input_bar)

        self._worker: SandboxWorker | None = None
        self._cwd = os.getcwd()

    # ============================================================
    # API
    # ============================================================

    def set_cwd(self, path: str) -> None:
        self._cwd = path

    def run_command(self) -> None:
        cmd = self.cmd_input.text().strip()
        if not cmd:
            return

        cwd_short = self._short_cwd(self._cwd)
        block_html = (
            f"<div style='margin-top:14px; padding:6px 10px; "
            f"background:rgba(167,139,250,0.07); "
            f"border-left:3px solid {Palette.ACCENT}; border-radius:4px;'>"
            f"<span style='color:{Palette.TEXT_DIM}; font-size:10px;'>{_html_escape(cwd_short)}</span>"
            f"<br><span style='color:{Palette.ACCENT}; font-weight:700;'>$</span> "
            f"<span style='color:{Palette.TEXT_PRIMARY};'>{_html_escape(cmd)}</span>"
            f"</div>"
        )
        self._append_html(block_html)
        self.cmd_input.clear()

        if self._worker and self._worker.isRunning():
            self._worker.stop()

        self._worker = SandboxWorker(cmd, self._cwd)
        self._worker.output_signal.connect(self._on_output)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

        self.run_btn.setEnabled(False)
        self.kill_btn.setEnabled(True)

    def kill_process(self) -> None:
        if self._worker:
            self._worker.stop()

    def clear_output(self) -> None:
        self.output.clear()

    def run_code_file(self, file_path: str) -> None:
        cmd = f'"{sys.executable}" "{file_path}"'
        self.cmd_input.setText(cmd)
        self.run_command()

    # ============================================================
    # стрим
    # ============================================================

    def _on_output(self, text: str) -> None:
        html = ansi_to_html(text)
        wrapped = (
            f"<div style='padding: 0 4px; color:{Palette.TEXT_PRIMARY};'>"
            f"{html}</div>"
        )
        self._append_html(wrapped)

    def _on_finished(self, code: int) -> None:
        color = Palette.ACCENT_GREEN if code == 0 else Palette.ACCENT_RED
        symbol = "✓" if code == 0 else "✕"
        self._append_html(
            f"<div style='margin-top:4px; color:{color}; font-size:11px;'>"
            f"{symbol} exit {code}</div>"
        )
        self.run_btn.setEnabled(True)
        self.kill_btn.setEnabled(False)

    # ============================================================
    # утилиты
    # ============================================================

    def _append_html(self, html: str) -> None:
        cur = self.output.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        self.output.setTextCursor(cur)
        self.output.insertHtml(html)
        sb = self.output.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _short_cwd(self, path: str) -> str:
        home = os.path.expanduser("~")
        if path.startswith(home):
            path = "~" + path[len(home):]
        if len(path) > 60:
            path = "…" + path[-58:]
        return path

    @staticmethod
    def _btn_style(primary: bool = False, danger: bool = False) -> str:
        if primary:
            return f"""
                QPushButton {{
                    background: {Palette.ACCENT}; color: white;
                    border: none; border-radius: 6px;
                    padding: 6px 10px; font-weight: 600; font-size: 12px;
                }}
                QPushButton:hover {{ background: #B8A3FF; }}
                QPushButton:disabled {{ background: {Palette.BORDER}; color: {Palette.TEXT_DIM}; }}
            """
        if danger:
            return f"""
                QPushButton {{
                    background: rgba(248,113,113,0.15); color: {Palette.ACCENT_RED};
                    border: 1px solid {Palette.ACCENT_RED}55; border-radius: 6px;
                    padding: 6px 10px; font-weight: 600; font-size: 12px;
                }}
                QPushButton:hover:enabled {{ background: rgba(248,113,113,0.25); }}
                QPushButton:disabled {{ background: transparent;
                    color: {Palette.TEXT_DIM}; border-color: {Palette.BORDER}; }}
            """
        return f"""
            QPushButton {{
                background: transparent; color: {Palette.TEXT_SECONDARY};
                border: 1px solid {Palette.BORDER}; border-radius: 6px;
                padding: 6px 10px; font-size: 13px;
            }}
            QPushButton:hover {{ color: {Palette.TEXT_PRIMARY};
                                  background: rgba(255,255,255,0.04); }}
        """
