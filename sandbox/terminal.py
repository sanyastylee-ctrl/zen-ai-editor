"""
Встроенный терминал. Запускает команды в подпроцессе и стримит вывод.

Логика взята из исходного zen_editor.py, разнесена в отдельный модуль.
"""

from __future__ import annotations

import os
import subprocess
import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
)


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
            self._proc = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                text=True,
                bufsize=1,
            )
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 10))
        self.output.setStyleSheet(
            "background-color:#0D0D0D; color:#00FF88; border:none; padding:6px;"
        )
        layout.addWidget(self.output)

        cmd_row = QHBoxLayout()
        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText("$ команда... (Enter)")
        self.cmd_input.setFont(QFont("Consolas", 11))
        self.cmd_input.returnPressed.connect(self.run_command)
        cmd_row.addWidget(self.cmd_input)

        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setFixedWidth(70)
        self.run_btn.clicked.connect(self.run_command)
        cmd_row.addWidget(self.run_btn)

        self.kill_btn = QPushButton("✕ Kill")
        self.kill_btn.setFixedWidth(70)
        self.kill_btn.setObjectName("stop_btn")
        self.kill_btn.setEnabled(False)
        self.kill_btn.clicked.connect(self.kill_process)
        cmd_row.addWidget(self.kill_btn)

        self.clear_btn = QPushButton("🗑")
        self.clear_btn.setFixedWidth(40)
        self.clear_btn.setObjectName("secondary")
        self.clear_btn.clicked.connect(self.output.clear)
        cmd_row.addWidget(self.clear_btn)

        layout.addLayout(cmd_row)

        self._worker: SandboxWorker | None = None
        self._cwd = os.getcwd()

    def set_cwd(self, path: str) -> None:
        self._cwd = path

    def run_command(self) -> None:
        cmd = self.cmd_input.text().strip()
        if not cmd:
            return
        self.output.append(f"<span style='color:#569CD6;'>$ {cmd}</span>")
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

    def _on_output(self, text: str) -> None:
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(text)
        sb = self.output.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, code: int) -> None:
        color = "#888888" if code == 0 else "#CE9178"
        self.output.append(f"<span style='color:{color};'>[exit {code}]</span><br>")
        self.run_btn.setEnabled(True)
        self.kill_btn.setEnabled(False)

    def run_code_file(self, file_path: str) -> None:
        cmd = f'{sys.executable} "{file_path}"'
        self.cmd_input.setText(cmd)
        self.run_command()
