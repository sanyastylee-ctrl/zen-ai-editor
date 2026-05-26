import sys
import os
import shlex
import subprocess
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QLabel)
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtCore import Qt, QThread, pyqtSignal

class SandboxWorker(QThread):
    output_signal   = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, command: str, cwd: str):
        super().__init__()
        self.command = command
        self.cwd     = cwd
        self._proc   = None
        self._stop   = False

    def stop(self):
        self._stop = True
        if self._proc:
            try: self._proc.terminate()
            except Exception: pass

    def write_stdin(self, text: str):
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(text + "\n")
                self._proc.stdin.flush()
            except Exception as e:
                self.output_signal.emit(f"[stdin error: {e}]\n")

    def is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def run(self):
        try:
            if sys.platform == 'win32':
                argv = ['cmd.exe', '/c', self.command]
            else:
                argv = ['/bin/sh', '-c', self.command]

            self._proc = subprocess.Popen(
                argv,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                cwd=self.cwd, text=True, bufsize=1
            )
            for line in self._proc.stdout:
                if self._stop: break
                self.output_signal.emit(line)
            self._proc.wait()
            self.finished_signal.emit(self._proc.returncode)
        except Exception as e:
            self.output_signal.emit(f"[Ошибка запуска]: {str(e)}\n")
            self.finished_signal.emit(-1)


class SandboxWidget(QWidget):
    def __init__(self, parent=None):
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
        self.cmd_input.returnPressed.connect(self.run_or_send)
        cmd_row.addWidget(self.cmd_input)

        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setFixedWidth(70)
        self.run_btn.clicked.connect(self.run_or_send)
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
        self._worker  = None
        self._cwd     = os.getcwd()
        
        self._stdin_label = QLabel()
        self._stdin_label.setStyleSheet("color:#888; font-size:11px;")
        layout.addWidget(self._stdin_label)

    def set_cwd(self, path: str):
        self._cwd = path

    def _process_active(self) -> bool:
        return self._worker is not None and self._worker.is_running()

    def run_or_send(self):
        text = self.cmd_input.text().strip()
        if not text: return

        if self._process_active():
            self.output.append(f"<span style='color:#DCDCAA;'>stdin&gt; {text}</span>")
            self.cmd_input.clear()
            self._worker.write_stdin(text)
        else:
            self.run_command(text)

    def run_command(self, cmd: str | None = None):
        if cmd is None:
            cmd = self.cmd_input.text().strip()
        if not cmd: return

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
        self.cmd_input.setPlaceholderText("stdin> ввод для процесса... (Enter)")
        self._stdin_label.setText("⚡ Процесс запущен — Enter отправляет stdin")

    def kill_process(self):
        if self._worker: self._worker.stop()

    def _on_output(self, text):
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(text)
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())

    def _on_finished(self, code):
        color = "#888888" if code == 0 else "#CE9178"
        self.output.append(f"<span style='color:{color};'>[exit {code}]</span><br>")
        self.run_btn.setEnabled(True)
        self.kill_btn.setEnabled(False)
        self.cmd_input.setPlaceholderText("$ команда... (Enter)")
        self._stdin_label.setText("")

    def run_code_file(self, file_path: str):
        abs_path = os.path.abspath(file_path)
        self.run_command(f"{shlex.quote(sys.executable)} {shlex.quote(abs_path)}")