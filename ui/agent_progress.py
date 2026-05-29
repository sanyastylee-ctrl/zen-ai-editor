from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class AgentProgressOverlay(QFrame):
    """Compact Coder progress panel fed by AgentRunState/TaskLedger snapshots."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("agent_progress")
        self.setVisible(False)
        self._collapsed = False

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.title_label = QLabel("Кодер работает")
        self.title_label.setObjectName("agentProgressTitle")
        head.addWidget(self.title_label)

        self.phase_label = QLabel("")
        self.phase_label.setObjectName("agentProgressMeta")
        head.addWidget(self.phase_label, 1)

        self.details_btn = QPushButton("Details")
        self.details_btn.setObjectName("secondaryCompact")
        self.details_btn.clicked.connect(self._toggle_details)
        head.addWidget(self.details_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("agentProgressStop")
        head.addWidget(self.stop_btn)
        root.addLayout(head)

        self.step_label = QLabel("")
        self.step_label.setObjectName("agentProgressStep")
        self.step_label.setWordWrap(True)
        root.addWidget(self.step_label)

        self.auto_label = QLabel("")
        self.auto_label.setObjectName("agentProgressMeta")
        self.auto_label.setVisible(False)
        root.addWidget(self.auto_label)

        self.details_label = QLabel("")
        self.details_label.setObjectName("agentProgressDetails")
        self.details_label.setWordWrap(True)
        self.details_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self.details_label)

    def update_state(self, snapshot: dict) -> None:
        self.setVisible(True)
        done = int(snapshot.get("done_steps") or 0)
        total = int(snapshot.get("total_steps") or 0)
        phase = str(snapshot.get("current_phase") or "")
        event = str(snapshot.get("event") or "state")
        if event == "blocked":
            self.setProperty("state", "blocked")
            self.title_label.setText(f"Требуется внимание: {done}/{total}")
        elif event == "finished":
            self.setProperty("state", "finished")
            self.title_label.setText(f"Готово: {done}/{total}")
        else:
            self.setProperty("state", "running")
            self.title_label.setText(f"Кодер работает: {done}/{total}" if total else "Кодер работает")
        self.style().unpolish(self)
        self.style().polish(self)

        self.phase_label.setText(phase)
        current = str(snapshot.get("current_step") or "")
        tool = str(snapshot.get("current_tool") or "")
        command = str(snapshot.get("current_command") or "")
        if command:
            self.step_label.setText(f"Текущий шаг: {current}\nПоследняя команда: {command}")
        elif tool:
            self.step_label.setText(f"Текущий шаг: {current}\nПоследний tool: {tool}")
        else:
            self.step_label.setText(f"Текущий шаг: {current}" if current else "")

        count = int(snapshot.get("auto_continue_count") or 0)
        maximum = int(snapshot.get("max_auto_continues") or 0)
        reason = str(snapshot.get("auto_continue_reason") or "")
        if count or reason:
            text = f"Автопродолжение: {count}/{maximum}"
            if reason:
                text += f" · {reason}"
            self.auto_label.setText(text)
            self.auto_label.setVisible(True)
        else:
            self.auto_label.setVisible(False)

        blocker = str(snapshot.get("blocker_reason") or "")
        if blocker:
            self.step_label.setText(f"{self.step_label.text()}\nПричина: {blocker}".strip())

        self.details_label.setText(self._format_details(snapshot))
        self.details_label.setVisible(not self._collapsed)

    def set_finished(self, snapshot: dict) -> None:
        snapshot = dict(snapshot)
        snapshot["event"] = "finished"
        self.update_state(snapshot)

    def set_blocked(self, snapshot: dict) -> None:
        snapshot = dict(snapshot)
        snapshot["event"] = "blocked"
        self.update_state(snapshot)

    def reset(self) -> None:
        self.setVisible(False)
        self.title_label.setText("Кодер работает")
        self.phase_label.setText("")
        self.step_label.setText("")
        self.auto_label.setText("")
        self.details_label.setText("")

    def _toggle_details(self) -> None:
        self._collapsed = not self._collapsed
        self.details_label.setVisible(not self._collapsed)
        self.details_btn.setText("Show" if self._collapsed else "Details")

    @staticmethod
    def _format_details(snapshot: dict) -> str:
        lines: list[str] = []
        plan = str(snapshot.get("plan") or "").strip()
        if plan:
            lines.append("План: " + plan[:240])
        for item in snapshot.get("ledger") or []:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "todo")
            mark = {
                "done": "✓",
                "doing": "⏳",
                "failed": "×",
                "blocked": "!",
                "skipped": "-",
            }.get(status, "○")
            desc = str(item.get("description") or item.get("command") or item.get("target_file") or item.get("id") or "")
            if desc:
                lines.append(f"{mark} {desc}")
        goals = snapshot.get("command_goals") or []
        done_goals = set(snapshot.get("command_goals_done") or [])
        if goals:
            lines.append("Команды:")
            for goal in goals:
                mark = "✓" if goal in done_goals else "○"
                lines.append(f"{mark} {goal}")
        return "\n".join(lines[-18:])
