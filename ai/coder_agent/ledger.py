from __future__ import annotations

from dataclasses import asdict

from core.tools import ToolCall, ToolResult

from .state import TaskLedgerItem, TaskStatus, TaskType
from .verification import CommandGoal


class TaskLedger:
    """Small source-of-truth ledger for Coder Agent v3 progress.

    The existing AgentWorker still performs model calls and tools. This ledger
    tracks whether the controller is allowed to finalize.
    """

    def __init__(self, items: list[TaskLedgerItem] | None = None) -> None:
        self.items: list[TaskLedgerItem] = items or []

    @classmethod
    def from_command_goals(cls, goals: list[CommandGoal]) -> "TaskLedger":
        items = [
            TaskLedgerItem(
                id=f"cmd-{i + 1}",
                description=f"Run verification command: {goal.example()}",
                type=TaskType.RUN_COMMAND,
                required_tool="run_terminal",
                command=goal.raw,
            )
            for i, goal in enumerate(goals)
        ]
        return cls(items)

    def ensure_file_item(self, path: str, task_type: TaskType, required_tool: str) -> None:
        if not path:
            return
        key = (path, task_type.value)
        for item in self.items:
            if (item.target_file, item.type.value) == key:
                return
        item = TaskLedgerItem(
            id=f"file-{len(self.items) + 1}",
            description=f"{task_type.value}: {path}",
            type=task_type,
            required_tool=required_tool,
            target_file=path,
        )
        command_index = next(
            (index for index, existing in enumerate(self.items) if existing.type == TaskType.RUN_COMMAND),
            len(self.items),
        )
        self.items.insert(command_index, item)

    def ensure_repair_item(self, repair_id: str, description: str, evidence: str = "") -> None:
        if not repair_id:
            return
        for item in self.items:
            if item.id == repair_id:
                item.status = TaskStatus.TODO
                if evidence:
                    item.evidence.append(evidence)
                return
        self.items.append(
            TaskLedgerItem(
                id=repair_id,
                description=description,
                type=TaskType.FIX,
                required_tool="edit_file/apply_patch/write_file",
                evidence=[evidence] if evidence else [],
            )
        )

    def mark_repair_done(self, repair_id: str, evidence: str = "") -> None:
        for item in self.items:
            if item.id == repair_id:
                item.status = TaskStatus.DONE
                if evidence:
                    item.evidence.append(evidence)

    def record_tool_result(self, call: ToolCall, result: ToolResult, matched_command_goal: str = "") -> None:
        if not result.ok or result.meta.get("duplicate"):
            return
        name = call.name
        if name in {"write_file", "create_file"}:
            path = str(result.meta.get("path") or call.args.get("path") or "")
            self.ensure_file_item(path, TaskType.CREATE_FILE, "write_file")
            self._mark_file_done(path, TaskType.CREATE_FILE, f"{name} ok")
        elif name in {"edit_file", "apply_patch", "patch_file"}:
            path = str(result.meta.get("path") or call.args.get("path") or "")
            self.ensure_file_item(path, TaskType.EDIT_FILE, name)
            self._mark_file_done(path, TaskType.EDIT_FILE, f"{name} ok")
        elif name in {"run_terminal", "run_command"} and matched_command_goal:
            self._mark_command_done(matched_command_goal, f"{name} ok")

    def mark_file_verified(self, path: str, evidence: str = "") -> None:
        for item in self.items:
            if item.target_file == path and item.type in {TaskType.CREATE_FILE, TaskType.EDIT_FILE, TaskType.PATCH}:
                item.status = TaskStatus.DONE
                if evidence:
                    item.evidence.append(evidence)

    def mark_file_failed(self, path: str, reason: str) -> None:
        for item in self.items:
            if item.target_file == path and item.type in {TaskType.CREATE_FILE, TaskType.EDIT_FILE, TaskType.PATCH}:
                item.status = TaskStatus.FAILED
                item.failure_reason = reason
                if reason:
                    item.evidence.append(reason)

    def _mark_file_done(self, path: str, task_type: TaskType, evidence: str) -> None:
        for item in self.items:
            if item.target_file == path and item.type == task_type:
                item.status = TaskStatus.DONE
                item.evidence.append(evidence)

    def _mark_command_done(self, command: str, evidence: str) -> None:
        for item in self.items:
            if item.type == TaskType.RUN_COMMAND and item.command == command:
                item.status = TaskStatus.DONE
                item.evidence.append(evidence)

    def pending(self) -> list[TaskLedgerItem]:
        return [
            item for item in self.items
            if item.status in {TaskStatus.TODO, TaskStatus.DOING, TaskStatus.FAILED, TaskStatus.BLOCKED}
        ]

    def current_item(self) -> TaskLedgerItem | None:
        pending = self.pending()
        return pending[0] if pending else None

    def mark_doing(self, item_id: str, evidence: str = "") -> None:
        for item in self.items:
            if item.id == item_id and item.status == TaskStatus.TODO:
                item.status = TaskStatus.DOING
                if evidence:
                    item.evidence.append(evidence)
                return

    def mark_failed(self, item_id: str, reason: str) -> None:
        for item in self.items:
            if item.id == item_id:
                item.status = TaskStatus.FAILED
                item.failure_reason = reason
                if reason:
                    item.evidence.append(reason)
                return

    def final_allowed(self) -> bool:
        return not self.pending()

    def next_step_text(self) -> str:
        pending = self.pending()
        if not pending:
            return "All ledger items are done; finalize with summary."
        item = pending[0]
        if item.command:
            return f"Run pending command: {item.command}"
        if item.target_file:
            return f"Finish {item.type.value}: {item.target_file}"
        return item.description

    def to_summary(self) -> list[dict]:
        return [asdict(item) for item in self.items]
