from __future__ import annotations

from core.tools import ToolCall, ToolResult

from .evaluator import evaluate_final_readiness
from .guards import GuardDecision
from .ledger import TaskLedger
from .state import AgentPhase, AgentRunStateV3, TaskLedgerItem, TaskStatus, TaskType
from .verification import normalize_command


READ_TOOLS = {"read_file", "search_files", "list_files", "project_map", "git_status", "git_diff"}
FILE_MUTATION_TOOLS = {"write_file", "create_file", "edit_file", "apply_patch", "patch_file"}
COMMAND_TOOLS = {"run_terminal", "run_command", "test_runner"}
VERIFY_TOOLS = {"read_file", "run_terminal", "run_command", "test_runner", "git_diff"}


PHASE_ALLOWED_TOOLS: dict[AgentPhase, set[str]] = {
    AgentPhase.IDLE: set(),
    AgentPhase.CLASSIFY_INTENT: set(),
    AgentPhase.RESOLVE_TASK: set(),
    AgentPhase.PLAN: set(),
    AgentPhase.BUILD_PLAN: set(),
    AgentPhase.BUILD_LEDGER: set(),
    AgentPhase.PROJECT_MAP: {"list_files", "project_map", "search_files", "git_status"},
    AgentPhase.INSPECT: READ_TOOLS,
    AgentPhase.EXECUTE: READ_TOOLS | FILE_MUTATION_TOOLS | COMMAND_TOOLS,
    AgentPhase.EXECUTE_TOOL: READ_TOOLS | FILE_MUTATION_TOOLS | COMMAND_TOOLS,
    AgentPhase.VERIFY: VERIFY_TOOLS,
    AgentPhase.REPAIR: READ_TOOLS | {"edit_file", "apply_patch", "patch_file", "write_file", "run_terminal", "run_command"},
    AgentPhase.EVALUATE: set(),
    AgentPhase.FINALIZE: set(),
    AgentPhase.CHECKPOINT: set(),
    AgentPhase.AUTO_CONTINUE: set(),
    AgentPhase.BLOCKED: set(),
    AgentPhase.DONE: set(),
}


class CoderAgentController:
    """Strict state controller used by AgentWorker.

    It intentionally does not know about Qt, llama.cpp, or UI widgets. The
    worker owns integration; this controller owns phase and completion policy.
    """

    def __init__(self, state: AgentRunStateV3, ledger: TaskLedger) -> None:
        self.state = state
        self.ledger = ledger

    def set_phase(self, phase: str | AgentPhase) -> None:
        if isinstance(phase, AgentPhase):
            value = phase
        else:
            aliases = {
                "execute_tools": AgentPhase.EXECUTE,
                "inspect_project": AgentPhase.INSPECT,
                "plan": AgentPhase.PLAN,
                "build_plan": AgentPhase.BUILD_PLAN,
                "execute_tool": AgentPhase.EXECUTE_TOOL,
            }
            phase_text = str(phase)
            value = aliases[phase_text] if phase_text in aliases else AgentPhase(phase_text)
        self.state.current_phase = value

    def validate_tool_call(self, call: ToolCall, facts: dict | None = None) -> GuardDecision:
        """Validate a model-requested tool against phase, ledger, and evidence.

        The worker still executes the tool, but only after the controller says
        this action is legal for the current state. This is the main boundary
        that keeps the model from acting as the process dispatcher.
        """
        facts = facts or {}
        tool = call.name
        phase = self.state.current_phase
        allowed = PHASE_ALLOWED_TOOLS.get(phase, set())
        if tool not in allowed:
            return GuardDecision(
                True,
                reason=f"tool {tool} is not allowed in phase {phase.value}",
                corrective=(
                    f"The current controller phase is {phase.value}. "
                    f"Allowed tools: {', '.join(sorted(allowed)) or 'none'}. "
                    "Do not skip phases; emit the next allowed tool call only."
                ),
            )

        duplicate = self._duplicate_done_command(call, facts)
        if duplicate.blocked:
            return duplicate

        if tool in COMMAND_TOOLS:
            blocked = self._command_blocker(call, facts)
            if blocked.blocked:
                return blocked

        item = self.ledger.current_item()
        if item is not None:
            blocked = self._ledger_item_blocker(call, item, facts)
            if blocked.blocked:
                return blocked
            if item.status == TaskStatus.TODO:
                self.ledger.mark_doing(item.id, f"controller allowed {tool}")
        return GuardDecision(False)

    def validate_final(self) -> GuardDecision:
        return self.final_guard()

    def record_tool_result(self, call: ToolCall, result: ToolResult, matched_command_goal: str = "") -> None:
        self.state.tool_history.append(
            {
                "name": call.name,
                "args": dict(call.args),
                "ok": result.ok,
                "meta": dict(result.meta),
            }
        )
        if call.name in {"run_terminal", "run_command"}:
            command = str(call.args.get("command", ""))
            self.state.command_history.append(
                {
                    "command": command,
                    "ok": result.ok,
                    "matched_goal": matched_command_goal,
                }
            )
            if not result.ok:
                self.state.failed_commands.append(command)
        self.ledger.record_tool_result(call, result, matched_command_goal=matched_command_goal)
        self.state.final_allowed = self.ledger.final_allowed()

    def final_guard(self) -> GuardDecision:
        evaluation = evaluate_final_readiness(self.state, self.ledger)
        self.state.final_allowed = evaluation.allowed
        if evaluation.allowed:
            self.state.blocked_reason = ""
            return GuardDecision(False)
        self.state.blocked_reason = evaluation.reason
        return GuardDecision(
            True,
            reason=evaluation.reason,
            corrective=(
                "You cannot give a final answer yet. The deterministic evaluator "
                f"found pending work: {evaluation.summary}. "
                f"Next required step: {self.ledger.next_step_text()}. Emit the needed XML tool call now."
            ),
        )

    def checkpoint_summary(self) -> dict:
        return {
            "current_phase": self.state.current_phase.value,
            "task_ledger": self.ledger.to_summary(),
            "final_allowed": self.state.final_allowed,
            "blocked_reason": self.state.blocked_reason,
        }

    def _duplicate_done_command(self, call: ToolCall, facts: dict) -> GuardDecision:
        if call.name not in COMMAND_TOOLS:
            return GuardDecision(False)
        command = str(call.args.get("command") or "")
        done = set(facts.get("command_goals_done") or self.state.command_goals_done or [])
        for goal in done:
            if normalize_command(command) == normalize_command(goal):
                return GuardDecision(
                    True,
                    reason=f"command goal already done: {goal}",
                    corrective=(
                        "This exact command goal is already verified. "
                        "Do not rerun completed goals; move to the first unfinished ledger item."
                    ),
                )
        return GuardDecision(False)

    def _command_blocker(self, call: ToolCall, facts: dict) -> GuardDecision:
        command = str(call.args.get("command") or "")
        if self._is_lightweight_verification_command(command):
            return GuardDecision(False)
        active_repair = facts.get("active_repair") if isinstance(facts.get("active_repair"), dict) else {}
        if active_repair and not bool(active_repair.get("touched_relevant")):
            failed = str(active_repair.get("failed_command") or "")
            if not failed or normalize_command(command) == normalize_command(failed):
                return GuardDecision(
                    True,
                    reason="repair requires relevant patch before rerun",
                    corrective=(
                        "A command already failed. Inspect and patch the traceback-relevant files "
                        "before rerunning the failed command."
                    ),
                )

        repair_touched = bool(active_repair.get("touched_relevant") or facts.get("repair_touched_relevant"))
        pending_before_command = [
            item for item in self.ledger.pending()
            if item.type in {TaskType.CREATE_FILE, TaskType.EDIT_FILE, TaskType.PATCH}
            or (item.type == TaskType.FIX and not repair_touched)
        ]
        if pending_before_command:
            return GuardDecision(
                True,
                reason="run_terminal blocked while required file/repair ledger items are pending",
                corrective=(
                    "Do not run verification commands yet. Finish the pending create/edit/repair "
                    f"item first: {pending_before_command[0].description}."
                ),
            )
        return GuardDecision(False)

    def _ledger_item_blocker(self, call: ToolCall, item: TaskLedgerItem, facts: dict) -> GuardDecision:
        tool = call.name
        if item.type == TaskType.RUN_COMMAND:
            if tool in READ_TOOLS | FILE_MUTATION_TOOLS:
                return GuardDecision(False)
            if tool not in COMMAND_TOOLS:
                return GuardDecision(
                    True,
                    reason=f"current ledger item requires command, got {tool}",
                    corrective=f"Current ledger item requires running: {item.command}. Emit run_terminal.",
                )
            command = str(call.args.get("command") or "")
            if self._is_lightweight_verification_command(command):
                return GuardDecision(False)
            if item.command and normalize_command(command) != normalize_command(item.command):
                return GuardDecision(
                    True,
                    reason=f"out-of-order command goal: expected {item.command}, got {command}",
                    corrective=f"Run the current pending command exactly: {item.command}",
                )
            return GuardDecision(False)

        if item.type in {TaskType.CREATE_FILE, TaskType.EDIT_FILE, TaskType.PATCH}:
            if tool in READ_TOOLS:
                expected_path = item.target_file
                actual_path = self._tool_path(call)
                if tool == "read_file" and expected_path and actual_path and expected_path != actual_path:
                    file_states = facts.get("file_states") if isinstance(facts.get("file_states"), dict) else {}
                    actual_state = file_states.get(actual_path) if isinstance(file_states.get(actual_path), dict) else {}
                    if (
                        int(actual_state.get("mutation_count", 0) or 0) > 0
                        and (
                            actual_state.get("needs_reread")
                            or not bool(actual_state.get("verified_after_mutation"))
                        )
                    ):
                        return GuardDecision(False)
                    return GuardDecision(
                        True,
                        reason=f"read_file target does not match current ledger item: expected {expected_path}, got {actual_path}",
                        corrective=(
                            "The current ledger item is a file change. Do not reread completed unrelated files; "
                            f"work on {expected_path} now with write_file/edit_file/apply_patch."
                        ),
                    )
                return GuardDecision(False)
            if tool not in FILE_MUTATION_TOOLS:
                return GuardDecision(
                    True,
                    reason=f"current ledger item requires file mutation, got {tool}",
                    corrective=(
                        "The current ledger item is a file change. Use write_file for new files "
                        "or edit_file/apply_patch for existing files."
                    ),
                )
            expected_path = item.target_file
            actual_path = self._tool_path(call)
            if expected_path and actual_path and expected_path != actual_path:
                if any(
                    pending.target_file == actual_path
                    and pending.type in {TaskType.CREATE_FILE, TaskType.EDIT_FILE, TaskType.PATCH}
                    for pending in self.ledger.pending()
                ):
                    return GuardDecision(False)
                return GuardDecision(
                    True,
                    reason=f"unrelated file for current ledger item: {actual_path}",
                    corrective=f"Work on the current ledger file only: {expected_path}.",
                )
            return GuardDecision(False)

        if item.type == TaskType.FIX:
            if tool in READ_TOOLS:
                return GuardDecision(False)
            if tool in COMMAND_TOOLS and not facts.get("repair_touched_relevant"):
                return GuardDecision(
                    True,
                    reason="repair command rerun blocked before relevant patch",
                    corrective="Patch a relevant file before rerunning the failed command.",
                )
            if tool not in (READ_TOOLS | FILE_MUTATION_TOOLS | COMMAND_TOOLS):
                return GuardDecision(
                    True,
                    reason=f"tool {tool} is unrelated to repair",
                    corrective="Use read_file plus edit_file/apply_patch on relevant files, then rerun the failed command.",
                )
        return GuardDecision(False)

    @staticmethod
    def _tool_path(call: ToolCall) -> str:
        if call.name == "apply_patch":
            return ""
        return str(call.args.get("path") or "")

    @staticmethod
    def _is_lightweight_verification_command(command: str) -> bool:
        normalized = normalize_command(command)
        return normalized.startswith("python -m py_compile ")
