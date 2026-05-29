from __future__ import annotations

from dataclasses import dataclass, field

from .ledger import TaskLedger
from .state import AgentRunStateV3, FileGoalStatus, TaskStatus


@dataclass(frozen=True)
class EvaluationResult:
    allowed: bool
    blockers: list[str] = field(default_factory=list)
    summary: str = ""

    @property
    def reason(self) -> str:
        return "; ".join(self.blockers)


def evaluate_final_readiness(
    state: AgentRunStateV3,
    ledger: TaskLedger,
    *,
    command_goals: list[str] | None = None,
    command_goals_done: set[str] | list[str] | None = None,
    active_repair: dict | None = None,
) -> EvaluationResult:
    """Decide if a coder run can produce a final answer."""

    blockers: list[str] = []
    pending = ledger.pending()
    if pending:
        # Check specifically for FAILED state which requires a Repair phase
        failed_items = [item for item in pending if item.status == TaskStatus.FAILED]
        if failed_items:
             blockers.append(
                "unresolved failures: "
                + ", ".join(item.id or item.description for item in failed_items[:3])
                + " (requires repair and re-verification)"
             )
        else:
            blockers.append(
                "pending ledger items: "
                + ", ".join(item.id or item.description for item in pending[:5])
            )

    goals = list(command_goals or state.command_goals or [])
    done = set(command_goals_done or state.command_goals_done or [])
    missing_goals = [goal for goal in goals if goal not in done]
    if missing_goals:
        blockers.append(
            "pending command goals: "
            + ", ".join(missing_goals[:5])
        )

    repair = active_repair if active_repair is not None else {}
    if repair:
        blockers.append(
            "pending repair: "
            + str(repair.get("description") or repair.get("failure_type") or repair.get("id") or "repair")
        )

    failed = list(state.failed_commands or [])
    if failed and missing_goals:
        blockers.append("failed commands need repair/rerun: " + ", ".join(failed[-3:]))

    missing_file_goals: list[str] = []
    failed_file_goals: list[str] = []
    pending_file_goals: list[str] = []
    for goal in getattr(state, "file_goals", []) or []:
        if not goal.required:
            continue
        if goal.status == FileGoalStatus.DONE:
            continue
        if goal.status == FileGoalStatus.FAILED:
            failed_file_goals.append(f"{goal.path}: {goal.failure_reason or 'verification failed'}")
        elif goal.must_exist:
            missing_file_goals.append(goal.path)
        else:
            pending_file_goals.append(goal.path)
    if missing_file_goals:
        blockers.append("missing required files: " + ", ".join(missing_file_goals[:5]))
    if failed_file_goals:
        blockers.append("failed file goals: " + "; ".join(failed_file_goals[:3]))
    if pending_file_goals:
        blockers.append("pending file goals: " + ", ".join(pending_file_goals[:5]))

    allowed = not blockers
    if allowed:
        summary = "all ledger items and required command goals are verified"
    else:
        summary = blockers[0]
    return EvaluationResult(allowed=allowed, blockers=blockers, summary=summary)
