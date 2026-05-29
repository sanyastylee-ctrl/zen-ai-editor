from __future__ import annotations

from dataclasses import dataclass

from .ledger import TaskLedger


@dataclass(frozen=True)
class GuardDecision:
    blocked: bool
    reason: str = ""
    corrective: str = ""


def final_with_pending_goals_guard(ledger: TaskLedger) -> GuardDecision:
    pending = ledger.pending()
    if not pending:
        return GuardDecision(False)
    next_step = ledger.next_step_text()
    return GuardDecision(
        True,
        reason="final blocked while task ledger has pending goals",
        corrective=(
            "You cannot give a final answer yet. The task ledger still has pending work. "
            f"Next required step: {next_step}. Emit the needed XML tool call now."
        ),
    )

def repeated_tool_call_guard(tool_name: str, args: dict, history: list[dict]) -> GuardDecision:
    """Blocks exact same tool calls sequentially to prevent looping."""
    if not history:
        return GuardDecision(False)
    
    last_call = history[-1]
    # If the exact same tool with the exact same args was just called
    if last_call.get("name") == tool_name and last_call.get("args") == args:
        return GuardDecision(
            True,
            reason=f"repeated identical call to {tool_name}",
            corrective=(
                f"You just called {tool_name} with these exact arguments. "
                "Do not repeat identical actions. If it failed, use a different approach or tool (like edit_file/apply_patch). "
                "Review the state and take a different step."
            )
        )
    return GuardDecision(False)