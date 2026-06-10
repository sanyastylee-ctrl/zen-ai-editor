"""Coder Agent v3 support layer.

This package keeps the strict controller state separate from the legacy
AgentWorker loop. AgentWorker remains the Qt integration boundary.
"""

from .state import (
    AgentPhase,
    AgentRunStateV3,
    AgentIntent,
    FileGoal,
    FileGoalStatus,
    FileState,
    TaskLedgerItem,
    TaskStatus,
    TaskType,
)
from .verification import CommandGoal, extract_command_goals, normalize_command
from .ledger import TaskLedger
from .controller import CoderAgentController
from .evaluator import EvaluationResult, evaluate_final_readiness
from .file_goals import (
    detect_lazy_placeholders,
    deserialize_file_goals,
    extract_file_goals,
    serialize_file_goals,
    verify_file_goal,
)
from .project_map import build_project_map
from .repair import TracebackFrame, TracebackInfo, parse_traceback
from .checkpoint import (
    AgentRunCheckpoint,
    checkpoint_path,
    latest_resumable_checkpoint,
    load_checkpoint,
    mark_checkpoint_dismissed,
    mark_checkpoint_done,
    save_checkpoint,
)

__all__ = [
    "AgentPhase",
    "AgentRunStateV3",
    "AgentIntent",
    "AgentRunCheckpoint",
    "CommandGoal",
    "CoderAgentController",
    "EvaluationResult",
    "FileGoal",
    "FileGoalStatus",
    "FileState",
    "TaskLedger",
    "TaskLedgerItem",
    "TaskStatus",
    "TaskType",
    "TracebackFrame",
    "TracebackInfo",
    "detect_lazy_placeholders",
    "deserialize_file_goals",
    "extract_command_goals",
    "extract_file_goals",
    "evaluate_final_readiness",
    "build_project_map",
    "checkpoint_path",
    "latest_resumable_checkpoint",
    "load_checkpoint",
    "mark_checkpoint_dismissed",
    "mark_checkpoint_done",
    "normalize_command",
    "parse_traceback",
    "save_checkpoint",
    "serialize_file_goals",
    "verify_file_goal",
]
