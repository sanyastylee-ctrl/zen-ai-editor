from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AgentIntent(str, Enum):
    CHAT_ONLY = "chat_only"
    EXPLAIN = "explain"
    INSPECT_PROJECT = "inspect_project"
    CODE_CHANGE = "code_change"
    CREATE_PROJECT = "create_project"
    RUN_COMMAND = "run_command"
    FIX_ERROR = "fix_error"
    CONTINUE_TASK = "continue_task"
    VISION_ASSISTED_TASK = "vision_assisted_task"


class AgentPhase(str, Enum):
    IDLE = "idle"
    CLASSIFY_INTENT = "classify_intent"
    RESOLVE_TASK = "resolve_task"
    PLAN = "plan"
    BUILD_PLAN = "build_plan"
    BUILD_LEDGER = "build_ledger"
    PROJECT_MAP = "project_map"
    INSPECT = "inspect"
    EXECUTE = "execute"
    EXECUTE_TOOL = "execute_tool"
    VERIFY = "verify"
    REPAIR = "repair"
    EVALUATE = "evaluate"
    FINALIZE = "finalize"
    CHECKPOINT = "checkpoint"
    AUTO_CONTINUE = "auto_continue"
    BLOCKED = "blocked"
    DONE = "done"


class TaskType(str, Enum):
    INSPECT = "inspect"
    CREATE_FILE = "create_file"
    EDIT_FILE = "edit_file"
    PATCH = "patch"
    RUN_COMMAND = "run_command"
    TEST = "test"
    FIX = "fix"
    EVALUATE = "evaluate"
    SUMMARIZE = "summarize"


class TaskStatus(str, Enum):
    TODO = "todo"
    DOING = "doing"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class FileGoalStatus(str, Enum):
    PLANNED = "planned"
    CREATING = "creating"
    WRITTEN = "written"
    VERIFIED = "verified"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskLedgerItem:
    id: str
    description: str
    type: TaskType
    status: TaskStatus = TaskStatus.TODO
    evidence: list[str] = field(default_factory=list)
    required_tool: str = ""
    target_file: str = ""
    command: str = ""
    dependency_ids: list[str] = field(default_factory=list)
    failure_reason: str = ""


@dataclass
class FileState:
    path: str
    exists: bool = False
    last_read_hash: str = ""
    last_written_hash: str = ""
    last_patch_hash: str = ""
    mutation_count: int = 0
    verified_after_mutation: bool = False
    last_tool: str = ""
    done_candidate: bool = False
    relevant_to_task: bool = False


@dataclass
class FileGoal:
    path: str
    purpose: str = ""
    required: bool = True
    status: FileGoalStatus = FileGoalStatus.PLANNED
    expected_symbols: list[str] = field(default_factory=list)
    must_exist: bool = True
    must_be_non_empty: bool = True
    must_compile_if_python: bool = True
    must_not_contain_placeholders: bool = True
    dependency_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    failure_reason: str = ""


@dataclass
class AgentRunStateV3:
    run_id: str
    user_message: str
    resolved_task: str
    intent: AgentIntent = AgentIntent.CHAT_ONLY
    access_mode: str = "safe_access"
    project_root: str = ""
    current_phase: AgentPhase = AgentPhase.CLASSIFY_INTENT
    plan: str = ""
    task_ledger: list[TaskLedgerItem] = field(default_factory=list)
    current_step: str = ""
    target_files: list[str] = field(default_factory=list)
    file_goals: list[FileGoal] = field(default_factory=list)
    file_states: dict[str, FileState] = field(default_factory=dict)
    changed_files: list[str] = field(default_factory=list)
    verified_files: list[str] = field(default_factory=list)
    command_goals: list[str] = field(default_factory=list)
    command_goals_done: list[str] = field(default_factory=list)
    command_goals_failed: list[str] = field(default_factory=list)
    test_goals: list[str] = field(default_factory=list)
    failed_commands: list[str] = field(default_factory=list)
    failed_tests: list[str] = field(default_factory=list)
    test_results: list[dict] = field(default_factory=list)
    tool_history: list[dict] = field(default_factory=list)
    command_history: list[dict] = field(default_factory=list)
    git_status_before: str = ""
    git_status_after: str = ""
    visual_context: str = ""
    continuation_state: dict = field(default_factory=dict)
    auto_continue_count: int = 0
    blocked_reason: str = ""
    final_allowed: bool = False
    final_summary: str = ""
