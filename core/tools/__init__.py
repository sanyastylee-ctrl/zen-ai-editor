from __future__ import annotations

from .base import Tool, ToolCall, ToolResult
from .edit import EditFileTool
from .list import ListFilesTool
from .patch import ApplyPatchTool
from .read import ReadFileTool
from .search import SearchFilesTool
from .term import RunTerminalTool
from .write import WriteFileTool


def default_tools(project_root: str | None = None) -> dict[str, Tool]:
    tools = [
        ReadFileTool(project_root),
        ListFilesTool(project_root),
        SearchFilesTool(project_root),
        WriteFileTool(project_root),
        EditFileTool(project_root),
        RunTerminalTool(project_root),
        ApplyPatchTool(project_root),
    ]
    result = {tool.name: tool for tool in tools}
    # Accept common model wording while keeping one implementation/policy per operation.
    result["create_file"] = result["write_file"]
    result["patch_file"] = result["apply_patch"]
    result["run_command"] = result["run_terminal"]
    return result


__all__ = [
    "Tool",
    "ToolCall",
    "ToolResult",
    "ReadFileTool",
    "ListFilesTool",
    "SearchFilesTool",
    "WriteFileTool",
    "EditFileTool",
    "RunTerminalTool",
    "ApplyPatchTool",
    "default_tools",
]
