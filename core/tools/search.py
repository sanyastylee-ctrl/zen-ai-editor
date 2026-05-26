from __future__ import annotations

import os
import re

from .base import Tool, ToolCall, ToolResult


class SearchFilesTool(Tool):
    name = "search_files"
    description = "Search text files in the current project."

    SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", ".zen_ai"}
    TEXT_EXTS = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt", ".toml",
        ".yaml", ".yml", ".html", ".css", ".scss", ".cs", ".cpp", ".h",
    }

    def execute(self, call: ToolCall) -> ToolResult:
        query = call.args.get("query", "").strip()
        if not query:
            return ToolResult.error("missing query")
        try:
            root = self.safe_path(call.args.get("path", ""))
            max_results = int(call.args.get("max_results", "80") or "80")
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            matches: list[str] = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS and not d.startswith(".")]
                for fname in filenames:
                    if len(matches) >= max_results:
                        break
                    ext = os.path.splitext(fname)[1].lower()
                    if ext and ext not in self.TEXT_EXTS:
                        continue
                    fpath = os.path.join(dirpath, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            for line_no, line in enumerate(f, 1):
                                if pattern.search(line):
                                    rel = self.relpath(fpath)
                                    matches.append(f"{rel}:{line_no}: {line.rstrip()}")
                                    if len(matches) >= max_results:
                                        break
                    except OSError:
                        continue
            return ToolResult(
                ok=True,
                title=f"Search: {query}",
                output="\n".join(matches) if matches else "[no matches]",
                meta={"query": query, "count": len(matches)},
            )
        except ValueError:
            return ToolResult.error("path outside project")
        except Exception as e:
            return ToolResult.error(str(e))
