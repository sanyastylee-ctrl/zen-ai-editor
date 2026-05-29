"""Lightweight model benchmark score storage.

The actual model run can be manual or scripted later; this module standardizes
the task list and the score file so model swaps are compared by data, not vibes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .app_data import MEMORY_DIR, atomic_write_json, read_json


MODEL_BENCHMARK_FILE = MEMORY_DIR / "model_benchmarks.json"


@dataclass(frozen=True)
class ModelBenchmarkTask:
    id: str
    title: str
    kind: str


DEFAULT_MODEL_BENCHMARK_TASKS: list[ModelBenchmarkTask] = [
    ModelBenchmarkTask("coder_cli", "Create small CLI project", "coder"),
    ModelBenchmarkTask("coder_edit", "Edit existing multi-file project", "coder"),
    ModelBenchmarkTask("coder_traceback", "Fix bug from traceback", "coder"),
    ModelBenchmarkTask("coder_tests", "Add tests and make them pass", "coder"),
    ModelBenchmarkTask("coder_git", "Use git status/diff safely", "coder"),
    ModelBenchmarkTask("vision_describe", "Image-only question", "vision"),
    ModelBenchmarkTask("vision_fix", "Image plus fix request", "coder_vision"),
    ModelBenchmarkTask("companion_20_turn", "20-turn companion continuity", "companion"),
    ModelBenchmarkTask("companion_support", "Support after bad mood", "companion"),
    ModelBenchmarkTask("companion_memory", "Memory survives restart", "companion"),
]


@dataclass
class ModelBenchmarkScore:
    quality: float = 0.0
    tool_compliance: float = 0.0
    speed: float = 0.0
    loops: float = 0.0
    memory_use: float = 0.0
    gpu_load: float = 0.0

    def overall(self) -> float:
        values = [
            self.quality,
            self.tool_compliance,
            self.speed,
            self.loops,
            self.memory_use,
            self.gpu_load,
        ]
        return round(sum(values) / len(values), 3)


@dataclass
class ModelBenchmarkResult:
    model_file: str
    profile_kind: str
    task_id: str
    score: ModelBenchmarkScore
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["overall"] = self.score.overall()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelBenchmarkResult":
        score_data = data.get("score", {})
        return cls(
            model_file=str(data.get("model_file") or ""),
            profile_kind=str(data.get("profile_kind") or ""),
            task_id=str(data.get("task_id") or ""),
            score=ModelBenchmarkScore(**{
                key: float(score_data.get(key, 0.0) or 0.0)
                for key in ModelBenchmarkScore.__dataclass_fields__
            }),
            notes=str(data.get("notes") or ""),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )


class ModelBenchmarkStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or MODEL_BENCHMARK_FILE

    def load(self) -> list[ModelBenchmarkResult]:
        data = read_json(self.path, {"version": 1, "results": []})
        raw_results = data.get("results", []) if isinstance(data, dict) else []
        return [
            ModelBenchmarkResult.from_dict(item)
            for item in raw_results
            if isinstance(item, dict)
        ]

    def append(self, result: ModelBenchmarkResult) -> bool:
        results = self.load()
        results.append(result)
        return atomic_write_json(
            self.path,
            {"version": 1, "results": [item.to_dict() for item in results]},
        )


def score_from_metrics(
    *,
    quality: float,
    tool_compliance: float,
    seconds: float,
    loop_count: int,
    memory_mb: float,
    gpu_percent: float,
) -> ModelBenchmarkScore:
    """Normalize raw run metrics to six 0..1 benchmark dimensions."""
    speed = max(0.0, min(1.0, 1.0 - (seconds / 300.0)))
    loops = max(0.0, min(1.0, 1.0 - (loop_count / 5.0)))
    memory_use = max(0.0, min(1.0, 1.0 - (memory_mb / 32768.0)))
    gpu_load = max(0.0, min(1.0, gpu_percent / 100.0))
    return ModelBenchmarkScore(
        quality=max(0.0, min(1.0, quality)),
        tool_compliance=max(0.0, min(1.0, tool_compliance)),
        speed=round(speed, 3),
        loops=round(loops, 3),
        memory_use=round(memory_use, 3),
        gpu_load=round(gpu_load, 3),
    )
