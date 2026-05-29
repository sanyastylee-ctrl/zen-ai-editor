from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.model_benchmark import (
    DEFAULT_MODEL_BENCHMARK_TASKS,
    ModelBenchmarkResult,
    ModelBenchmarkScore,
    ModelBenchmarkStore,
    score_from_metrics,
)
from core.profiles import (
    DEFAULT_CODER_MODEL_FILE,
    DEFAULT_COMPANION_MODEL_FILE,
    DEFAULT_VISION_MMPROJ_FILE,
    DEFAULT_VISION_MODEL_FILE,
    ProfileKind,
    ProfileManager,
)


class ModelBenchmarkTests(unittest.TestCase):
    def test_default_model_benchmark_tasks_cover_core_modes(self):
        kinds = {task.kind for task in DEFAULT_MODEL_BENCHMARK_TASKS}
        ids = {task.id for task in DEFAULT_MODEL_BENCHMARK_TASKS}

        self.assertIn("coder", kinds)
        self.assertIn("companion", kinds)
        self.assertIn("vision", kinds)
        self.assertIn("coder_vision", kinds)
        self.assertIn("companion_memory", ids)

    def test_score_from_metrics_normalizes_dimensions(self):
        score = score_from_metrics(
            quality=0.9,
            tool_compliance=1.2,
            seconds=30,
            loop_count=1,
            memory_mb=4096,
            gpu_percent=85,
        )

        self.assertEqual(score.tool_compliance, 1.0)
        self.assertGreater(score.speed, 0.8)
        self.assertGreater(score.loops, 0.7)
        self.assertAlmostEqual(score.gpu_load, 0.85)
        self.assertGreater(score.overall(), 0.7)

    def test_model_benchmark_store_persists_results(self):
        with tempfile.TemporaryDirectory() as td:
            store = ModelBenchmarkStore(Path(td) / "model_benchmarks.json")
            result = ModelBenchmarkResult(
                model_file="candidate.gguf",
                profile_kind="coder",
                task_id="coder_cli",
                score=ModelBenchmarkScore(quality=0.8, tool_compliance=0.9),
                notes="ok",
            )

            self.assertTrue(store.append(result))
            loaded = store.load()

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].model_file, "candidate.gguf")
            self.assertEqual(loaded[0].score.quality, 0.8)

    def test_seeded_default_profiles_use_v2_models_and_parameters(self):
        pm = ProfileManager()
        pm._seed_defaults()
        coder = pm.get_active(ProfileKind.CODER)
        companion = pm.get_active(ProfileKind.COMPANION)
        vision = pm.get_active(ProfileKind.VISION)

        self.assertEqual(coder.model_file, DEFAULT_CODER_MODEL_FILE)
        self.assertEqual(coder.n_ctx, 10240)
        self.assertEqual(coder.n_gpu_layers, -1)
        self.assertGreaterEqual(coder.repeat_penalty, 1.08)
        self.assertLessEqual(coder.repeat_penalty, 1.12)
        self.assertEqual(companion.model_file, DEFAULT_COMPANION_MODEL_FILE)
        self.assertGreaterEqual(companion.temperature, 0.85)
        self.assertLessEqual(companion.temperature, 0.95)
        self.assertGreaterEqual(companion.max_tokens, 1536)
        self.assertEqual(vision.model_file, DEFAULT_VISION_MODEL_FILE)
        self.assertEqual(vision.mmproj_file, DEFAULT_VISION_MMPROJ_FILE)


if __name__ == "__main__":
    unittest.main()
