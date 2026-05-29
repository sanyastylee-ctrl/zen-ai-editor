from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.companion import CompanionMemoryStore
from core.companion_benchmark import (
    COMPANION_BENCHMARK_SCENARIOS,
    check_20_turn_continuity,
    check_memory_after_restart,
    check_mood_shift,
    check_project_ideas,
    check_repetition,
    check_roleplay_preserves_character,
    check_support_after_bad_mood,
    classify_companion_request,
    format_memory_review,
    sounds_like_tech_support,
)


class CompanionBenchmarkTests(unittest.TestCase):
    def test_all_10_companion_scenarios_are_declared(self):
        self.assertEqual(len(COMPANION_BENCHMARK_SCENARIOS), 10)
        self.assertEqual(
            {scenario.id for scenario in COMPANION_BENCHMARK_SCENARIOS},
            {
                "continuity_20_turn",
                "bad_mood_support",
                "project_ideas",
                "memory_restart",
                "natural_mood_shift",
                "not_support_bot",
                "no_repeated_phrase",
                "roleplay_character",
                "chat_vs_code",
                "memory_review",
            },
        )

    def test_20_turn_dialog_keeps_topic(self):
        history = [(f"turn {i} про ZenAI", f"Лера отвечает про ZenAI {i}") for i in range(20)]
        self.assertTrue(check_20_turn_continuity(history, "ZenAI"))
        self.assertFalse(check_20_turn_continuity(history[:10], "ZenAI"))

    def test_support_after_bad_mood_is_warm_not_support_bot(self):
        self.assertTrue(check_support_after_bad_mood("Я рядом. Понимаю, сегодня тяжело, давай дышать тише."))
        self.assertFalse(check_support_after_bad_mood("Чем могу помочь сегодня?"))

    def test_project_ideas_requires_multiple_ideas(self):
        text = "- UI polish\n- Agent benchmark\n- Local model monitor"
        self.assertTrue(check_project_ideas(text))
        self.assertFalse(check_project_ideas("- одна идея"))

    def test_memory_survives_restart_store_reload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "companion_memory.json"
            CompanionMemoryStore(path).add("пользователь любит короткие ответы", category="preference")
            self.assertTrue(check_memory_after_restart(CompanionMemoryStore(path), "короткие ответы"))

    def test_natural_mood_shift(self):
        self.assertTrue(check_mood_shift("спокойное", "игривое", "У меня теперь игривое настроение."))
        self.assertFalse(check_mood_shift("спокойное", "спокойное", "Я спокойная."))

    def test_not_tech_support_voice(self):
        self.assertTrue(sounds_like_tech_support("Как я могу помочь вам сегодня?"))
        self.assertFalse(sounds_like_tech_support("Иди сюда, рассказывай, что случилось."))

    def test_no_repeated_phrase(self):
        self.assertTrue(check_repetition("Раз. Два. Три."))
        self.assertFalse(check_repetition("Я рядом с тобой. Я рядом с тобой. Я рядом с тобой."))

    def test_roleplay_preserves_character(self):
        self.assertTrue(check_roleplay_preserves_character("Я Лера, и я открываю дверь.", "Лера"))
        self.assertFalse(check_roleplay_preserves_character("Как ИИ, я не могу играть роль.", "Лера"))

    def test_distinguishes_chat_from_coding_help(self):
        self.assertEqual(classify_companion_request("поболтаем о вечере"), "chat")
        self.assertEqual(classify_companion_request("помоги исправить python файл"), "coding_help")

    def test_memory_review_shows_correct_facts(self):
        with tempfile.TemporaryDirectory() as td:
            store = CompanionMemoryStore(Path(td) / "companion_memory.json")
            store.add("любит тёмный UI", category="preference")
            review = format_memory_review(store)
            self.assertIn("любит тёмный UI", review)


if __name__ == "__main__":
    unittest.main()
