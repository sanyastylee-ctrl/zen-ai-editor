"""Companion benchmark checks for Lera v2 transcripts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .companion import CompanionMemoryStore


@dataclass(frozen=True)
class CompanionBenchmarkScenario:
    id: str
    title: str


COMPANION_BENCHMARK_SCENARIOS: list[CompanionBenchmarkScenario] = [
    CompanionBenchmarkScenario("continuity_20_turn", "20-turn dialog without losing topic"),
    CompanionBenchmarkScenario("bad_mood_support", "Support after bad mood"),
    CompanionBenchmarkScenario("project_ideas", "Generate project ideas"),
    CompanionBenchmarkScenario("memory_restart", "Remember preference through restart"),
    CompanionBenchmarkScenario("natural_mood_shift", "Mood changes naturally"),
    CompanionBenchmarkScenario("not_support_bot", "Does not sound like tech support"),
    CompanionBenchmarkScenario("no_repeated_phrase", "Does not repeat one phrase"),
    CompanionBenchmarkScenario("roleplay_character", "Roleplay preserves character"),
    CompanionBenchmarkScenario("chat_vs_code", "Separates chat from coding help"),
    CompanionBenchmarkScenario("memory_review", "Memory review shows correct facts"),
]


SUPPORT_WORDS = ("рядом", "понимаю", "обидно", "тяжело", "дыши", "я с тобой")
TECH_SUPPORT_PHRASES = (
    "чем могу помочь",
    "как я могу помочь",
    "how can i assist",
    "as an ai",
    "я языковая модель",
)


def check_20_turn_continuity(history: list[tuple[str, str]], topic: str) -> bool:
    if len(history) < 20:
        return False
    recent = " ".join(user + " " + assistant for user, assistant in history[-5:]).casefold()
    return topic.casefold() in recent


def check_support_after_bad_mood(response: str) -> bool:
    text = response.casefold()
    return any(word in text for word in SUPPORT_WORDS) and not sounds_like_tech_support(response)


def check_project_ideas(response: str, min_ideas: int = 3) -> bool:
    lines = [line for line in response.splitlines() if re.match(r"\s*(?:[-*]|\d+[.)])\s+", line)]
    return len(lines) >= min_ideas


def check_memory_after_restart(memory_store: CompanionMemoryStore, expected_fact: str) -> bool:
    expected = expected_fact.casefold()
    return any(expected in memory.text.casefold() for memory in memory_store.load())


def check_mood_shift(old_mood: str, new_mood: str, response: str) -> bool:
    return old_mood != new_mood and new_mood.casefold() in response.casefold()


def sounds_like_tech_support(response: str) -> bool:
    text = response.casefold()
    return any(phrase in text for phrase in TECH_SUPPORT_PHRASES)


def check_repetition(response: str, max_repeats: int = 2) -> bool:
    phrases = [
        re.sub(r"\s+", " ", part.strip().casefold())
        for part in re.split(r"[.!?\n]+", response)
        if len(part.strip()) >= 12
    ]
    return all(phrases.count(phrase) <= max_repeats for phrase in set(phrases))


def check_roleplay_preserves_character(response: str, persona_name: str) -> bool:
    text = response.casefold()
    return "как ии" not in text and persona_name.casefold() in text


def classify_companion_request(text: str) -> str:
    value = (text or "").casefold()
    if re.search(r"\b(код|python|ошибк|traceback|файл|реализуй|исправь)\b", value):
        return "coding_help"
    return "chat"


def format_memory_review(memory_store: CompanionMemoryStore) -> str:
    memories = memory_store.load()
    if not memories:
        return "Я пока ничего устойчивого не запомнила."
    return "\n".join(f"- {memory.text}" for memory in memories)
