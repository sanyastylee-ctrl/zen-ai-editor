"""
Финальная сборка промпта перед отправкой в модель.

Бьём задачу на две ветки:
- CODER: системный промпт + код-контекст + RAG + user-запрос.
- COMPANION: персона-промпт (с подстановкой) + история диалога + user-запрос.

Эту логику намеренно держим в одном месте, чтобы был один диалог "вот это
именно то, что уходит в модель" — для отладки.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.profiles import AIProfile, ProfileKind, ChatTemplate
from core.chat_templates import format_prompt, detect_template, render_persona
from core.token_budget import TokenBudget


@dataclass
class BuiltPrompt:
    """Что реально уйдёт в модель + диагностика."""
    formatted: str          # готовая строка для llama.generate()
    system: str             # отдельно — для превью
    user: str
    history_used: int       # сколько турнов истории взяли
    code_context_trimmed: bool


def build(
    profile: AIProfile,
    user_message: str,
    code_context: str = "",
    rag_snippets: str = "",
    history: list[tuple[str, str]] | None = None,
    user_name: str = "",
) -> BuiltPrompt:
    """
    Главная точка входа. Возвращает готовый промпт.
    """
    history = history or []
    budget = TokenBudget(
        n_ctx=profile.n_ctx,
        max_response_tokens=profile.max_tokens,
    )

    # 1. Резолвим шаблон чата
    template = profile.chat_template
    if template == ChatTemplate.AUTO:
        template = detect_template(profile.model_file)

    # 2. Готовим system по типу профиля
    if profile.kind == ProfileKind.COMPANION:
        system = render_persona(profile.system_prompt, profile.persona, user_name)
        user_content = user_message
        used_code_context = ""
        code_trimmed = False

    elif profile.kind == ProfileKind.CODER:
        system = profile.system_prompt
        # код и RAG идут в user-блок как контекст
        parts = []
        if code_context.strip():
            trimmed_ctx, code_trimmed = budget.trim_code_context(
                code_context, system, user_message
            )
            if trimmed_ctx:
                parts.append(f"### Current file context\n```\n{trimmed_ctx}\n```")
        else:
            code_trimmed = False
            used_code_context = ""
        if rag_snippets.strip():
            parts.append(f"### Relevant code from project\n{rag_snippets}")
        parts.append(f"### Task\n{user_message}")
        user_content = "\n\n".join(parts)
        used_code_context = code_context

    else:  # GENERIC
        system = profile.system_prompt
        user_content = user_message
        if code_context.strip():
            user_content += f"\n\n### Context\n{code_context}"
        code_trimmed = False

    # 3. Обрезаем историю по бюджету (только если она есть)
    if history:
        history = budget.trim_history(history, system, user_content, code_context)

    # 4. Форматируем под шаблон модели
    formatted = format_prompt(template, system, user_content, history)

    return BuiltPrompt(
        formatted=formatted,
        system=system,
        user=user_content,
        history_used=len(history),
        code_context_trimmed=code_trimmed,
    )
