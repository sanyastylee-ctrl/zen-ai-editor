"""
Финальная сборка промпта перед отправкой в модель.

Две формы выхода:
- build()          → BuiltPrompt с формат-строкой (для legacy completion).
- build_messages() → BuiltMessages со списком dict (для chat-completion и Vision).

Обе функции делят между собой одну и ту же логику:
- render_persona для COMPANION
- code_context + RAG для CODER
- trim истории по token-budget
- одинаково обработать VISION (как CODER без code_context, без RAG)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.profiles import AIProfile, ProfileKind, ChatTemplate
from core.chat_templates import format_prompt, detect_template, render_persona
from core.token_budget import TokenBudget


@dataclass
class BuiltPrompt:
    """Готовая строка-промпт (legacy completion)."""
    formatted: str
    system: str
    user: str
    history_used: int
    code_context_trimmed: bool


@dataclass
class BuiltMessages:
    """Список messages для chat-completion API (включая Vision)."""
    messages: list[dict]
    system: str
    history_used: int
    code_context_trimmed: bool


# ============================================================
# Внутреннее: общая подготовка system/user/history
# ============================================================

def _prepare(
    profile: AIProfile,
    user_message: str,
    code_context: str,
    rag_snippets: str,
    history: list[tuple[str, str]],
    user_name: str,
) -> tuple[str, str, list[tuple[str, str]], bool]:
    """Возвращает (system, user_content, trimmed_history, code_was_trimmed)."""
    budget = TokenBudget(
        n_ctx=profile.n_ctx,
        max_response_tokens=profile.max_tokens,
    )

    # 1. system
    if profile.kind == ProfileKind.COMPANION:
        system = render_persona(profile.system_prompt, profile.persona, user_name)
    else:
        system = profile.system_prompt
        
    # Внедрение правил создания файлов, если включен Agent Mode.
    # Формат намеренно использует тройные кавычки как закрытие блока — модель
    # органически закрывает их (Qwen-Coder натренирован на этом), и нам не нужно
    # надеяться, что она вспомнит про закрывающий тег.
    if getattr(profile, "agent_mode", False) and profile.kind == ProfileKind.CODER:
        agent_instructions = (
            "\n\n=== AGENT MODE ===\n"
            "You can create, edit, delete files and run terminal commands in the user's project.\n"
            "Use ONLY these exact formats. Do NOT invent variations.\n\n"
            "## Create or overwrite a file\n"
            "Write the marker, then the code inside a fenced code block. "
            "The closing ``` is what ends the operation — there is no closing marker.\n\n"
            "[FILE: path/to/file.py]\n"
            "```python\n"
            "def hello():\n"
            "    print('hi')\n"
            "```\n\n"
            "## Delete a file\n"
            "[DELETE: path/to/file.py]\n\n"
            "## Run a shell command\n"
            "[RUN: pytest tests/ -v]\n\n"
            "## Rules\n"
            "- Always use RELATIVE paths from the project root.\n"
            "- ONE file per [FILE:] marker. To create N files, write N markers.\n"
            "- Always specify the language after ``` (```python, ```javascript, ```html, etc).\n"
            "- Put a blank line between operation blocks.\n"
            "- Briefly explain your plan in plain text BEFORE the operation blocks.\n"
            "- After operations, you can summarize what you did.\n"
            "- Do NOT wrap explanations or summaries in [FILE:] markers — those are only for files to be saved.\n"
        )
        if "=== AGENT MODE ===" not in system:
            system += agent_instructions

    # 2. user-блок: кодеру и vision добавляем code_context/RAG/файлы
    code_trimmed = False
    if profile.kind == ProfileKind.CODER:
        parts = []
        if code_context.strip():
            trimmed_ctx, code_trimmed = budget.trim_code_context(
                code_context, system, user_message
            )
            if trimmed_ctx:
                parts.append(f"### Контекст проекта\n```\n{trimmed_ctx}\n```")
        if rag_snippets.strip():
            parts.append(f"### Релевантные фрагменты из проекта\n{rag_snippets}")
        parts.append(f"### Запрос\n{user_message}")
        user_content = "\n\n".join(parts)

    elif profile.kind == ProfileKind.VISION:
        # для Vision не пихаем RAG/деревья — фокус на картинке и тексте
        if code_context.strip():
            trimmed_ctx, code_trimmed = budget.trim_code_context(
                code_context, system, user_message
            )
            user_content = f"{user_message}\n\n### Контекст\n{trimmed_ctx}" if trimmed_ctx else user_message
        else:
            user_content = user_message

    else:  # COMPANION / GENERIC
        user_content = user_message
        if code_context.strip() and profile.kind != ProfileKind.COMPANION:
            user_content += f"\n\n### Context\n{code_context}"

    # 3. обрезка истории по бюджету
    if history:
        history = budget.trim_history(history, system, user_content, code_context)

    return system, user_content, history, code_trimmed


# ============================================================
# Публичные API
# ============================================================

def build(
    profile: AIProfile,
    user_message: str,
    code_context: str = "",
    rag_snippets: str = "",
    history: list[tuple[str, str]] | None = None,
    user_name: str = "",
) -> BuiltPrompt:
    """Старый API — собирает строку под формат модели."""
    history = history or []
    system, user_content, history, code_trimmed = _prepare(
        profile, user_message, code_context, rag_snippets, history, user_name
    )

    template = profile.chat_template
    if template == ChatTemplate.AUTO:
        template = detect_template(profile.model_file)

    formatted = format_prompt(template, system, user_content, history)

    return BuiltPrompt(
        formatted=formatted,
        system=system,
        user=user_content,
        history_used=len(history),
        code_context_trimmed=code_trimmed,
    )


def build_messages(
    profile: AIProfile,
    user_message: str,
    code_context: str = "",
    rag_snippets: str = "",
    history: list[tuple[str, str]] | None = None,
    user_name: str = "",
) -> BuiltMessages:
    """
    Новый API — собирает список messages для chat-completion.

    Используется для Vision (туда потом докидываются image_url) и в любых
    случаях, когда хочется идти через model.create_chat_completion().
    """
    history = history or []
    system, user_content, history, code_trimmed = _prepare(
        profile, user_message, code_context, rag_snippets, history, user_name
    )

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    for h_u, h_a in history:
        messages.append({"role": "user", "content": h_u})
        messages.append({"role": "assistant", "content": h_a})
    messages.append({"role": "user", "content": user_content})

    return BuiltMessages(
        messages=messages,
        system=system,
        history_used=len(history),
        code_context_trimmed=code_trimmed,
    )