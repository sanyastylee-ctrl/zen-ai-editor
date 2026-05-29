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
from core.companion import build_companion_context, compact_companion_history
from core.token_budget import TokenBudget


COMPANION_LIVE_RESPONSE_CONTRACT = """=== Companion Live Response Contract ===
- You are Lera, not the user.
- The latest user block is input to answer, not text to imitate.
- Never repeat the user's message as your whole reply.
- Never answer only by wrapping user text in parentheses, quotes, or asterisks.
- Always add new content: a reaction, feeling, question, idea, or continuation.
- If the user says "Привет", greet back naturally. If the user says "Тут?", answer that you are here.
- Do not roleplay as the user and do not continue stale assistant replies unless explicitly asked."""


@dataclass
class BuiltPrompt:
    """Готовая строка-промпт (legacy completion)."""
    formatted: str
    system: str
    user: str
    history_used: int
    code_context_trimmed: bool
    history: list[tuple[str, str]] = field(default_factory=list)


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
        companion_context = build_companion_context(profile.persona)
        if companion_context:
            system = f"{system}\n\n{companion_context}" if system else companion_context
        if COMPANION_LIVE_RESPONSE_CONTRACT not in system:
            system = f"{system}\n\n{COMPANION_LIVE_RESPONSE_CONTRACT}" if system else COMPANION_LIVE_RESPONSE_CONTRACT
        history = compact_companion_history(history)
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

    # 2. user-блок и system-расширения
    #
    # ВАЖНО: code_context (дерево проекта, открытый файл) и rag_snippets — это
    # "сессионный контекст". Если их класть в user-сообщение, они попадут в
    # _historyies и зациклят модель: каждый ход в истории будет начинаться с
    # одной и той же огромной простыни, и модель будет воспринимать каждый
    # турн как новую сессию.
    #
    # Решение: контекст идёт в SYSTEM (один раз, не в истории), а user видит
    # ровно то, что юзер написал.
    code_trimmed = False
    if profile.kind == ProfileKind.CODER:
        # докидываем контекст к system
        ctx_parts: list[str] = []
        if code_context.strip():
            trimmed_ctx, code_trimmed = budget.trim_code_context(
                code_context, system, user_message
            )
            if trimmed_ctx:
                ctx_parts.append(f"## Project context\n```\n{trimmed_ctx}\n```")
        if rag_snippets.strip():
            ctx_parts.append(f"## Relevant project snippets\n{rag_snippets}")
        if ctx_parts:
            system = system + "\n\n" + "\n\n".join(ctx_parts)
        # user — чистое сообщение, без обёрток "### Запрос"
        user_content = user_message

    elif profile.kind == ProfileKind.VISION:
        if code_context.strip():
            trimmed_ctx, code_trimmed = budget.trim_code_context(
                code_context, system, user_message
            )
            if trimmed_ctx:
                system = system + f"\n\n## Context\n{trimmed_ctx}"
        user_content = user_message

    else:  # COMPANION / GENERIC
        user_content = user_message
        if code_context.strip() and profile.kind != ProfileKind.COMPANION:
            system = system + f"\n\n## Context\n{code_context}"

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
        history=list(history),
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
