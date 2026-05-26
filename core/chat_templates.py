"""
Chat templates для разных семейств моделей.

llama-cpp-python умеет применять chat_format сам, но:
1. Не все семейства покрыты корректно.
2. Мы хотим точный контроль над тем, что уходит в модель (для отладки превью).
3. Auto-detect по имени файла модели нужен, и его проще держать тут.
"""

from __future__ import annotations

from .profiles import ChatTemplate


def detect_template(model_filename: str) -> ChatTemplate:
    """Определяет шаблон по имени файла .gguf."""
    name = model_filename.lower()

    if "llama-3" in name or "llama3" in name:
        return ChatTemplate.LLAMA3
    if "mistral" in name or "mixtral" in name:
        return ChatTemplate.MISTRAL
    if "gemma" in name:
        return ChatTemplate.GEMMA
    if "deepseek" in name:
        return ChatTemplate.DEEPSEEK
    # qwen, hermes, dolphin, openchat, yi, nous и многие другие — chatml
    return ChatTemplate.CHATML


def format_prompt(
    template: ChatTemplate,
    system: str,
    user: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """
    Собирает финальный промпт под формат конкретной модели.

    history — список (user_msg, assistant_msg), последний завершённый турн идёт
    как assistant_msg=...; текущий запрос всегда передаётся через user.
    """
    history = history or []

    if template == ChatTemplate.LLAMA3:
        return _format_llama3(system, user, history)
    if template == ChatTemplate.MISTRAL:
        return _format_mistral(system, user, history)
    if template == ChatTemplate.GEMMA:
        return _format_gemma(system, user, history)
    if template == ChatTemplate.DEEPSEEK:
        return _format_deepseek(system, user, history)
    # CHATML и AUTO (auto разрешается выше по стеку)
    return _format_chatml(system, user, history)


# ============================================================
# ИМПЛЕМЕНТАЦИИ
# ============================================================

def _format_chatml(system: str, user: str, history: list[tuple[str, str]]) -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    for u, a in history:
        parts.append(f"<|im_start|>user\n{u}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{a}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _format_llama3(system: str, user: str, history: list[tuple[str, str]]) -> str:
    parts = ["<|begin_of_text|>"]
    if system:
        parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>")
    for u, a in history:
        parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>")
        parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>")
    parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _format_mistral(system: str, user: str, history: list[tuple[str, str]]) -> str:
    # Mistral не имеет отдельного system-токена — system клеится в первый user-turn
    out = "<s>"
    first = True
    for u, a in history:
        prefix = f"{system}\n\n{u}" if (first and system) else u
        out += f"[INST] {prefix} [/INST] {a}</s>"
        first = False
    final_user = f"{system}\n\n{user}" if (first and system) else user
    out += f"[INST] {final_user} [/INST]"
    return out


def _format_gemma(system: str, user: str, history: list[tuple[str, str]]) -> str:
    # У gemma тоже нет system — пихаем в первый user-turn
    parts = ["<bos>"]
    first = True
    for u, a in history:
        u_text = f"{system}\n\n{u}" if (first and system) else u
        parts.append(f"<start_of_turn>user\n{u_text}<end_of_turn>")
        parts.append(f"<start_of_turn>model\n{a}<end_of_turn>")
        first = False
    final_user = f"{system}\n\n{user}" if (first and system) else user
    parts.append(f"<start_of_turn>user\n{final_user}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


def _format_deepseek(system: str, user: str, history: list[tuple[str, str]]) -> str:
    parts = ["<｜begin▁of▁sentence｜>"]
    if system:
        parts.append(system)
    for u, a in history:
        parts.append(f"<｜User｜>{u}<｜Assistant｜>{a}<｜end▁of▁sentence｜>")
    parts.append(f"<｜User｜>{user}<｜Assistant｜>")
    return "".join(parts)


# ============================================================
# RENDER PERSONA
# ============================================================

def render_persona(template_string: str, persona: dict[str, str], user_name: str = "") -> str:
    """
    Подставляет {переменные} в системный промпт компаньона.

    Тихо игнорирует отсутствующие ключи — оставляет как было,
    чтобы не падать на пустых полях.
    """
    if not template_string:
        return ""

    result = template_string
    data = dict(persona)
    # user_name из аргумента имеет приоритет над пустым в persona
    if user_name:
        data["user_name"] = user_name
    # дефолт для user_name чтобы плейсхолдер не зиял
    if not data.get("user_name"):
        data["user_name"] = "you"

    for key, value in data.items():
        placeholder = "{" + key + "}"
        result = result.replace(placeholder, str(value) if value else "—")

    return result
