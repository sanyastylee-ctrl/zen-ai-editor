"""
Эстимация и обрезка контекста под n_ctx модели.

Эстимация по символам — это грубо, но для UI-подсказки "сколько токенов
ты сжёг" достаточно. Точный подсчёт делается уже в самой Llama при инференсе.
"""

from __future__ import annotations


class TokenBudget:
    """
    Один экземпляр на профиль, потому что n_ctx и max_tokens у каждого свои.
    """

    # Эмпирически для русско-английского кода ~3.5 символа на токен
    CHARS_PER_TOKEN = 3.5

    def __init__(
        self,
        n_ctx: int = 8192,
        max_response_tokens: int = 2048,
        system_reserve: int = 512,
    ) -> None:
        self.n_ctx = n_ctx
        self.max_response_tokens = max_response_tokens
        self.system_reserve = system_reserve

    # ---------- расчёты ----------

    @property
    def input_budget(self) -> int:
        """Сколько токенов мы можем потратить на (system + history + user + context)."""
        return self.n_ctx - self.max_response_tokens

    @classmethod
    def estimate(cls, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / cls.CHARS_PER_TOKEN))

    def usage_percent(self, *parts: str) -> int:
        used = sum(self.estimate(p) for p in parts)
        return min(100, int(used / self.n_ctx * 100))

    # ---------- обрезка ----------

    def trim_code_context(
        self,
        code_context: str,
        system: str,
        user: str,
        history_text: str = "",
    ) -> tuple[str, bool]:
        """
        Если входной материал не помещается — обрезает code_context с КОНЦА
        (хвост обычно менее важен, чем начало файла).

        Возвращает (trimmed_context, was_trimmed).
        """
        sys_t = self.estimate(system)
        usr_t = self.estimate(user)
        hist_t = self.estimate(history_text)

        available = self.input_budget - sys_t - usr_t - hist_t
        if available <= 0:
            return "", True

        ctx_t = self.estimate(code_context)
        if ctx_t <= available:
            return code_context, False

        max_chars = int(available * self.CHARS_PER_TOKEN)
        return code_context[:max_chars], True

    def trim_history(
        self,
        history: list[tuple[str, str]],
        system: str,
        user: str,
        code_context: str = "",
    ) -> list[tuple[str, str]]:
        """
        Для компаньона: история разговора может разрастись. Обрезаем СТАРЫЕ турны
        (компаньон должен помнить недавнее, древнее не критично).
        """
        sys_t = self.estimate(system)
        usr_t = self.estimate(user)
        ctx_t = self.estimate(code_context)
        available = self.input_budget - sys_t - usr_t - ctx_t

        if available <= 0:
            return []

        # Считаем с конца, обрываем когда упёрлись
        kept: list[tuple[str, str]] = []
        used = 0
        for u, a in reversed(history):
            turn_t = self.estimate(u) + self.estimate(a)
            if used + turn_t > available:
                break
            kept.append((u, a))
            used += turn_t

        return list(reversed(kept))
