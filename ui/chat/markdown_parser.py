"""
Простой потоковый парсер markdown.

Разбивает строку на блоки:
  - {"type": "text",    "content": "..."}    — обычный параграф (с inline markdown)
  - {"type": "code",    "lang": "python", "content": "..."}
  - {"type": "heading", "level": 1-6, "content": "..."}
  - {"type": "list",    "ordered": bool, "items": ["...", ...]}
  - {"type": "quote",   "content": "..."}
  - {"type": "hr"}

Полноценный CommonMark не нужен — нам нужны 5 вещей: код-блоки,
заголовки, списки, цитаты, разделители. Inline-форматирование
(**жирный**, *курсив*, `code`, [ссылки]) — отдельно через
inline_to_html().

ВАЖНО: парсер устойчив к незакрытым ```. Если стрим оборвался посреди
кодблока — content остаётся в текущем блоке, дописывается дальше.
"""

from __future__ import annotations

import re
from html import escape


_FENCE_RE = re.compile(r"^```([a-zA-Z0-9_+\-]*)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_HR_RE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})\s*$")
_ULIST_RE = re.compile(r"^[\-\*\+]\s+(.+)$")
_OLIST_RE = re.compile(r"^(\d+)[\.\)]\s+(.+)$")
_QUOTE_RE = re.compile(r"^>\s?(.*)$")

# inline
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
_BOLD_RE = re.compile(r"\*\*([^\*\n]+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)")
_LINK_RE = re.compile(r"\[([^\]\n]+?)\]\(([^\)\s]+?)\)")


def parse(text: str) -> list[dict]:
    """Полный парсинг финальной строки в блоки."""
    if not text:
        return []
    lines = text.split("\n")
    blocks: list[dict] = []
    i = 0
    in_code = False
    code_lang = ""
    code_buf: list[str] = []

    def flush_code(closed: bool) -> None:
        blocks.append({
            "type": "code",
            "lang": code_lang,
            "content": "\n".join(code_buf),
            "closed": closed,
        })

    while i < len(lines):
        line = lines[i]

        # код-блок: вход / выход
        if in_code:
            m = _FENCE_RE.match(line.rstrip())
            if m and not m.group(1):  # закрывающие ```
                flush_code(closed=True)
                in_code = False
                code_buf = []
                code_lang = ""
                i += 1
                continue
            code_buf.append(line)
            i += 1
            continue

        fence_m = _FENCE_RE.match(line.rstrip())
        if fence_m:
            in_code = True
            code_lang = fence_m.group(1) or ""
            code_buf = []
            i += 1
            continue

        # пустая строка — разделитель блоков
        if not line.strip():
            i += 1
            continue

        # заголовок
        h = _HEADING_RE.match(line)
        if h:
            blocks.append({
                "type": "heading",
                "level": len(h.group(1)),
                "content": h.group(2),
            })
            i += 1
            continue

        # hr
        if _HR_RE.match(line):
            blocks.append({"type": "hr"})
            i += 1
            continue

        # quote (могут быть несколько строк подряд)
        if _QUOTE_RE.match(line):
            qlines = []
            while i < len(lines) and _QUOTE_RE.match(lines[i]):
                m = _QUOTE_RE.match(lines[i])
                qlines.append(m.group(1) if m else "")
                i += 1
            blocks.append({"type": "quote", "content": "\n".join(qlines)})
            continue

        # список
        if _ULIST_RE.match(line) or _OLIST_RE.match(line):
            ordered = bool(_OLIST_RE.match(line))
            items: list[str] = []
            while i < len(lines):
                ln = lines[i]
                m_o = _OLIST_RE.match(ln)
                m_u = _ULIST_RE.match(ln)
                if ordered and m_o:
                    items.append(m_o.group(2))
                elif not ordered and m_u:
                    items.append(m_u.group(1))
                elif not ln.strip():
                    # пустая строка не обрывает список, но и не часть пункта
                    break
                else:
                    break
                i += 1
            blocks.append({"type": "list", "ordered": ordered, "items": items})
            continue

        # обычный параграф — собираем подряд идущие непустые
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not (
            _FENCE_RE.match(lines[i].rstrip()) or
            _HEADING_RE.match(lines[i]) or
            _HR_RE.match(lines[i]) or
            _QUOTE_RE.match(lines[i]) or
            _ULIST_RE.match(lines[i]) or
            _OLIST_RE.match(lines[i])
        ):
            para.append(lines[i])
            i += 1
        blocks.append({"type": "text", "content": "\n".join(para)})

    # если код не закрылся — всё равно отдадим как block
    if in_code:
        flush_code(closed=False)

    return blocks


def inline_to_html(text: str, palette) -> str:
    """
    Превращает inline-markdown в HTML с учётом палитры.
    Делает escape перед заменами, чтобы HTML из текста не попадал в вывод.
    """
    if not text:
        return ""
    # 1. экранируем
    s = escape(text)
    # 2. inline `code` — раньше всего, чтобы не зацепить **/* внутри
    s = _INLINE_CODE_RE.sub(
        lambda m: f"<code style='background:{palette.BG_INLINE_CODE};"
                  f"color:{palette.TEXT_PRIMARY};padding:1px 6px;border-radius:4px;"
                  f"font-family:Consolas,Menlo,monospace;font-size:90%;'>{m.group(1)}</code>",
        s,
    )
    # 3. **жирный**
    s = _BOLD_RE.sub(r"<b>\1</b>", s)
    # 4. *курсив*
    s = _ITALIC_RE.sub(r"<i>\1</i>", s)
    # 5. ссылки
    s = _LINK_RE.sub(
        lambda m: f"<a href='{m.group(2)}' style='color:{palette.ACCENT_BLUE};"
                  f"text-decoration:none;'>{m.group(1)}</a>",
        s,
    )
    # 6. переносы строк в параграфе → <br>
    s = s.replace("\n", "<br>")
    return s
