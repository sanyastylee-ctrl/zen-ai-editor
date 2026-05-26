"""
Палитра, шрифты, отступы для UI чата.
Один источник правды — изменишь тут, поменяется везде.
"""

from __future__ import annotations

from PyQt6.QtGui import QFont, QFontDatabase


# ============================================================
# ПАЛИТРА
# ============================================================

class Palette:
    # фоны
    BG_APP        = "#16161D"           # самый тёмный — основной
    BG_CHAT       = "#1B1B23"           # фон скролла чата
    BG_USER       = "#2A2640"           # карточка юзера, фиолетовый отлив
    BG_ASSISTANT  = "#22232C"           # карточка ассистента
    BG_TOOL_OK    = "#1E2A22"           # tool успешно
    BG_TOOL_ERR   = "#2A1E22"           # tool ошибка
    BG_TOOL_RUN   = "#23252E"           # tool в процессе
    BG_SYSTEM     = "#1F1F26"           # system-сообщения
    BG_CODE       = "#0F0F14"           # код-блок
    BG_CODE_HEADER = "#1A1A22"          # шапка код-блока
    BG_INLINE_CODE = "#2A2A35"          # inline `code`

    # границы
    BORDER        = "#2D2D38"
    BORDER_LIGHT  = "#3A3A48"

    # текст
    TEXT_PRIMARY   = "#E6E6EC"
    TEXT_SECONDARY = "#9C9CAB"
    TEXT_DIM       = "#6D6D7A"
    TEXT_INVERTED  = "#16161D"

    # акценты
    ACCENT          = "#A78BFA"         # лавандовый — основной акцент
    ACCENT_HOVER    = "#B8A3FF"
    ACCENT_GREEN    = "#4DD49F"         # успех / tool ok
    ACCENT_RED      = "#F87171"         # ошибка
    ACCENT_AMBER    = "#FBBF24"         # warning
    ACCENT_BLUE     = "#7DD3FC"         # info / link / bash
    ACCENT_PINK     = "#F472B6"         # companion

    # код-подсветка (минимальная)
    CODE_KEYWORD = "#C792EA"
    CODE_STRING  = "#A5D6A7"
    CODE_NUMBER  = "#F78C6C"
    CODE_COMMENT = "#5C6370"
    CODE_FUNC    = "#82AAFF"


# ============================================================
# ШРИФТЫ
# ============================================================

# Inter и JetBrains Mono — если стоят в системе или embedded.
# Fallback порядок проверен на Win / macOS / Linux.
UI_FONT_FAMILIES = ["Inter", "SF Pro Display", "Segoe UI Variable", "Segoe UI", "Helvetica Neue", "Arial"]
MONO_FONT_FAMILIES = ["JetBrains Mono", "Cascadia Code", "Fira Code", "Consolas", "Menlo", "Courier New"]


def ui_font(size: int = 13, weight: int = QFont.Weight.Normal) -> QFont:
    db_families = set(QFontDatabase.families())
    family = next((f for f in UI_FONT_FAMILIES if f in db_families), UI_FONT_FAMILIES[-1])
    f = QFont(family, size)
    f.setWeight(weight)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return f


def mono_font(size: int = 12) -> QFont:
    db_families = set(QFontDatabase.families())
    family = next((f for f in MONO_FONT_FAMILIES if f in db_families), MONO_FONT_FAMILIES[-1])
    f = QFont(family, size)
    f.setStyleHint(QFont.StyleHint.TypeWriter)
    f.setFixedPitch(True)
    return f


# ============================================================
# ОТСТУПЫ
# ============================================================

class Spacing:
    MESSAGE_GAP        = 18          # между сообщениями
    CARD_PADDING       = 16          # внутри карточки
    CARD_RADIUS        = 14          # скругление карточек
    CODE_RADIUS        = 10          # скругление код-блоков
    AVATAR_SIZE        = 32
    AVATAR_GAP         = 12          # между аватаром и контентом
