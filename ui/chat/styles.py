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
    BG_PANEL      = "#1F2028"           # панели/сайдбар/workspace
    BG_INPUT      = "#2A2C38"           # поля и вторичные кнопки
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
    # форсируем субпиксельный антиалиасинг — на Windows без этого шрифт тонкий и дешёвый
    f.setStyleStrategy(
        QFont.StyleStrategy.PreferAntialias |
        QFont.StyleStrategy.PreferOutline
    )
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return f


def chat_font(size: int = 14) -> QFont:
    """Шрифт для текста сообщений — чуть крупнее и чуть жирнее чем ui_font."""
    return ui_font(size, weight=QFont.Weight.Normal)


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
    MESSAGE_GAP        = 14          # между сообщениями
    CARD_PADDING       = 14          # внутри карточки
    CARD_RADIUS        = 14          # скругление карточек
    CODE_RADIUS        = 9           # скругление код-блоков
    AVATAR_SIZE        = 30
    AVATAR_GAP         = 10          # между аватаром и контентом


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(167,139,250,{alpha:.2f})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def form_controls_qss() -> str:
    """Shared QSS for settings/profile/persona forms."""
    accent_soft = _hex_to_rgba(Palette.ACCENT, 0.10)
    accent_softer = _hex_to_rgba(Palette.ACCENT, 0.06)
    return f"""
        QWidget {{
            background: transparent;
            color: {Palette.TEXT_PRIMARY};
            font-size: 12px;
        }}
        QLabel {{
            color: {Palette.TEXT_SECONDARY};
            font-size: 12px;
        }}
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {Palette.BG_CODE};
            color: {Palette.TEXT_PRIMARY};
            border: 1px solid {Palette.BORDER};
            border-radius: 6px;
            padding: 6px 9px;
            selection-background-color: {accent_soft};
            selection-color: {Palette.TEXT_PRIMARY};
        }}
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {Palette.ACCENT};
        }}
        QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled,
        QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            color: {Palette.TEXT_DIM};
            border-color: {Palette.BORDER};
            background: {Palette.BG_CODE_HEADER};
        }}
        QLineEdit[placeholderText], QTextEdit[placeholderText] {{
            color: {Palette.TEXT_PRIMARY};
        }}
        QComboBox {{
            padding-right: 9px;
        }}
        QComboBox::drop-down {{
            width: 0;
            border: none;
            background: transparent;
        }}
        QComboBox::down-arrow {{
            image: none;
            width: 0;
            height: 0;
        }}
        QComboBox QAbstractItemView {{
            background: {Palette.BG_ASSISTANT};
            color: {Palette.TEXT_PRIMARY};
            border: 1px solid {Palette.BORDER_LIGHT};
            selection-background-color: {Palette.ACCENT};
            selection-color: white;
            outline: none;
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background: transparent;
            border: none;
            width: 18px;
        }}
        QCheckBox {{
            color: {Palette.TEXT_PRIMARY};
            spacing: 8px;
            padding: 2px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {Palette.BORDER_LIGHT};
            border-radius: 4px;
            background: {Palette.BG_CODE};
        }}
        QCheckBox::indicator:hover {{
            border-color: {Palette.ACCENT};
            background: {accent_softer};
        }}
        QCheckBox::indicator:checked {{
            background: {Palette.ACCENT};
            border-color: {Palette.ACCENT};
        }}
        QPushButton {{
            background: {Palette.ACCENT};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background: {Palette.ACCENT_HOVER};
        }}
        QPushButton:pressed {{
            background: #8B6FE8;
        }}
        QPushButton#secondary, QPushButton#secondaryCompact, QPushButton#danger {{
            background: transparent;
            color: {Palette.TEXT_SECONDARY};
            border: 1px solid {Palette.BORDER};
        }}
        QPushButton#secondary:hover, QPushButton#secondaryCompact:hover, QPushButton#danger:hover {{
            background: {accent_softer};
            color: {Palette.TEXT_PRIMARY};
            border-color: {Palette.BORDER_LIGHT};
        }}
        QPushButton#secondaryCompact {{
            padding: 6px 10px;
            min-height: 26px;
            font-size: 11px;
        }}
        QPushButton#danger {{
            color: {Palette.ACCENT_RED};
        }}
        QScrollBar:vertical, QScrollBar:horizontal {{
            background: transparent;
            border: none;
        }}
        QScrollBar:vertical {{
            width: 10px;
        }}
        QScrollBar:horizontal {{
            height: 10px;
        }}
        QScrollBar::handle {{
            background: {_hex_to_rgba(Palette.ACCENT, 0.35)};
            border-radius: 5px;
            min-height: 28px;
            min-width: 28px;
        }}
        QScrollBar::handle:hover {{
            background: {_hex_to_rgba(Palette.ACCENT_HOVER, 0.55)};
        }}
        QScrollBar::add-line, QScrollBar::sub-line,
        QScrollBar::add-page, QScrollBar::sub-page {{
            background: transparent;
            border: none;
            width: 0;
            height: 0;
        }}
    """
