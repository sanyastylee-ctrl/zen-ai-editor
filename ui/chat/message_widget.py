"""
MessageWidget — карточка одного сообщения.

Layout:
[avatar]  [header (sender + time)]
          [контент: TextBlock / CodeBlock / ToolBlock / ...]
"""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PyQt6.QtWidgets import (
    QFrame, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
)

from .styles import Palette, ui_font, Spacing
from .markdown_parser import parse, inline_to_html
from .code_block import CodeBlockWidget
from .tool_block import ToolBlockWidget


# ============================================================
# AVATAR
# ============================================================

class _Avatar(QWidget):
    """Круглый аватар с буквой/символом. Цвет от роли/профиля."""

    def __init__(self, role: str, profile_kind=None, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(Spacing.AVATAR_SIZE, Spacing.AVATAR_SIZE)
        self._role = role
        self._profile_kind = profile_kind

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # цвет фона аватара
        bg = self._bg_color()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(bg)))
        p.drawEllipse(0, 0, self.width(), self.height())

        # символ
        symbol = self._symbol()
        p.setPen(QPen(QColor(Palette.TEXT_INVERTED if self._is_light(bg) else "#FFFFFF")))
        font = ui_font(13, weight=QFont.Weight.DemiBold)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
        p.end()

    def _symbol(self) -> str:
        if self._role == "user":
            return "Я"
        if self._role == "tool":
            return "⚙"
        if self._role == "system":
            return "ⓘ"
        # assistant
        kind_str = self._profile_kind.value if hasattr(self._profile_kind, "value") else str(self._profile_kind or "")
        return {
            "coder": "⌨",
            "companion": "♡",
            "vision": "◉",
        }.get(kind_str, "A")

    def _bg_color(self) -> str:
        if self._role == "user":
            return Palette.ACCENT
        if self._role == "tool":
            return Palette.TEXT_DIM
        if self._role == "system":
            return Palette.BORDER_LIGHT
        kind_str = self._profile_kind.value if hasattr(self._profile_kind, "value") else str(self._profile_kind or "")
        return {
            "coder": Palette.ACCENT_BLUE,
            "companion": Palette.ACCENT_PINK,
            "vision": Palette.ACCENT_GREEN,
        }.get(kind_str, Palette.ACCENT)

    @staticmethod
    def _is_light(hex_color: str) -> bool:
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return False
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (0.299 * r + 0.587 * g + 0.114 * b) > 160


# ============================================================
# MESSAGE WIDGET
# ============================================================

class MessageWidget(QFrame):
    """
    Один сообщений-блок. Контент строится из record-словаря:
      record = {
        "role": "user" | "assistant" | "tool" | "system",
        "sender": str,
        "text": str,             # markdown (для user/assistant/system)
        "time": str,             # ISO
        "streaming": bool,
        "profile_kind": ProfileKind,
        # tool-only:
        "tool_name": str, "detail": str, "output": str, "ok": bool|None,
      }
    """

    insert_requested = pyqtSignal(str)

    def __init__(self, record: dict, parent=None) -> None:
        super().__init__(parent)
        self._record = record
        self._content_widgets: list[QWidget] = []  # для обновления при стриме

        self.setObjectName("msg_root")
        self.setStyleSheet("QFrame#msg_root { background: transparent; }")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(Spacing.AVATAR_GAP)

        # для user — выравниваем вправо: spacer слева
        is_user = record.get("role") == "user"
        if is_user:
            outer.addStretch(1)

        # ===== Аватар =====
        self._avatar = _Avatar(record.get("role", "assistant"), record.get("profile_kind"))
        if is_user:
            # для юзера аватар справа
            avatar_holder = QVBoxLayout()
            avatar_holder.addWidget(self._avatar)
            avatar_holder.addStretch()
        else:
            avatar_holder = QVBoxLayout()
            avatar_holder.addWidget(self._avatar)
            avatar_holder.addStretch()

        # ===== Карточка =====
        self._card = QFrame()
        self._card.setObjectName("msg_card")
        self._card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._apply_card_style()

        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(
            Spacing.CARD_PADDING, Spacing.CARD_PADDING - 4,
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
        )
        card_layout.setSpacing(8)
        self._card_layout = card_layout

        # ===== Header (имя + время) =====
        header = QHBoxLayout()
        header.setSpacing(8)
        sender_label = QLabel(record.get("sender", ""))
        sender_label.setStyleSheet(
            f"color:{Palette.TEXT_PRIMARY}; font-size:12px; font-weight:600;"
        )
        header.addWidget(sender_label)

        time_str = self._format_time(record.get("time"))
        if time_str:
            time_label = QLabel(time_str)
            time_label.setStyleSheet(f"color:{Palette.TEXT_DIM}; font-size:11px;")
            header.addWidget(time_label)
        header.addStretch()
        card_layout.addLayout(header)

        # ===== Контент =====
        self._content_holder = QVBoxLayout()
        self._content_holder.setSpacing(8)
        self._content_holder.setContentsMargins(0, 0, 0, 0)
        card_layout.addLayout(self._content_holder)

        # сборка контента
        self._rebuild_content()

        # компоновка
        if is_user:
            outer.addLayout(avatar_holder)
            outer.addWidget(self._card, 4)  # карточка
        else:
            outer.addLayout(avatar_holder)
            outer.addWidget(self._card, 1)
            # справа stretch чтобы карточка не растягивалась на весь экран? нет, наоборот пусть растёт
            # но ограничим левым отступом для tool/system
            outer.addStretch(0)

        # лимит ширины
        self.setMaximumWidth(16777215)  # без жёсткого лимита, родитель сам ограничит

    # ============================================================
    # API
    # ============================================================

    def update_record(self, record: dict) -> None:
        """Обновляет содержимое (для стрима или финального состояния tool)."""
        prev_text = self._record.get("text", "")
        prev_role = self._record.get("role")
        self._record = record

        if prev_role == "tool" or record.get("role") == "tool":
            self._rebuild_content()
            return

        # для assistant/user — пересобираем только если текст изменился
        if record.get("text", "") != prev_text or record.get("streaming"):
            self._rebuild_content()

    # ============================================================
    # внутренности
    # ============================================================

    def _apply_card_style(self) -> None:
        role = self._record.get("role", "assistant")
        bg = {
            "user": Palette.BG_USER,
            "assistant": Palette.BG_ASSISTANT,
            "system": Palette.BG_SYSTEM,
            "tool": "transparent",   # для tool своя обёртка
        }.get(role, Palette.BG_ASSISTANT)
        border = Palette.BORDER if role != "tool" else "transparent"
        radius = Spacing.CARD_RADIUS
        self._card.setStyleSheet(f"""
            QFrame#msg_card {{
                background: {bg};
                border: 1px solid {border};
                border-radius: {radius}px;
            }}
        """)

    def _clear_content(self) -> None:
        for w in self._content_widgets:
            w.deleteLater()
        self._content_widgets.clear()
        # удаляем все элементы из layout
        while self._content_holder.count():
            item = self._content_holder.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _rebuild_content(self) -> None:
        self._clear_content()
        role = self._record.get("role")

        if role == "tool":
            self._build_tool_content()
        else:
            self._build_text_content()

    def _build_tool_content(self) -> None:
        tool = ToolBlockWidget(
            tool_name=self._record.get("tool_name", ""),
            detail=self._record.get("detail", "") or "",
            output=self._record.get("output", "") or "",
            ok=self._record.get("ok"),
        )
        self._content_holder.addWidget(tool)
        self._content_widgets.append(tool)

    def _build_text_content(self) -> None:
        text = self._record.get("text", "") or ""
        streaming = self._record.get("streaming", False)
        blocks = parse(text)
        if not blocks and streaming:
            # пока пусто, но стрим начался — показываем мигающий курсор
            spinner = QLabel("▍")
            spinner.setStyleSheet(f"color:{Palette.ACCENT}; font-size:14px;")
            self._content_holder.addWidget(spinner)
            self._content_widgets.append(spinner)
            return

        for idx, blk in enumerate(blocks):
            w = self._render_block(blk, is_last=(idx == len(blocks) - 1), streaming=streaming)
            if w is not None:
                self._content_holder.addWidget(w)
                self._content_widgets.append(w)

    def _render_block(self, blk: dict, is_last: bool, streaming: bool):
        btype = blk["type"]

        if btype == "code":
            code = blk.get("content", "")
            # дописываем мигающий курсор только если стрим идёт И код не закрыт
            if streaming and is_last and not blk.get("closed", False):
                code = code + " ▍"
            lang = blk.get("lang", "")
            cb = CodeBlockWidget(code=code, lang=lang)
            cb.insert_requested.connect(self.insert_requested.emit)
            return cb

        if btype == "heading":
            level = blk["level"]
            sizes = {1: 17, 2: 15, 3: 14, 4: 13, 5: 13, 6: 12}
            size = sizes.get(level, 13)
            html = inline_to_html(blk["content"], Palette)
            label = QLabel(html)
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setStyleSheet(
                f"color:{Palette.TEXT_PRIMARY}; font-size:{size}px;"
                f"font-weight:700; padding-top:4px;"
            )
            return label

        if btype == "hr":
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setStyleSheet(f"background:{Palette.BORDER}; max-height:1px;")
            return line

        if btype == "quote":
            html = inline_to_html(blk["content"], Palette)
            label = QLabel(html)
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setStyleSheet(
                f"color:{Palette.TEXT_SECONDARY};"
                f"border-left:3px solid {Palette.ACCENT};"
                f"padding:2px 0 2px 10px;"
                f"background: rgba(167,139,250,0.04);"
                f"border-top-right-radius:4px; border-bottom-right-radius:4px;"
            )
            return label

        if btype == "list":
            items = blk.get("items", [])
            ordered = blk.get("ordered", False)
            lines = []
            for i, it in enumerate(items, 1):
                bullet = f"{i}." if ordered else "•"
                content_html = inline_to_html(it, Palette)
                lines.append(
                    f"<div style='padding:2px 0;'>"
                    f"<span style='color:{Palette.TEXT_DIM}; "
                    f"display:inline-block; min-width:24px;'>{bullet}</span>"
                    f"<span style='color:{Palette.TEXT_PRIMARY};'>{content_html}</span>"
                    f"</div>"
                )
            label = QLabel("".join(lines))
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            return label

        # text
        text_content = blk.get("content", "")
        if streaming and is_last:
            text_content = text_content + " ▍"
        html = inline_to_html(text_content, Palette)
        label = QLabel(html)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        label.setOpenExternalLinks(True)
        label.setStyleSheet(
            f"color:{Palette.TEXT_PRIMARY}; font-size:13px;"
            f"line-height:1.5;"
        )
        return label

    @staticmethod
    def _format_time(iso: str | None) -> str:
        if not iso:
            return ""
        try:
            dt = datetime.fromisoformat(iso)
            return dt.strftime("%H:%M")
        except Exception:
            return iso[:5] if len(iso) >= 5 else ""
