"""
ToolBlockWidget — карточка действий агента (write/run/read/...).

Цвет зависит от статуса: ok / err / running.
Output в свёрнутом виде, разворачивается по клику.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
)

from .styles import Palette, mono_font, Spacing


class ToolBlockWidget(QFrame):
    """Карточка одного tool-call'а."""

    def __init__(self, tool_name: str, detail: str, output: str, ok=None, parent=None) -> None:
        super().__init__(parent)
        self._collapsed = True
        self.setObjectName("tool_card")
        self._apply_status_style(ok)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 12)
        outer.setSpacing(6)

        # ===== Header =====
        head = QHBoxLayout()
        head.setSpacing(8)
        head.setContentsMargins(0, 0, 0, 0)

        # статус-индикатор кружком
        self._dot = QLabel()
        self._dot.setFixedSize(8, 8)
        self._update_dot(ok)
        head.addWidget(self._dot)

        self._name_label = QLabel(tool_name)
        self._name_label.setStyleSheet(
            f"color:{Palette.TEXT_PRIMARY}; font-size:12px; font-weight:600;"
        )
        head.addWidget(self._name_label)
        head.addStretch()

        # toggle для разворачивания output'а
        self._toggle_btn = QPushButton("▾ показать")
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Palette.TEXT_SECONDARY};"
            f" border: none; font-size: 11px; padding: 2px 6px; }}"
            f"QPushButton:hover {{ color: {Palette.TEXT_PRIMARY}; }}"
        )
        self._toggle_btn.clicked.connect(self._toggle_output)
        head.addWidget(self._toggle_btn)

        outer.addLayout(head)

        # ===== Detail =====
        self._detail_label = QLabel()
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet(
            f"color:{Palette.TEXT_SECONDARY}; font-size:11px;"
            "font-family: Consolas, Menlo, monospace;"
        )
        self._detail_label.setVisible(False)
        outer.addWidget(self._detail_label)

        # ===== Output =====
        self._output_view = QPlainTextEdit()
        self._output_view.setReadOnly(True)
        self._output_view.setFont(mono_font(10))
        self._output_view.setStyleSheet(
            f"QPlainTextEdit {{ background: rgba(0,0,0,0.25);"
            f" color: {Palette.TEXT_SECONDARY}; border: none;"
            f" border-radius: 6px; padding: 8px 10px; }}"
        )
        self._output_view.setMaximumHeight(180)
        self._output_view.setVisible(False)
        outer.addWidget(self._output_view)

        self.update_state(tool_name=tool_name, detail=detail, output=output, ok=ok)

    # ---------- API ----------

    def update_state(self, tool_name: str, detail: str, output: str, ok) -> None:
        self._name_label.setText(tool_name)
        if detail:
            self._detail_label.setText(detail)
            self._detail_label.setVisible(True)
        else:
            self._detail_label.setVisible(False)
        self._output_view.setPlainText(output or "")
        # если output короткий — показываем сразу, не сворачиваем
        short = output and (len(output) < 200 and output.count("\n") < 4)
        if short:
            self._output_view.setVisible(True)
            self._toggle_btn.setVisible(False)
            self._collapsed = False
        else:
            self._toggle_btn.setVisible(bool(output))
        self._update_dot(ok)
        self._apply_status_style(ok)

    # ---------- внутренности ----------

    def _toggle_output(self) -> None:
        self._collapsed = not self._collapsed
        self._output_view.setVisible(not self._collapsed)
        self._toggle_btn.setText("▴ скрыть" if not self._collapsed else "▾ показать")

    def _update_dot(self, ok) -> None:
        if ok is True:
            color = Palette.ACCENT_GREEN
        elif ok is False:
            color = Palette.ACCENT_RED
        else:
            color = Palette.ACCENT_AMBER
        self._dot.setStyleSheet(
            f"background: {color}; border-radius: 4px;"
        )

    def _apply_status_style(self, ok) -> None:
        if ok is True:
            bg = Palette.BG_TOOL_OK
            border = Palette.ACCENT_GREEN + "55"
        elif ok is False:
            bg = Palette.BG_TOOL_ERR
            border = Palette.ACCENT_RED + "55"
        else:
            bg = Palette.BG_TOOL_RUN
            border = Palette.BORDER_LIGHT
        self.setStyleSheet(f"""
            QFrame#tool_card {{
                background: {bg};
                border: 1px solid {border};
                border-radius: {Spacing.CARD_RADIUS - 2}px;
            }}
        """)
