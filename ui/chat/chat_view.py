"""
ChatView — главный виджет чата.

Внутри: QScrollArea с QVBoxLayout, в который складываются MessageWidget'ы.
Авто-скролл к низу при добавлении сообщения. Управление через record-API:
  - set_records(list)
  - add_record(record)
  - update_record(record)

Сигнал insert_requested(code) — для кнопки "↙ В редактор" в код-блоках.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFrame,
)

from .styles import Palette, Spacing
from .message_widget import MessageWidget


class ChatView(QScrollArea):
    insert_requested = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(f"""
            QScrollArea {{ background: {Palette.BG_CHAT}; border: none; }}
            QWidget#chat_inner {{ background: {Palette.BG_CHAT}; }}
            QScrollBar:vertical {{
                background: transparent; width: 10px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {Palette.BORDER_LIGHT};
                border-radius: 5px; min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Palette.TEXT_DIM};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0; background: transparent;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
        """)

        self._inner = QWidget()
        self._inner.setObjectName("chat_inner")
        self.setWidget(self._inner)

        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(20, 16, 20, 16)
        self._layout.setSpacing(Spacing.MESSAGE_GAP)
        self._layout.addStretch(1)  # подушка снизу, добавляем сообщения перед ней

        # карта record-id -> widget. Поскольку record-dict передаётся по ссылке,
        # используем id(record) как ключ.
        self._widgets: dict[int, MessageWidget] = {}
        self._records_ref: list[dict] = []

        # авто-скролл flag
        self._stick_to_bottom = True
        self.verticalScrollBar().valueChanged.connect(self._on_scrollbar_changed)

    # ============================================================
    # PUBLIC API — совместим с тем что вызывает main_window
    # ============================================================

    def set_records(self, records: list[dict]) -> None:
        """Полная перерисовка под новый список (переключение профиля)."""
        # очистка
        for w in self._widgets.values():
            self._layout.removeWidget(w)
            w.deleteLater()
        self._widgets.clear()
        self._records_ref = records

        for rec in records:
            self._insert_widget(rec)
        self._scroll_to_bottom_soon()

    def add_record(self, record: dict) -> None:
        """Добавить новое сообщение в конец."""
        self._records_ref.append(record) if record not in self._records_ref else None
        self._insert_widget(record)
        if self._stick_to_bottom:
            self._scroll_to_bottom_soon()

    def update_record(self, record: dict) -> None:
        """Обновить существующее сообщение (стрим / финиш tool-а)."""
        w = self._widgets.get(id(record))
        if w is None:
            # запись новая — добавим (например, появилась в обход add_record)
            self._insert_widget(record)
            return
        w.update_record(record)
        if self._stick_to_bottom:
            self._scroll_to_bottom_soon()

    # ============================================================
    # внутренности
    # ============================================================

    def _insert_widget(self, record: dict) -> None:
        w = MessageWidget(record)
        w.insert_requested.connect(self.insert_requested.emit)
        # вставляем перед stretch
        insert_at = self._layout.count() - 1
        self._layout.insertWidget(insert_at, w)
        self._widgets[id(record)] = w

    def _scroll_to_bottom_soon(self) -> None:
        """Скролл к низу после того, как layout пересчитается."""
        QTimer.singleShot(0, self._scroll_to_bottom_now)

    def _scroll_to_bottom_now(self) -> None:
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_scrollbar_changed(self, value: int) -> None:
        sb = self.verticalScrollBar()
        # если юзер прокрутил вверх — отключаем авто-скролл; если у дна — включаем
        at_bottom = value >= sb.maximum() - 20
        self._stick_to_bottom = at_bottom
