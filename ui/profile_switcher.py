"""
Сегментированный переключатель активного профиля.

Выглядит как [ ⌨ Кодер | ♡ Алиса ] — одна активная кнопка подсвечена.
Эмитит signal profile_changed(profile_id: str) при клике.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QPushButton

from core.profiles import AIProfile, ProfileKind


class ProfileSwitcher(QFrame):
    profile_changed = pyqtSignal(str)  # profile_id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("profile_switcher_frame")
        self._buttons: dict[str, QPushButton] = {}
        self._active_id: str | None = None

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(0)

        self.setStyleSheet("""
            QFrame#profile_switcher_frame {
                background: #2D2D2D;
                border: 1px solid #3A3A3A;
                border-radius: 6px;
            }
            QPushButton {
                background: transparent;
                color: #888888;
                border: none;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: 500;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #3A3A3A;
                color: #D4D4D4;
            }
            QPushButton[active="true"] {
                background: #0E639C;
                color: #FFFFFF;
            }
        """)

    def set_profiles(self, profiles: list[AIProfile], active_id: str | None) -> None:
        """Пересоздаёт кнопки. Вызывается при загрузке и после изменений в настройках."""
        # очистка
        for btn in self._buttons.values():
            self._layout.removeWidget(btn)
            btn.deleteLater()
        self._buttons.clear()

        for profile in profiles:
            icon = self._kind_icon(profile.kind)
            btn = QPushButton(f"{icon}  {profile.name}")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, pid=profile.id: self._on_click(pid))
            self._buttons[profile.id] = btn
            self._layout.addWidget(btn)

        self._active_id = active_id
        self._refresh_active_state()

    def set_active(self, profile_id: str) -> None:
        if profile_id in self._buttons and profile_id != self._active_id:
            self._active_id = profile_id
            self._refresh_active_state()

    def active_id(self) -> str | None:
        return self._active_id

    # ---------- internals ----------

    def _on_click(self, profile_id: str) -> None:
        if profile_id == self._active_id:
            return
        self._active_id = profile_id
        self._refresh_active_state()
        self.profile_changed.emit(profile_id)

    def _refresh_active_state(self) -> None:
        for pid, btn in self._buttons.items():
            is_active = pid == self._active_id
            btn.setProperty("active", "true" if is_active else "false")
            # перерисовка
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    @staticmethod
    def _kind_icon(kind: ProfileKind) -> str:
        return {
            ProfileKind.CODER: "⌨",
            ProfileKind.COMPANION: "♡",
            ProfileKind.VISION: "👁",
            ProfileKind.GENERIC: "○",
        }.get(kind, "○")
