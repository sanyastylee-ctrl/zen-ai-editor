"""
Сегментированный переключатель активного профиля.

Выглядит как [ ⌨ Кодер | ♡ Алиса ] — одна активная кнопка подсвечена.
Эмитит signal profile_changed(profile_id: str) при клике.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QPushButton

from core.profiles import AIProfile, ProfileKind
from ui.chat.styles import Palette


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

        self.setStyleSheet(f"""
            QFrame#profile_switcher_frame {{
                background: {Palette.BG_ASSISTANT};
                border: 1px solid {Palette.BORDER};
                border-radius: 8px;
            }}
            QPushButton {{
                background: transparent;
                color: {Palette.TEXT_SECONDARY};
                border: none;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: 600;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background: rgba(167,139,250,0.06);
                color: {Palette.TEXT_PRIMARY};
            }}
            QPushButton[active="true"] {{
                background: {Palette.ACCENT};
                color: #FFFFFF;
            }}
        """)

    def set_profiles(self, profiles: list[AIProfile], active_id: str | None) -> None:
        """Пересоздаёт кнопки. Вызывается при загрузке и после изменений в настройках."""
        # очистка
        for btn in self._buttons.values():
            self._layout.removeWidget(btn)
            btn.deleteLater()
        self._buttons.clear()

        visible_kinds = {ProfileKind.CODER, ProfileKind.COMPANION, ProfileKind.RESEARCHER}
        visible_profiles = [p for p in profiles if p.kind in visible_kinds]
        if active_id not in {p.id for p in visible_profiles}:
            active_id = visible_profiles[0].id if visible_profiles else None

        for profile in visible_profiles:
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
            ProfileKind.RESEARCHER: "⌕",
            ProfileKind.VISION: "◉",
            ProfileKind.GENERIC: "○",
        }.get(kind, "○")
