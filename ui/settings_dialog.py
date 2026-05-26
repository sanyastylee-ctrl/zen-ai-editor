"""
Главный диалог настроек.

Структура:
┌─────────────────────────────────────────┐
│ [⌨ Кодер] [♡ Алиса] [+ Новый] [🗑]      │  ← переключатель профилей
├─────────────────────────────────────────┤
│ < ProfileEditor для активного >         │
│   Модель / Промпт / Параметры [/ Персона]│
├─────────────────────────────────────────┤
│ < Общие настройки (RAG, тема, кэш) >    │  ← внизу отдельная секция
├─────────────────────────────────────────┤
│                    [Отмена] [Сохранить] │
└─────────────────────────────────────────┘
"""

from __future__ import annotations

import uuid

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QLabel,
    QStackedWidget, QInputDialog, QMessageBox, QComboBox, QCheckBox,
    QSpinBox, QFormLayout, QGroupBox, QWidget,
)

from core.profiles import (
    AIProfile, ProfileKind, ProfileManager, ChatTemplate,
    DEFAULT_CODER_PROMPT, DEFAULT_COMPANION_PROMPT,
)
from .profile_editor import ProfileEditor


class SettingsDialog(QDialog):
    """
    Возвращает True если что-то изменилось и применилось.
    """

    def __init__(self, profile_manager: ProfileManager, app_settings: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.resize(820, 640)
        self.setMinimumSize(720, 540)

        self.pm = profile_manager
        self.app_settings = dict(app_settings)  # копия, применяется на Save

        self._editors: dict[str, ProfileEditor] = {}

        self._build()
        self._load_profiles()

    # ---------- build ----------

    def _build(self) -> None:
        self.setStyleSheet(self._stylesheet())

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ===== ВЕРХНЯЯ ПАНЕЛЬ: переключатель профилей =====
        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(6)

        profile_label = QLabel("Профиль:")
        profile_label.setStyleSheet("color:#B0B0B0; font-size:12px;")
        header_layout.addWidget(profile_label)

        self.profile_buttons_container = QHBoxLayout()
        self.profile_buttons_container.setSpacing(4)
        header_layout.addLayout(self.profile_buttons_container)
        header_layout.addStretch()

        self.new_btn = QPushButton("+ Новый")
        self.new_btn.setObjectName("secondary")
        self.new_btn.clicked.connect(self._on_new_profile)
        header_layout.addWidget(self.new_btn)

        self.delete_btn = QPushButton("🗑 Удалить")
        self.delete_btn.setObjectName("secondary")
        self.delete_btn.clicked.connect(self._on_delete_profile)
        header_layout.addWidget(self.delete_btn)

        outer.addWidget(header)

        # ===== КОНТЕНТ: stacked editors =====
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        # ===== ОБЩИЕ НАСТРОЙКИ =====
        general_frame = self._build_general_panel()
        outer.addWidget(general_frame)

        # ===== КНОПКИ =====
        btns = QHBoxLayout()
        btns.setContentsMargins(14, 10, 14, 12)
        btns.addStretch()

        cancel_btn = QPushButton("Отмена")
        cancel_btn.setObjectName("secondary")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)

        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self._on_save)
        btns.addWidget(save_btn)
        outer.addLayout(btns)

    def _build_general_panel(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("general_section")
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(4)

        title = QLabel("Общие настройки")
        title.setStyleSheet("color:#888; font-size:11px; font-weight:bold; padding:2px 0;")
        layout.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(20)

        # RAG
        self.rag_check = QCheckBox("Использовать RAG для кодера")
        self.rag_check.setChecked(self.app_settings.get("use_rag", False))
        row.addWidget(self.rag_check)

        # Diff
        self.diff_check = QCheckBox("Показывать diff перед вставкой кода")
        self.diff_check.setChecked(self.app_settings.get("diff_before_apply", True))
        row.addWidget(self.diff_check)

        row.addStretch()
        layout.addLayout(row)

        return wrap

    # ---------- profiles ----------

    def _load_profiles(self) -> None:
        # очистка
        while self.profile_buttons_container.count():
            item = self.profile_buttons_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        while self.stack.count():
            w = self.stack.widget(0)
            self.stack.removeWidget(w)
            w.deleteLater()
        self._editors.clear()

        # перестроить
        for i, profile in enumerate(self.pm.all()):
            btn = QPushButton(self._profile_button_label(profile))
            btn.setCheckable(True)
            btn.setObjectName("profile_tab")
            btn.clicked.connect(lambda _, pid=profile.id: self._select_profile(pid))
            self.profile_buttons_container.addWidget(btn)

            editor = ProfileEditor(profile)
            self.stack.addWidget(editor)
            self._editors[profile.id] = editor

        # выбираем первый
        first = self.pm.all()[0] if self.pm.all() else None
        if first:
            self._select_profile(first.id)

    def _select_profile(self, profile_id: str) -> None:
        if profile_id not in self._editors:
            return
        # переключить кнопки
        idx = 0
        target_idx = 0
        for i in range(self.profile_buttons_container.count()):
            w = self.profile_buttons_container.itemAt(i).widget()
            if isinstance(w, QPushButton):
                is_active = (list(self._editors.keys())[idx] == profile_id)
                w.setChecked(is_active)
                if is_active:
                    target_idx = idx
                idx += 1
        # переключить стек
        self.stack.setCurrentIndex(target_idx)

    def _profile_button_label(self, profile: AIProfile) -> str:
        icon = {
            ProfileKind.CODER: "⌨",
            ProfileKind.COMPANION: "♡",
            ProfileKind.GENERIC: "○",
        }.get(profile.kind, "○")
        return f"{icon}  {profile.name}"

    def _current_profile_id(self) -> str | None:
        idx = self.stack.currentIndex()
        keys = list(self._editors.keys())
        if 0 <= idx < len(keys):
            return keys[idx]
        return None

    # ---------- create / delete ----------

    def _on_new_profile(self) -> None:
        # выбор типа
        kinds = {
            "Компаньон (живой персонаж)": ProfileKind.COMPANION,
            "Кодер": ProfileKind.CODER,
            "Общий": ProfileKind.GENERIC,
        }
        kind_label, ok = QInputDialog.getItem(
            self, "Новый профиль", "Тип профиля:", list(kinds.keys()), 0, False
        )
        if not ok:
            return
        kind = kinds[kind_label]

        name, ok = QInputDialog.getText(self, "Новый профиль", "Имя:")
        if not ok or not name.strip():
            return

        # создаём
        defaults_by_kind = {
            ProfileKind.CODER: dict(
                system_prompt=DEFAULT_CODER_PROMPT,
                temperature=0.2, top_p=0.9, top_k=20,
                repeat_penalty=1.05, max_tokens=4096, n_ctx=8192,
            ),
            ProfileKind.COMPANION: dict(
                system_prompt=DEFAULT_COMPANION_PROMPT,
                temperature=0.85, top_p=0.95, top_k=50,
                repeat_penalty=1.15, max_tokens=1024, n_ctx=8192,
                persona={"character_name": name.strip(), "user_name": "", "current_mood": "спокойное"},
            ),
            ProfileKind.GENERIC: dict(
                system_prompt="You are a helpful assistant.",
                temperature=0.7, max_tokens=2048, n_ctx=8192,
            ),
        }

        new = AIProfile(
            id=str(uuid.uuid4()),
            name=name.strip(),
            kind=kind,
            chat_template=ChatTemplate.CHATML,
            **defaults_by_kind[kind],
        )

        self.pm.add(new)
        self._load_profiles()
        self._select_profile(new.id)

    def _on_delete_profile(self) -> None:
        pid = self._current_profile_id()
        if not pid:
            return
        profile = self.pm.get(pid)
        if not profile:
            return

        # не даём удалить последний профиль данного типа
        same_kind = self.pm.by_kind(profile.kind)
        if len(same_kind) <= 1:
            QMessageBox.warning(
                self, "Нельзя удалить",
                f"Это единственный профиль типа «{profile.kind.value}». Сначала создайте другой."
            )
            return

        confirm = QMessageBox.question(
            self, "Удалить профиль?",
            f"Удалить профиль «{profile.name}»?\nЭто действие нельзя отменить.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.pm.delete(pid)
            self._load_profiles()

    # ---------- save ----------

    def _on_save(self) -> None:
        # 1. сохранить все профили
        for pid, editor in self._editors.items():
            updated = editor.apply_to_profile()
            self.pm.update(updated)

        # 2. сохранить общие настройки
        self.app_settings["use_rag"] = self.rag_check.isChecked()
        self.app_settings["diff_before_apply"] = self.diff_check.isChecked()

        self.accept()

    def get_app_settings(self) -> dict:
        return self.app_settings

    # ---------- styles ----------

    @staticmethod
    def _stylesheet() -> str:
        return """
            QDialog { background-color: #1E1E1E; color: #D4D4D4; }
            QFrame#header { background: #252526; border-bottom: 1px solid #3A3A3A; }
            QFrame#general_section { background: #252526; border-top: 1px solid #3A3A3A; }
            QLabel { color: #D4D4D4; }
            QCheckBox { color: #D4D4D4; font-size: 12px; padding: 2px; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #555; border-radius: 3px;
                background: #1E1E1E;
            }
            QCheckBox::indicator:checked {
                background: #0E639C; border-color: #0E639C;
            }
            QPushButton {
                background-color: #0E639C; color: white;
                border-radius: 4px; padding: 7px 16px;
                font-size: 12px; font-weight: 500; border: none;
            }
            QPushButton:hover { background-color: #1177BB; }
            QPushButton#secondary {
                background-color: transparent; color: #B0B0B0;
                border: 1px solid #3A3A3A;
            }
            QPushButton#secondary:hover {
                background-color: #2A2A2A; color: #E0E0E0;
                border-color: #4A4A4A;
            }
            QPushButton#profile_tab {
                background: transparent;
                color: #888;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: 500;
            }
            QPushButton#profile_tab:hover {
                background: #2A2A2A; color: #D4D4D4;
            }
            QPushButton#profile_tab:checked {
                background: #0E639C; color: white;
            }
            QStackedWidget { background: #1E1E1E; }
        """
