"""
Редактор персоны для companion-профиля.

Отдельный виджет, чтобы поля персонажа не смешивались с техническими
настройками модели. Используется как одна из вкладок в ProfileEditor.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QTextEdit,
    QLabel, QScrollArea, QGroupBox, QComboBox, QSlider, QCheckBox,
    QPushButton, QHBoxLayout, QMessageBox,
)

from core.companion import CompanionMemoryStore
from ui.chat.styles import Palette, form_controls_qss


# Маппинг ключей персоны на человеческие подписи и тип поля
PERSONA_FIELDS = [
    # (key, label, kind, placeholder, multiline)
    ("character_name", "Имя",                  "text",    "Алиса",              False),
    ("age",            "Возраст",              "text",    "23",                 False),
    ("user_name",      "Как обращается к тебе","text",    "Саша",               False),
    ("relationship_to_user","Отношения с тобой","text",   "девушка, близкая подруга", False),

    ("appearance",     "Внешность",            "long",    "Светлые волосы, серые глаза, любит свитера…", True),
    ("personality",    "Характер",             "long",    "Тёплая, любопытная, с иронией…",            True),
    ("speaking_style", "Манера речи",          "long",    "Разговорная, без формальностей, иногда короткие реплики…", True),
    ("background",     "Биография",            "long",    "Студентка дизайна, живёт одна с котом…",    True),
    ("current_mood",   "Настроение сейчас",    "mood",    "",                                          False),
]


MOOD_PRESETS = [
    "спокойное",
    "игривое",
    "усталое",
    "влюблённое",
    "задумчивое",
    "раздражённое",
    "грустное",
    "энергичное",
    "флиртующее",
]

MODE_PRESETS = [
    ("Chat", "chat"),
    ("Support", "support"),
    ("Ideas", "ideas"),
    ("Project brainstorm", "project_brainstorm"),
    ("Roleplay", "roleplay"),
    ("Memory review/edit", "memory_review"),
]


TUNING_FIELDS = [
    ("tenderness", "Нежность", 7),
    ("playfulness", "Игривость", 6),
    ("initiative", "Инициативность", 5),
    ("romance", "Романтичность", 5),
    ("humor", "Юмор", 5),
    ("autonomy", "Самостоятельность", 5),
]


class PersonaEditor(QWidget):
    """Редактор персоны. get_persona() возвращает dict со всеми полями."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._fields: dict[str, QLineEdit | QTextEdit | QComboBox] = {}
        self._sliders: dict[str, QSlider] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        # подсказка сверху
        hint = QLabel(
            "Эти поля подставятся в системный промпт через {переменные}. "
            "Чем подробнее — тем устойчивее персонаж."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {Palette.TEXT_DIM}; font-size: 11px; padding: 4px 2px;")
        layout.addWidget(hint)

        # группа: личность
        identity_box = QGroupBox("Личность")
        identity_form = QFormLayout(identity_box)
        identity_form.setSpacing(8)
        for key, label, kind, placeholder, _ in PERSONA_FIELDS[:4]:
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            self._fields[key] = edit
            identity_form.addRow(label + ":", edit)
        layout.addWidget(identity_box)

        # группа: характер
        char_box = QGroupBox("Характер и история")
        char_form = QFormLayout(char_box)
        char_form.setSpacing(8)
        for key, label, kind, placeholder, _ in PERSONA_FIELDS[4:8]:
            edit = QTextEdit()
            edit.setPlaceholderText(placeholder)
            edit.setMinimumHeight(60)
            edit.setMaximumHeight(100)
            self._fields[key] = edit
            char_form.addRow(label + ":", edit)
        layout.addWidget(char_box)

        # группа: настроение
        mood_box = QGroupBox("Настроение")
        mood_form = QFormLayout(mood_box)
        mood_combo = QComboBox()
        mood_combo.setEditable(True)
        mood_combo.addItems(MOOD_PRESETS)
        self._fields["current_mood"] = mood_combo
        mood_form.addRow("Сейчас:", mood_combo)

        mood_hint = QLabel("Можно выбрать из пресетов или ввести своё. Меняй на лету в чате.")
        mood_hint.setStyleSheet(f"color: {Palette.TEXT_DIM}; font-size: 11px;")
        mood_form.addRow("", mood_hint)
        layout.addWidget(mood_box)

        mode_box = QGroupBox("Режим общения")
        mode_form = QFormLayout(mode_box)
        self.mode_combo = QComboBox()
        for label, value in MODE_PRESETS:
            self.mode_combo.addItem(label, value)
        self._fields["companion_mode"] = self.mode_combo
        mode_form.addRow("Режим:", self.mode_combo)
        layout.addWidget(mode_box)

        tuning_box = QGroupBox("Динамика характера")
        tuning_form = QFormLayout(tuning_box)
        for key, label, default in TUNING_FIELDS:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 10)
            slider.setValue(default)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(1)
            self._sliders[key] = slider
            tuning_form.addRow(f"{label}:", slider)
        layout.addWidget(tuning_box)

        prefs_box = QGroupBox("Предпочтения и границы")
        prefs_form = QFormLayout(prefs_box)
        for key, label, placeholder in (
            ("user_preferences", "Предпочтения", "Что пользователь любит, как с ним лучше говорить…"),
            ("boundaries", "Границы/taboo", "Темы, стиль или поведение, которых пользователь не хочет…"),
            ("project_interests", "Интересы в проектах", "ZenAI, игры, UI, локальные модели… по строке на пункт"),
        ):
            edit = QTextEdit()
            edit.setPlaceholderText(placeholder)
            edit.setMinimumHeight(54)
            edit.setMaximumHeight(90)
            self._fields[key] = edit
            prefs_form.addRow(label + ":", edit)
        layout.addWidget(prefs_box)

        memory_box = QGroupBox("Память")
        memory_layout = QVBoxLayout(memory_box)
        self.memory_check = QCheckBox("Запоминать только явные устойчивые факты")
        self._fields["memory_enabled"] = self.memory_check
        memory_layout.addWidget(self.memory_check)
        buttons = QHBoxLayout()
        review_btn = QPushButton("Что помнишь?")
        review_btn.setObjectName("secondaryCompact")
        review_btn.clicked.connect(self._show_memories)
        clear_btn = QPushButton("Забыть всё")
        clear_btn.setObjectName("secondaryCompact")
        clear_btn.clicked.connect(self._clear_memories)
        buttons.addWidget(review_btn)
        buttons.addWidget(clear_btn)
        buttons.addStretch()
        memory_layout.addLayout(buttons)
        memory_hint = QLabel("Автосохранение срабатывает на явные фразы вроде «запомни ...».")
        memory_hint.setWordWrap(True)
        memory_hint.setStyleSheet(f"color: {Palette.TEXT_DIM}; font-size: 11px;")
        memory_layout.addWidget(memory_hint)
        layout.addWidget(memory_box)

        layout.addStretch()

        self.setStyleSheet(form_controls_qss() + f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QGroupBox {{
                color: {Palette.TEXT_PRIMARY};
                font-size: 12px;
                font-weight: 600;
                border: 1px solid {Palette.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px 8px 8px 8px;
                background: {Palette.BG_ASSISTANT};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: {Palette.TEXT_SECONDARY};
                background: {Palette.BG_ASSISTANT};
            }}
        """)

    # ---------- I/O ----------

    def set_persona(self, persona: dict[str, str]) -> None:
        for key, widget in self._fields.items():
            value = persona.get(key, "")
            if isinstance(widget, QLineEdit):
                widget.setText(value)
            elif isinstance(widget, QTextEdit):
                widget.setPlainText(value)
            elif isinstance(widget, QComboBox):
                idx = widget.findData(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                else:
                    widget.setCurrentText(value or MOOD_PRESETS[0])
            elif isinstance(widget, QCheckBox):
                widget.setChecked(str(value or "true").lower() not in {"0", "false", "no", "off", "нет"})
        for key, slider in self._sliders.items():
            try:
                slider.setValue(int(persona.get(key, slider.value())))
            except (TypeError, ValueError):
                pass

    def get_persona(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, widget in self._fields.items():
            if isinstance(widget, QLineEdit):
                out[key] = widget.text().strip()
            elif isinstance(widget, QTextEdit):
                out[key] = widget.toPlainText().strip()
            elif isinstance(widget, QComboBox):
                out[key] = str(widget.currentData() or widget.currentText()).strip()
            elif isinstance(widget, QCheckBox):
                out[key] = "true" if widget.isChecked() else "false"
        for key, slider in self._sliders.items():
            out[key] = str(slider.value())
        return out

    def _show_memories(self) -> None:
        memories = CompanionMemoryStore().load()
        if not memories:
            text = "Пока ничего не сохранено."
        else:
            text = "\n".join(f"- {m.text} ({m.category}, {m.confidence:.1f})" for m in memories)
        QMessageBox.information(self, "Память Леры", text)

    def _clear_memories(self) -> None:
        if QMessageBox.question(
            self,
            "Очистить память?",
            "Удалить все сохранённые воспоминания Леры?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            CompanionMemoryStore().clear()
