"""
Редактор персоны для companion-профиля.

Отдельный виджет, чтобы поля персонажа не смешивались с техническими
настройками модели. Используется как одна из вкладок в ProfileEditor.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QTextEdit,
    QLabel, QScrollArea, QGroupBox, QComboBox,
)


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


class PersonaEditor(QWidget):
    """Редактор персоны. get_persona() возвращает dict со всеми полями."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._fields: dict[str, QLineEdit | QTextEdit | QComboBox] = {}

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
        hint.setStyleSheet("color: #888; font-size: 11px; padding: 4px 2px;")
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
        mood_hint.setStyleSheet("color: #666; font-size: 11px;")
        mood_form.addRow("", mood_hint)
        layout.addWidget(mood_box)

        layout.addStretch()

        self.setStyleSheet("""
            QGroupBox {
                color: #D4D4D4;
                font-size: 12px;
                font-weight: bold;
                border: 1px solid #3A3A3A;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                background: #252526;
            }
            QLabel { color: #B0B0B0; font-size: 12px; }
            QLineEdit, QTextEdit, QComboBox {
                background: #1E1E1E;
                color: #E0E0E0;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 12px;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 1px solid #0E639C;
            }
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
                widget.setCurrentText(value or MOOD_PRESETS[0])

    def get_persona(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, widget in self._fields.items():
            if isinstance(widget, QLineEdit):
                out[key] = widget.text().strip()
            elif isinstance(widget, QTextEdit):
                out[key] = widget.toPlainText().strip()
            elif isinstance(widget, QComboBox):
                out[key] = widget.currentText().strip()
        return out
