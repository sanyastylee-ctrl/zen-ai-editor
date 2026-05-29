"""
Редактор одного AIProfile. Вкладки:
- Модель: файл .gguf, шаблон чата, n_ctx, n_gpu_layers
- Промпт: системный промпт (большое поле + кнопка превью + счётчик)
- Параметры: temperature, top_p, top_k, repeat_penalty, max_tokens, stop_sequences
- Персона (только для COMPANION): PersonaEditor
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget,
    QLineEdit, QTextEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QLabel, QFileDialog, QFrame, QCheckBox,
)

from core.profiles import (
    AIProfile, ProfileKind, ChatTemplate, DEFAULT_AGENT_CODER_PROMPT,
    DEFAULT_RESEARCHER_PROMPT,
)
from core.paths import list_available_models
from core.token_budget import TokenBudget
from ui.chat.styles import Palette, form_controls_qss, mono_font
from .persona_editor import PersonaEditor


# Готовые шаблоны системных промптов
PROMPT_PRESETS_CODER = {
    "По умолчанию (Qwen Coder)": "default",
    "Агент-кодер (tools)": DEFAULT_AGENT_CODER_PROMPT,
    "С RAG (учитывает релевантный код проекта)": """You are an expert software engineer.
Use the provided code context and the relevant project snippets to write code that fits the codebase.
Never invent APIs that aren't in the context.
Reply with code in markdown blocks. Brief explanations only.""",
    "Ревьюер (только анализ, без правок)": """You are a senior code reviewer. Analyse the given code for:
- bugs and edge cases
- performance issues
- readability and naming
- security concerns
Reply with a numbered list of findings. Don't rewrite the code unless asked.""",
}

PROMPT_PRESETS_COMPANION = {
    "По умолчанию (живой персонаж)": "default",
    "Близкая девушка (без цензуры)": None,  # = дефолтный, он уже про девушку
}

PROMPT_PRESETS_VISION = {
    "По умолчанию (Vision-ассистент)": "default",
    "Транскрипция текста с изображения": """You are an OCR assistant. Your only job is to extract text from the image exactly as it appears, preserving line breaks and structure. Do not interpret, summarize, or comment — only transcribe. Reply with the transcribed text in a code block.""",
    "Анализ скриншота с кодом": """You are a code review assistant looking at screenshots. For each image:
1. Identify the programming language.
2. Transcribe the visible code into a markdown code block.
3. Point out bugs, issues, or improvements you can see.
Reply in Russian.""",
    "Анализ UI / интерфейса": """You analyze UI screenshots. Describe:
- Layout structure (header, sidebar, main area)
- Interactive elements (buttons, inputs, menus)
- Visual issues (alignment, contrast, hierarchy)
- Suggested improvements
Be concrete, reference what you actually see. Reply in Russian.""",
}

PROMPT_PRESETS_RESEARCHER = {
    "По умолчанию (Поисковик)": "default",
    "Короткие ответы с источниками": """You are ZenAI Researcher. Answer briefly and clearly.
Use web search for fresh/current facts. Never invent sources. If search was used, cite sources.""",
    "Сравнение вариантов": """You are ZenAI Researcher. Compare options by criteria, tradeoffs, risks, and sources.
Use web search for current products, model versions, prices, and releases. Cite sources.""",
}


class ProfileEditor(QWidget):
    """Виджет редактора. Сам не показывает кнопки Save/Cancel — это делает диалог-обёртка."""

    changed = pyqtSignal()

    def __init__(self, profile: AIProfile, parent=None) -> None:
        super().__init__(parent)
        self._profile = profile
        self._build()
        self._load_from_profile()

    # ---------- public ----------

    def apply_to_profile(self) -> AIProfile:
        """Собирает значения из полей обратно в self._profile и возвращает его."""
        p = self._profile

        p.name = self.name_edit.text().strip() or "Профиль"
        p.model_file = self.model_combo.currentText()
        p.chat_template = ChatTemplate(self.template_combo.currentData())
        p.n_ctx = self.n_ctx_spin.value()
        p.n_gpu_layers = self.gpu_layers_spin.value()

        p.system_prompt = self.prompt_edit.toPlainText()
        if p.kind == ProfileKind.CODER and hasattr(self, "agent_mode_check"):
            p.agent_mode = self.agent_mode_check.isChecked()
        if p.kind == ProfileKind.CODER and hasattr(self, "vision_assist_check"):
            p.enable_vision_assist = self.vision_assist_check.isChecked()
            if hasattr(self, "vision_model_combo"):
                value = self.vision_model_combo.currentText()
                p.vision_model_file = value if value and not value.startswith("(") else ""
            if hasattr(self, "mmproj_combo"):
                value = self.mmproj_combo.currentText()
                p.mmproj_file = value if value and not value.startswith("(") else ""
            if hasattr(self, "vision_handler_combo"):
                p.vision_handler = self.vision_handler_combo.currentData() or ""
            if hasattr(self, "max_visual_context_spin"):
                p.max_visual_context_chars = self.max_visual_context_spin.value()
            if hasattr(self, "vision_policy_combo"):
                p.vision_first_policy = self.vision_policy_combo.currentData() or "auto"
        if p.kind == ProfileKind.RESEARCHER and hasattr(self, "search_enabled_check"):
            p.search_enabled = self.search_enabled_check.isChecked()
            p.max_search_results = self.max_search_results_spin.value()
            p.max_pages_to_read = self.max_pages_to_read_spin.value()
            p.require_sources_for_fresh_info = self.require_sources_check.isChecked()
            p.answer_style = self.answer_style_combo.currentData() or "detailed"

        p.temperature = self.temp_spin.value()
        p.top_p = self.top_p_spin.value()
        p.top_k = self.top_k_spin.value()
        p.repeat_penalty = self.rep_pen_spin.value()
        p.max_tokens = self.max_tokens_spin.value()
        stop_text = self.stop_edit.text().strip()
        p.stop_sequences = [s.strip() for s in stop_text.split(",") if s.strip()] if stop_text else []

        if p.kind == ProfileKind.COMPANION and self.persona_editor is not None:
            p.persona = self.persona_editor.get_persona()

        if p.kind == ProfileKind.VISION:
            if hasattr(self, "mmproj_combo"):
                value = self.mmproj_combo.currentText()
                # игнорируем плейсхолдеры типа "(нет mmproj в models/)"
                p.mmproj_file = value if value and not value.startswith("(") else ""
            if hasattr(self, "vision_handler_combo"):
                p.vision_handler = self.vision_handler_combo.currentData() or ""

        return p

    # ---------- build ----------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # шапка: имя профиля
        header = QHBoxLayout()
        header.setContentsMargins(4, 4, 4, 4)
        name_label = QLabel("Имя профиля:")
        name_label.setStyleSheet(f"color:{Palette.TEXT_SECONDARY}; font-size:12px;")
        header.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setMaximumWidth(280)
        header.addWidget(self.name_edit)
        header.addStretch()
        kind_label = QLabel(self._kind_caption())
        kind_label.setStyleSheet(f"color:{Palette.TEXT_DIM}; font-size:11px;")
        header.addWidget(kind_label)
        outer.addLayout(header)

        # вкладки
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        outer.addWidget(self.tabs)

        self.tabs.addTab(self._build_model_tab(), "Модель")
        self.tabs.addTab(self._build_prompt_tab(), "Промпт")
        self.tabs.addTab(self._build_params_tab(), "Параметры")

        self.persona_editor: PersonaEditor | None = None
        if self._profile.kind == ProfileKind.COMPANION:
            self.persona_editor = PersonaEditor()
            self.tabs.addTab(self.persona_editor, "Персона")

        # стили
        self.setStyleSheet(self._stylesheet())

    def _build_model_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(10)
        form.setContentsMargins(16, 16, 16, 16)

        # файл модели
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(280)
        self._refresh_models()
        model_row.addWidget(self.model_combo, 1)

        refresh_btn = QPushButton("Обновить")
        refresh_btn.setFixedWidth(82)
        refresh_btn.setObjectName("secondaryCompact")
        refresh_btn.setToolTip("Обновить список моделей из /models")
        refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(refresh_btn)

        browse_btn = QPushButton("Выбрать")
        browse_btn.setFixedWidth(78)
        browse_btn.setObjectName("secondaryCompact")
        browse_btn.setToolTip("Выбрать .gguf файл с диска")
        browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(browse_btn)
        form.addRow("Файл модели:", self._wrap_row(model_row))

        # шаблон чата
        self.template_combo = QComboBox()
        for t in ChatTemplate:
            label = self._template_label(t)
            self.template_combo.addItem(label, t.value)
        form.addRow("Шаблон чата:", self.template_combo)

        # n_ctx
        self.n_ctx_spin = QSpinBox()
        self.n_ctx_spin.setRange(512, 131072)
        self.n_ctx_spin.setSingleStep(1024)
        self.n_ctx_spin.setSuffix("  токенов")
        form.addRow("Размер контекста (n_ctx):", self.n_ctx_spin)

        # gpu_layers
        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(-1, 200)
        self.gpu_layers_spin.setSpecialValueText("все (-1)")
        form.addRow("Слоёв на GPU:", self.gpu_layers_spin)

        # подсказка
        hint = QLabel(
            "Положите .gguf файлы в папку <code>models/</code> в корне проекта. "
            "Для Qwen Coder/Hermes используется шаблон <b>ChatML</b>."
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setStyleSheet(f"color:{Palette.TEXT_DIM}; font-size:11px; padding:4px;")
        form.addRow("", hint)

        # === Vision-секция (только для Vision-профилей) ===
        if self._profile.kind == ProfileKind.CODER:
            sep_label = QLabel("Vision Assist / Глаза")
            sep_label.setStyleSheet(
                f"color:{Palette.ACCENT_GREEN}; font-size:11px; font-weight:bold; "
                f"padding:8px 0 4px 0; border-top:1px solid {Palette.BORDER}; margin-top:8px;"
            )
            form.addRow(sep_label)

            self.vision_assist_check = QCheckBox("Анализировать прикреплённые скриншоты перед Coder Agent")
            form.addRow("Vision Assist:", self.vision_assist_check)

            vision_model_row = QHBoxLayout()
            self.vision_model_combo = QComboBox()
            self.vision_model_combo.setMinimumWidth(280)
            self._refresh_vision_model_list()
            vision_model_row.addWidget(self.vision_model_combo, 1)

            vision_model_refresh = QPushButton("Обновить")
            vision_model_refresh.setFixedWidth(82)
            vision_model_refresh.setObjectName("secondaryCompact")
            vision_model_refresh.setToolTip("Обновить список vision-моделей")
            vision_model_refresh.clicked.connect(self._refresh_vision_model_list)
            vision_model_row.addWidget(vision_model_refresh)
            form.addRow("Vision модель:", self._wrap_row(vision_model_row))

            mmproj_row = QHBoxLayout()
            self.mmproj_combo = QComboBox()
            self.mmproj_combo.setMinimumWidth(280)
            self._refresh_mmproj_list()
            mmproj_row.addWidget(self.mmproj_combo, 1)

            mmproj_refresh = QPushButton("Обновить")
            mmproj_refresh.setFixedWidth(82)
            mmproj_refresh.setObjectName("secondaryCompact")
            mmproj_refresh.setToolTip("Обновить список mmproj файлов")
            mmproj_refresh.clicked.connect(self._refresh_mmproj_list)
            mmproj_row.addWidget(mmproj_refresh)
            form.addRow("mmproj файл:", self._wrap_row(mmproj_row))

            self.vision_handler_combo = QComboBox()
            self._fill_vision_handlers(self.vision_handler_combo)
            form.addRow("Vision handler:", self.vision_handler_combo)

            self.max_visual_context_spin = QSpinBox()
            self.max_visual_context_spin.setRange(1000, 12000)
            self.max_visual_context_spin.setSingleStep(500)
            self.max_visual_context_spin.setSuffix(" символов")
            form.addRow("Visual context:", self.max_visual_context_spin)

            self.vision_policy_combo = QComboBox()
            self.vision_policy_combo.addItem("Auto: только когда есть изображение", "auto")
            self.vision_policy_combo.addItem("Always: всегда сначала Vision при изображении", "always")
            self.vision_policy_combo.addItem("Never: не запускать Vision", "never")
            form.addRow("Vision first:", self.vision_policy_combo)

            vhint = QLabel(
                "Vision Assist не редактирует файлы и не запускает терминал. "
                "Он только создаёт compact visual_context, который затем получает Кодер."
            )
            vhint.setWordWrap(True)
            vhint.setTextFormat(Qt.TextFormat.RichText)
            vhint.setStyleSheet(f"color:{Palette.TEXT_DIM}; font-size:11px; padding:4px;")
            form.addRow("", vhint)

        if self._profile.kind == ProfileKind.RESEARCHER:
            sep_label = QLabel("Поисковик / Researcher")
            sep_label.setStyleSheet(
                f"color:{Palette.ACCENT_GREEN}; font-size:11px; font-weight:bold; "
                f"padding:8px 0 4px 0; border-top:1px solid {Palette.BORDER}; margin-top:8px;"
            )
            form.addRow(sep_label)

            self.search_enabled_check = QCheckBox("Разрешить web-search pipeline для свежих фактов")
            form.addRow("Интернет-поиск:", self.search_enabled_check)

            self.max_search_results_spin = QSpinBox()
            self.max_search_results_spin.setRange(1, 20)
            form.addRow("Результатов поиска:", self.max_search_results_spin)

            self.max_pages_to_read_spin = QSpinBox()
            self.max_pages_to_read_spin.setRange(0, 10)
            form.addRow("Страниц читать:", self.max_pages_to_read_spin)

            self.require_sources_check = QCheckBox("Требовать источники для актуальной информации")
            form.addRow("Fresh info:", self.require_sources_check)

            self.answer_style_combo = QComboBox()
            self.answer_style_combo.addItem("Коротко", "short")
            self.answer_style_combo.addItem("Подробно", "detailed")
            self.answer_style_combo.addItem("Сравнение", "compare")
            form.addRow("Стиль ответа:", self.answer_style_combo)

            rhint = QLabel(
                "Если backend поиска не настроен, Поисковик честно сообщит, что web search недоступен, "
                "и не будет придумывать источники."
            )
            rhint.setWordWrap(True)
            rhint.setTextFormat(Qt.TextFormat.RichText)
            rhint.setStyleSheet(f"color:{Palette.TEXT_DIM}; font-size:11px; padding:4px;")
            form.addRow("", rhint)

        if self._profile.kind == ProfileKind.VISION:
            # разделитель
            sep_label = QLabel("Vision (распознавание изображений)")
            sep_label.setStyleSheet(
                f"color:{Palette.ACCENT_GREEN}; font-size:11px; font-weight:bold; "
                f"padding:8px 0 4px 0; border-top:1px solid {Palette.BORDER}; margin-top:8px;"
            )
            form.addRow(sep_label)

            # mmproj файл
            mmproj_row = QHBoxLayout()
            self.mmproj_combo = QComboBox()
            self.mmproj_combo.setMinimumWidth(280)
            self._refresh_mmproj_list()
            mmproj_row.addWidget(self.mmproj_combo, 1)

            mmproj_refresh = QPushButton("Обновить")
            mmproj_refresh.setFixedWidth(82)
            mmproj_refresh.setObjectName("secondaryCompact")
            mmproj_refresh.setToolTip("Обновить список mmproj файлов")
            mmproj_refresh.clicked.connect(self._refresh_mmproj_list)
            mmproj_row.addWidget(mmproj_refresh)
            form.addRow("mmproj файл:", self._wrap_row(mmproj_row))

            # тип handler'а
            self.vision_handler_combo = QComboBox()
            self._fill_vision_handlers(self.vision_handler_combo)
            form.addRow("Vision handler:", self.vision_handler_combo)

            vhint = QLabel(
                "Для <b>Qwen2.5-VL</b>: выберите модель Qwen2.5-VL и её mmproj-f16.gguf, "
                "handler <b>Qwen 2.5 VL</b>. mmproj должен быть от <b>той же</b> модели."
            )
            vhint.setWordWrap(True)
            vhint.setTextFormat(Qt.TextFormat.RichText)
            vhint.setStyleSheet(f"color:{Palette.TEXT_DIM}; font-size:11px; padding:4px;")
            form.addRow("", vhint)

        return w

    def _build_prompt_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # пресеты
        preset_row = QHBoxLayout()
        preset_label = QLabel("Шаблон:")
        preset_label.setStyleSheet(f"color:{Palette.TEXT_SECONDARY}; font-size:12px;")
        preset_row.addWidget(preset_label)

        self.preset_combo = QComboBox()
        if self._profile.kind == ProfileKind.CODER:
            presets = PROMPT_PRESETS_CODER
        elif self._profile.kind == ProfileKind.VISION:
            presets = PROMPT_PRESETS_VISION
        elif self._profile.kind == ProfileKind.RESEARCHER:
            presets = PROMPT_PRESETS_RESEARCHER
        else:
            presets = PROMPT_PRESETS_COMPANION
        self.preset_combo.addItem("— не менять —", None)
        for name, value in presets.items():
            self.preset_combo.addItem(name, value)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        preset_row.addWidget(self.preset_combo, 1)
        layout.addLayout(preset_row)

        # подсказка про переменные
        if self._profile.kind == ProfileKind.COMPANION:
            vars_hint = QLabel(
                "<b>Переменные:</b> {character_name}, {age}, {user_name}, {personality}, "
                "{speaking_style}, {appearance}, {background}, {current_mood}, {relationship_to_user}. "
                "Дополнительные настройки CompanionState и памяти заполняются на вкладке <b>Персона</b>."
            )
            vars_hint.setWordWrap(True)
            vars_hint.setTextFormat(Qt.TextFormat.RichText)
            vars_hint.setStyleSheet(
                f"color:{Palette.TEXT_DIM}; font-size:11px; padding:8px;"
                f"background:{Palette.BG_CODE}; border:1px solid {Palette.BORDER}; border-radius:6px;"
            )
            layout.addWidget(vars_hint)

        # сам промпт
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setFont(mono_font(11))
        self.prompt_edit.setPlaceholderText("Системный промпт...")
        self.prompt_edit.textChanged.connect(self._update_prompt_counter)
        layout.addWidget(self.prompt_edit, 1)

        # счётчик токенов промпта
        counter_row = QHBoxLayout()
        self.prompt_counter = QLabel("0 токенов")
        self.prompt_counter.setStyleSheet(f"color:{Palette.TEXT_DIM}; font-size:11px;")
        counter_row.addWidget(self.prompt_counter)
        counter_row.addStretch()
        layout.addLayout(counter_row)

        return w

    def _build_params_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(10)
        form.setContentsMargins(16, 16, 16, 16)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setDecimals(2)
        form.addRow("Temperature:", self.temp_spin)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setDecimals(2)
        form.addRow("Top-p:", self.top_p_spin)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(0, 1000)
        form.addRow("Top-k:", self.top_k_spin)

        self.rep_pen_spin = QDoubleSpinBox()
        self.rep_pen_spin.setRange(0.0, 2.0)
        self.rep_pen_spin.setSingleStep(0.05)
        self.rep_pen_spin.setDecimals(2)
        form.addRow("Repeat penalty:", self.rep_pen_spin)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 16384)
        self.max_tokens_spin.setSingleStep(128)
        form.addRow("Макс. токенов ответа:", self.max_tokens_spin)

        self.stop_edit = QLineEdit()
        self.stop_edit.setPlaceholderText("Через запятую: <|im_end|>, </s>")
        form.addRow("Стоп-токены:", self.stop_edit)

        # подсказка
        if self._profile.kind == ProfileKind.CODER:
            self.agent_mode_check = QCheckBox("Agent mode: читать/писать файлы и запускать tools")
            self.agent_mode_check.setToolTip(
                "Кодер сможет вызывать XML-инструменты read_file/list_files/search_files "
                "и получать результаты обратно в контекст."
            )
            form.addRow("Агент:", self.agent_mode_check)

            hint_text = (
                "Для кода: <b>temperature 0.1–0.3</b>, top_p 0.9. "
                "Низкая температура = более точный детерминированный код."
            )
        else:
            hint_text = (
                "Для живого диалога: <b>temperature 0.8–1.0</b>, top_p 0.95, top_k 50. "
                "Repeat penalty 1.1–1.2 помогает избежать повторов."
            )
        hint = QLabel(hint_text)
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setStyleSheet(f"color:{Palette.TEXT_DIM}; font-size:11px; padding:6px;")
        form.addRow("", hint)

        return w

    # ---------- load/preset/refresh ----------

    def _load_from_profile(self) -> None:
        p = self._profile
        self.name_edit.setText(p.name)

        # модель — добавим её имя в комбо, если её нет в списке (вдруг файла нет на диске)
        if p.model_file and self.model_combo.findText(p.model_file) == -1:
            self.model_combo.addItem(p.model_file)
        self.model_combo.setCurrentText(p.model_file)

        # шаблон
        idx = self.template_combo.findData(p.chat_template.value)
        if idx >= 0:
            self.template_combo.setCurrentIndex(idx)

        self.n_ctx_spin.setValue(p.n_ctx)
        self.gpu_layers_spin.setValue(p.n_gpu_layers)

        self.prompt_edit.setPlainText(p.system_prompt)
        self._update_prompt_counter()

        self.temp_spin.setValue(p.temperature)
        self.top_p_spin.setValue(p.top_p)
        self.top_k_spin.setValue(p.top_k)
        self.rep_pen_spin.setValue(p.repeat_penalty)
        self.max_tokens_spin.setValue(p.max_tokens)
        self.stop_edit.setText(", ".join(p.stop_sequences))
        if p.kind == ProfileKind.CODER and hasattr(self, "agent_mode_check"):
            self.agent_mode_check.setChecked(getattr(p, "agent_mode", False))
        if p.kind == ProfileKind.CODER and hasattr(self, "vision_assist_check"):
            self.vision_assist_check.setChecked(getattr(p, "enable_vision_assist", False))
            if hasattr(self, "vision_model_combo"):
                value = getattr(p, "vision_model_file", "")
                if value and self.vision_model_combo.findText(value) == -1:
                    self.vision_model_combo.addItem(value)
                if value:
                    self.vision_model_combo.setCurrentText(value)
            if hasattr(self, "mmproj_combo"):
                value = getattr(p, "mmproj_file", "")
                if value and self.mmproj_combo.findText(value) == -1:
                    self.mmproj_combo.addItem(value)
                if value:
                    self.mmproj_combo.setCurrentText(value)
            if hasattr(self, "vision_handler_combo") and getattr(p, "vision_handler", ""):
                idx = self.vision_handler_combo.findData(p.vision_handler)
                if idx >= 0:
                    self.vision_handler_combo.setCurrentIndex(idx)
            if hasattr(self, "max_visual_context_spin"):
                self.max_visual_context_spin.setValue(getattr(p, "max_visual_context_chars", 4000))
            if hasattr(self, "vision_policy_combo"):
                idx = self.vision_policy_combo.findData(getattr(p, "vision_first_policy", "auto"))
                if idx >= 0:
                    self.vision_policy_combo.setCurrentIndex(idx)
        if p.kind == ProfileKind.RESEARCHER and hasattr(self, "search_enabled_check"):
            self.search_enabled_check.setChecked(getattr(p, "search_enabled", True))
            self.max_search_results_spin.setValue(getattr(p, "max_search_results", 5))
            self.max_pages_to_read_spin.setValue(getattr(p, "max_pages_to_read", 3))
            self.require_sources_check.setChecked(getattr(p, "require_sources_for_fresh_info", True))
            idx = self.answer_style_combo.findData(getattr(p, "answer_style", "detailed"))
            if idx >= 0:
                self.answer_style_combo.setCurrentIndex(idx)

        if self.persona_editor is not None:
            self.persona_editor.set_persona(p.persona)

        # Vision-поля
        if p.kind == ProfileKind.VISION and hasattr(self, "mmproj_combo"):
            if p.mmproj_file and self.mmproj_combo.findText(p.mmproj_file) == -1:
                self.mmproj_combo.addItem(p.mmproj_file)
            if p.mmproj_file:
                self.mmproj_combo.setCurrentText(p.mmproj_file)
            if hasattr(self, "vision_handler_combo") and p.vision_handler:
                idx = self.vision_handler_combo.findData(p.vision_handler)
                if idx >= 0:
                    self.vision_handler_combo.setCurrentIndex(idx)

    def _refresh_models(self) -> None:
        current = self.model_combo.currentText() if self.model_combo.count() else ""
        self.model_combo.clear()
        models = list_available_models()
        # отфильтруем mmproj-файлы — они не основные модели
        models = [m for m in models if "mmproj" not in m.lower()]
        if not models:
            self.model_combo.addItem("(нет .gguf в папке models/)")
            self.model_combo.setEnabled(False)
        else:
            self.model_combo.setEnabled(True)
            for m in models:
                self.model_combo.addItem(m)
            if current and current in models:
                self.model_combo.setCurrentText(current)

    def _refresh_mmproj_list(self) -> None:
        if not hasattr(self, "mmproj_combo"):
            return
        current = self.mmproj_combo.currentText() if self.mmproj_combo.count() else ""
        self.mmproj_combo.clear()
        all_models = list_available_models()
        # mmproj-файлы — те, где в имени есть mmproj
        mmprojs = [m for m in all_models if "mmproj" in m.lower()]
        if not mmprojs:
            self.mmproj_combo.addItem("(нет mmproj-*.gguf в models/)")
            self.mmproj_combo.setEnabled(False)
        else:
            self.mmproj_combo.setEnabled(True)
            for m in mmprojs:
                self.mmproj_combo.addItem(m)
            if current and current in mmprojs:
                self.mmproj_combo.setCurrentText(current)

    def _refresh_vision_model_list(self) -> None:
        if not hasattr(self, "vision_model_combo"):
            return
        current = self.vision_model_combo.currentText() if self.vision_model_combo.count() else ""
        self.vision_model_combo.clear()
        models = [m for m in list_available_models() if "mmproj" not in m.lower()]
        if not models:
            self.vision_model_combo.addItem("(нет vision .gguf в папке models/)")
            self.vision_model_combo.setEnabled(False)
        else:
            self.vision_model_combo.setEnabled(True)
            for m in models:
                self.vision_model_combo.addItem(m)
            if current and current in models:
                self.vision_model_combo.setCurrentText(current)

    @staticmethod
    def _fill_vision_handlers(combo: QComboBox) -> None:
        from core.model_manager import available_vision_handlers
        available = available_vision_handlers()
        handler_labels = {
            "qwen25vl": "Qwen 2.5 VL (рекомендуется для Qwen2.5-VL)",
            "llava15": "LLaVA 1.5",
            "llava16": "LLaVA 1.6",
            "minicpmv26": "MiniCPM-V 2.6",
        }
        if not available:
            combo.addItem("(в установленной llama-cpp нет vision)", "")
            combo.setEnabled(False)
        else:
            for handler_id in available:
                combo.addItem(handler_labels.get(handler_id, handler_id), handler_id)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите .gguf файл", "", "GGUF (*.gguf)"
        )
        if path:
            # добавим в комбо как абсолютный путь
            if self.model_combo.findText(path) == -1:
                self.model_combo.addItem(path)
            self.model_combo.setCurrentText(path)

    def _apply_preset(self, idx: int) -> None:
        value = self.preset_combo.currentData()
        if value is None:
            return  # "не менять"

        from core.profiles import (
            DEFAULT_CODER_PROMPT, DEFAULT_COMPANION_PROMPT, DEFAULT_VISION_PROMPT,
        )
        if value == "default":
            if self._profile.kind == ProfileKind.CODER:
                text = DEFAULT_CODER_PROMPT
            elif self._profile.kind == ProfileKind.VISION:
                text = DEFAULT_VISION_PROMPT
            elif self._profile.kind == ProfileKind.RESEARCHER:
                text = DEFAULT_RESEARCHER_PROMPT
            else:
                text = DEFAULT_COMPANION_PROMPT
        else:
            text = value

        self.prompt_edit.setPlainText(text)
        # после применения возвращаем индекс на "не менять"
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

    def _update_prompt_counter(self) -> None:
        text = self.prompt_edit.toPlainText()
        toks = TokenBudget.estimate(text)
        self.prompt_counter.setText(f"~{toks} токенов")
        # подкрашиваем если жирный промпт
        if toks > 800:
            self.prompt_counter.setStyleSheet(f"color:{Palette.ACCENT_RED}; font-size:11px;")
        elif toks > 400:
            self.prompt_counter.setStyleSheet(f"color:{Palette.ACCENT_AMBER}; font-size:11px;")
        else:
            self.prompt_counter.setStyleSheet(f"color:{Palette.TEXT_DIM}; font-size:11px;")

    # ---------- helpers ----------

    def _kind_caption(self) -> str:
        return {
            ProfileKind.CODER: "тип: кодер",
            ProfileKind.COMPANION: "тип: компаньон",
            ProfileKind.RESEARCHER: "тип: поисковик",
            ProfileKind.VISION: "тип: vision (изображения)",
            ProfileKind.GENERIC: "тип: общий",
        }.get(self._profile.kind, "")

    @staticmethod
    def _template_label(t: ChatTemplate) -> str:
        return {
            ChatTemplate.AUTO: "Авто (по имени файла)",
            ChatTemplate.CHATML: "ChatML (Qwen, Hermes, Dolphin)",
            ChatTemplate.LLAMA3: "Llama 3",
            ChatTemplate.MISTRAL: "Mistral / Mixtral",
            ChatTemplate.GEMMA: "Gemma",
            ChatTemplate.DEEPSEEK: "DeepSeek",
        }[t]

    @staticmethod
    def _wrap_row(layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

    @staticmethod
    def _stylesheet() -> str:
        return form_controls_qss() + f"""
            QTabWidget::pane {{
                border: 1px solid {Palette.BORDER};
                background: {Palette.BG_ASSISTANT};
                border-radius: 8px;
                top: -1px;
            }}
            QTabBar::tab {{
                background: transparent;
                color: {Palette.TEXT_SECONDARY};
                padding: 8px 18px;
                font-size: 12px;
                border: none;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {Palette.TEXT_PRIMARY};
                border-bottom: 2px solid {Palette.ACCENT};
            }}
            QTabBar::tab:hover {{
                color: {Palette.TEXT_PRIMARY};
                background: rgba(167,139,250,0.06);
            }}
        """
