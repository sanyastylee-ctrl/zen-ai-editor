"""
Главное окно Zen AI Editor.

Связывает все подсистемы:
- ProfileManager (хранит профили Кодер / Компаньон)
- ModelManager (одна модель в VRAM, переключение через выгрузку)
- InferenceWorker (стриминг ответов из QThread)
- ProjectRAG (опционально — контекст из кода проекта)
- SandboxWidget (терминал)
- QScintilla editor + DiffApplyDialog

Раздельные истории чата для каждого профиля — переключение между Кодером и
Алисой не сбрасывает разговор.
"""

from __future__ import annotations

import html
import os
import re
import sys
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFileSystemModel, QFont, QKeySequence, QShortcut, QAction
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QFrame, QTreeView, QTabWidget,
    QLabel, QProgressBar, QFileDialog, QMessageBox, QMenu, QSizePolicy
)

from core.profiles import ProfileManager, ProfileKind, AIProfile
from core.model_manager import ModelManager, LLAMA_AVAILABLE
from core.paths import resolve_model_path
from core.token_budget import TokenBudget
from core.projects import ProjectManager
from core.settings import PersistentSettings
from core.chat_store import ChatSessionStore
from core.companion import CompanionMemoryStore, extract_explicit_memory
from core.diagnostics import write_log
from ai.agent import AgentWorker, sanitize_agent_history
from ai.research import ResearchWorker
from ai.vision import VisionWorker
from ai.worker import InferenceWorker
from core.research import needs_web_search
from ui.chat import ChatView
from ui.chat.styles import Palette
from ui.agent_progress import AgentProgressOverlay
from ui.profile_switcher import ProfileSwitcher
from ui.settings_dialog import SettingsDialog
from rag.project_rag import ProjectRAG, RagIndexWorker, RAG_AVAILABLE
from sandbox.terminal import SandboxWidget
from widgets.code_editor import (
    make_scintilla_editor, set_lexer_for_file, PythonHighlighter,
    DiffApplyDialog, QSCI_AVAILABLE,
)


# ============================================================
# MAIN WINDOW
# ============================================================

class ZenEditor(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Zen AI Editor")
        self.resize(1400, 900)
        self.setMinimumSize(900, 600)

        # ---------- стейт ----------
        self.pm = ProfileManager()
        self.pm.load()

        # ProjectManager восстанавливает последний проект и делает chdir на него.
        # Это должно быть ДО _build_ui, чтобы дерево файлов взяло правильный root.
        self.projects = ProjectManager.instance()

        self.app_settings = {
            "use_rag": False,
            "diff_before_apply": True,
            "agent_confirmation_policy": "confirm_changes",
        }
        saved_settings = PersistentSettings.load()
        for key in self.app_settings:
            if key in saved_settings:
                self.app_settings[key] = saved_settings[key]

        self.current_file_path: str = ""
        self.worker: InferenceWorker | AgentWorker | None = None

        # стрим текущего ответа (для apply_code_from_chat)
        self._current_ai_response = ""
        self._current_message_buffer = ""
        self._current_response_profile_id: str = ""
        self._current_assistant_record: dict | None = None
        self._current_assistant_sender = "Ассистент"
        self._current_assistant_kind = ""
        self._agent_tool_records: dict[str, tuple[str, dict]] = {}
        self._agent_continuation_by_profile: dict[str, dict] = {}
        self.workspace_visible = False
        self.workspace_pinned = False
        self.active_workspace_tab = "editor"
        self.auto_open_reason = ""
        self._stream_update_timer = QTimer(self)
        self._stream_update_timer.setSingleShot(True)
        self._stream_update_timer.setInterval(50)
        self._stream_update_timer.timeout.connect(self._flush_stream_update)

        # Раздельные истории для разных профилей.
        # Ключ — profile_id, значение — list[(user, assistant)].
        self._histories: dict[str, list[tuple[str, str]]] = {}
        # Визуальные записи чата для ChatView (по профилю).
        self._chat_records: dict[str, list[dict]] = {}
        self._chat_store = ChatSessionStore()
        self._restore_chat_sessions()

        self.rag = ProjectRAG()
        self._rag_worker: RagIndexWorker | None = None
        self._rag_chunks = 0
        self._terminal_history: list[str] = []

        # ---------- ui ----------
        self.setStyleSheet(self._stylesheet())
        self._build_ui()

        # Подключаем колбэки ModelManager для индикатора статуса
        mm = ModelManager.instance()
        mm.on_load_start(self._on_model_load_start)
        mm.on_load_finish(self._on_model_load_finish)
        mm.on_evict(self._on_model_evict)

        # Подгружаем RAG из кэша
        loaded = self.rag.load_index()
        if loaded > 0:
            self._rag_chunks = loaded
            self.rag_status_label.setText(f"RAG: {loaded} чанков (кэш)")

        # Восстанавливаем активный профиль в свитчере
        active_companion = self.pm.get_active(ProfileKind.COMPANION)
        active_coder = self.pm.get_active(ProfileKind.CODER)
        active_researcher = self.pm.get_active(ProfileKind.RESEARCHER)
        restored_active = self.pm.get(self._chat_store.last_profile_id())
        if restored_active and restored_active.kind == ProfileKind.VISION:
            restored_active = None
        first_active = restored_active or active_coder or active_companion or active_researcher
        if first_active:
            self.profile_switcher.set_profiles(self._main_switcher_profiles(), first_active.id)
            self._refresh_chat_view(first_active.id)
            self._chat_store.set_last_profile_id(first_active.id)
        self._update_token_bar()

        # Заголовок и статус с именем проекта
        self._update_window_title()

        # начальное состояние workspace-кнопок (workspace скрыт на старте)
        self._sync_workspace_toggle_buttons()

    # ============================================================
    # UI BUILD
    # ============================================================

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)

        self.main_splitter.addWidget(self._build_sidebar())
        self.main_splitter.addWidget(self._build_right_zone())
        self.main_splitter.setSizes([250, 1150])
        self.sidebar.hide()  # стартуем со скрытым проектом

    def _build_sidebar(self) -> QWidget:
        self.sidebar = QFrame()
        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # RAG
        rag_row = QHBoxLayout()
        self.index_btn = QPushButton("⟳ Индекс RAG")
        self.index_btn.setObjectName("green_btn")
        self.index_btn.setToolTip("Проиндексировать проект (faiss + sentence-transformers)")
        self.index_btn.clicked.connect(self.start_rag_indexing)
        rag_row.addWidget(self.index_btn)

        self.rag_status_label = QLabel("RAG: –")
        self.rag_status_label.setObjectName("token_label")
        rag_row.addWidget(self.rag_status_label)
        layout.addLayout(rag_row)

        # Файловое дерево
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(self.projects.current)

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(self.projects.current))
        self.tree_view.hideColumn(1)
        self.tree_view.hideColumn(2)
        self.tree_view.hideColumn(3)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.doubleClicked.connect(self.open_file)
        
        # Включаем контекстное меню для прикрепления файлов из дерева
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._on_tree_context_menu)
        
        layout.addWidget(self.tree_view)

        return self.sidebar

    def _build_right_zone(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(5)

        # --- ТОП-БАР ---
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        self.toggle_btn = QPushButton("☰ Дерево")
        self.toggle_btn.setObjectName("secondary")
        self.toggle_btn.setToolTip("Показать / скрыть дерево файлов")
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        top_bar.addWidget(self.toggle_btn)

        # Кнопка проекта с выпадающим меню "Недавние"
        self.project_btn = QPushButton("📁 Проект")
        self.project_btn.setObjectName("secondary")
        self.project_btn.setToolTip("Открыть другой проект (Ctrl+Shift+O)\nКлик — меню недавних")
        self.project_btn.clicked.connect(self._show_project_menu)
        top_bar.addWidget(self.project_btn)

        self.save_btn = QPushButton("💾 Сохранить")
        self.save_btn.setObjectName("secondary")
        self.save_btn.clicked.connect(self.save_file)
        top_bar.addWidget(self.save_btn)

        self.run_file_btn = QPushButton("▶ Запустить")
        self.run_file_btn.setObjectName("green_btn")
        self.run_file_btn.clicked.connect(self.run_current_file)
        top_bar.addWidget(self.run_file_btn)

        # Постоянные переключатели рабочей области — всегда видны,
        # даже когда workspace скрыт. Без них юзер не знает где редактор/терминал.
        self.editor_toggle_btn = QPushButton("⌨ Редактор")
        self.editor_toggle_btn.setObjectName("secondary")
        self.editor_toggle_btn.setToolTip("Открыть/скрыть редактор (Ctrl+E)")
        self.editor_toggle_btn.clicked.connect(lambda: self._toggle_workspace_tab("editor"))
        top_bar.addWidget(self.editor_toggle_btn)

        self.terminal_toggle_btn = QPushButton("⌦ Терминал")
        self.terminal_toggle_btn.setObjectName("secondary")
        self.terminal_toggle_btn.setToolTip("Открыть/скрыть терминал (Ctrl+`)")
        self.terminal_toggle_btn.clicked.connect(lambda: self._toggle_workspace_tab("terminal"))
        top_bar.addWidget(self.terminal_toggle_btn)

        top_bar.addStretch()

        # переключатель профилей
        self.profile_switcher = ProfileSwitcher()
        self.profile_switcher.profile_changed.connect(self._on_profile_changed)
        top_bar.addWidget(self.profile_switcher)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setObjectName("secondary")
        self.settings_btn.setFixedWidth(36)
        self.settings_btn.setToolTip("Настройки")
        self.settings_btn.clicked.connect(self.open_settings)
        top_bar.addWidget(self.settings_btn)

        layout.addLayout(top_bar)

        # --- СТАТУС МОДЕЛИ ---
        status_row = QHBoxLayout()
        self.model_status = QLabel("Модель не загружена")
        self.model_status.setObjectName("token_label")
        status_row.addWidget(self.model_status)
        status_row.addStretch()

        self.token_label = QLabel("Токены: 0 / 8192")
        self.token_label.setObjectName("token_label")
        status_row.addWidget(self.token_label)

        self.token_bar = QProgressBar()
        self.token_bar.setRange(0, 100)
        self.token_bar.setValue(0)
        self.token_bar.setFormat("%p%")
        self.token_bar.setMaximumWidth(150)
        status_row.addWidget(self.token_bar)
        layout.addLayout(status_row)

        # --- ШОРТКАТЫ ---
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_file)
        QShortcut(QKeySequence("Ctrl+Shift+O"), self).activated.connect(self.open_project_dialog)
        QShortcut(QKeySequence("Ctrl+E"), self).activated.connect(lambda: self._toggle_workspace_tab("editor"))
        QShortcut(QKeySequence("Ctrl+`"), self).activated.connect(lambda: self._toggle_workspace_tab("terminal"))

        # --- WORK SPLITTER ---
        self.work_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.work_splitter, 1)

        self.work_splitter.addWidget(self._build_chat_zone())
        workspace = self._build_editor_zone()
        self.work_splitter.addWidget(workspace)
        workspace.setVisible(False)
        self.work_splitter.setSizes([900, 420])

        return right

    def _build_chat_zone(self) -> QWidget:
        zone = QWidget()
        layout = QVBoxLayout(zone)
        layout.setContentsMargins(0, 0, 8, 0)

        self.chat_view = ChatView()
        self.chat_view.insert_requested.connect(self._insert_code_from_chat)
        layout.addWidget(self.chat_view, 1)

        self.agent_progress = AgentProgressOverlay()
        self.agent_progress.stop_btn.clicked.connect(self.stop_generation)
        layout.addWidget(self.agent_progress)

        # ввод
        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.attach_btn = QPushButton("📎")
        self.attach_btn.setObjectName("secondary")
        self.attach_btn.setFixedWidth(36)
        self.attach_btn.setToolTip("Прикрепить файл к запросу")
        self.attach_btn.clicked.connect(self.attach_file)
        input_row.addWidget(self.attach_btn)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Сообщение... (Enter)")
        self.chat_input.returnPressed.connect(self._on_send_enter)
        self.chat_input.textChanged.connect(self._update_token_bar)
        input_row.addWidget(self.chat_input)

        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setFixedWidth(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Остановить генерацию")
        self.stop_btn.clicked.connect(self.stop_generation)
        input_row.addWidget(self.stop_btn)

        layout.addLayout(input_row)

        # прикреплённые файлы — список под полем ввода
        self.attached_files: list[str] = []
        self.attached_label = QLabel("")
        self.attached_label.setObjectName("token_label")
        self.attached_label.setVisible(False)
        layout.addWidget(self.attached_label)

        return zone

    def _build_editor_zone(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("workspace_panel")
        panel.setMinimumWidth(360)
        panel.setMaximumWidth(760)
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.workspace_title_label = QLabel("Workspace")
        self.workspace_title_label.setObjectName("workspace_title")
        head.addWidget(self.workspace_title_label)

        self.workspace_reason_label = QLabel("Файл не открыт")
        self.workspace_reason_label.setObjectName("workspace_reason")
        head.addWidget(self.workspace_reason_label, 1)

        self.workspace_pin_btn = QPushButton("Pin")
        self.workspace_pin_btn.setObjectName("secondaryCompact")
        self.workspace_pin_btn.setCheckable(True)
        self.workspace_pin_btn.setToolTip("Закрепить рабочую область")
        self.workspace_pin_btn.clicked.connect(self._toggle_workspace_pin)
        head.addWidget(self.workspace_pin_btn)

        self.workspace_hide_btn = QPushButton("Hide")
        self.workspace_hide_btn.setObjectName("secondaryCompact")
        self.workspace_hide_btn.setToolTip("Скрыть рабочую область")
        self.workspace_hide_btn.clicked.connect(self._hide_workspace_manual)
        head.addWidget(self.workspace_hide_btn)

        panel_layout.addLayout(head)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.South)

        if QSCI_AVAILABLE:
            self.code_editor = make_scintilla_editor()
            self.code_editor.textChanged.connect(self._update_token_bar)
            editor_widget = self.code_editor
        else:
            self.code_editor = QTextEdit()
            self.code_editor.setFont(QFont("Consolas", 11))
            self.code_editor.textChanged.connect(self._update_token_bar)
            self._highlighter = PythonHighlighter(self.code_editor.document())
            editor_widget = self.code_editor

        self._editor_widget = editor_widget
        self._editor_widget.setVisible(False)

        editor_page = QWidget()
        editor_layout = QVBoxLayout(editor_page)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)
        self._editor_empty_label = QLabel(
            "Файл не открыт\nКогда кодер изменит файл, он появится здесь"
        )
        self._editor_empty_label.setObjectName("workspace_empty")
        self._editor_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._editor_empty_label.setWordWrap(True)
        editor_layout.addWidget(self._editor_empty_label, 1)
        editor_layout.addWidget(self._editor_widget, 1)

        tabs.addTab(editor_page, "Редактор")

        self.sandbox = SandboxWidget()
        tabs.addTab(self.sandbox, "Терминал")

        self._editor_tabs = tabs
        panel_layout.addWidget(tabs, 1)
        self.workspace_panel = panel
        return panel

    def _workspace_tab_index(self, tab: str) -> int:
        return {"editor": 0, "terminal": 1}.get(tab, 0)

    def _toggle_workspace_tab(self, tab: str) -> None:
        """
        Клик по кнопке Редактор/Терминал в топбаре:
        - workspace скрыт → показать на нужном табе
        - workspace виден, но на другом табе → переключить таб
        - workspace виден и уже на этом табе → скрыть
        """
        if not hasattr(self, "workspace_panel"):
            return
        currently_visible = self.workspace_visible
        current_tab = self.active_workspace_tab
        if currently_visible and current_tab == tab:
            self._hide_workspace_manual()
        else:
            # ручное открытие — пиним, чтобы авто-логика не закрыла
            self.workspace_pinned = True
            if hasattr(self, "workspace_pin_btn"):
                self.workspace_pin_btn.setChecked(True)
                self.workspace_pin_btn.setText("Pinned")
            label = "Терминал" if tab == "terminal" else "Редактор"
            self._show_workspace(tab, label)
        self._sync_workspace_toggle_buttons()

    def _sync_workspace_toggle_buttons(self) -> None:
        """Подсветка активной кнопки в топбаре под текущее состояние."""
        if not hasattr(self, "editor_toggle_btn"):
            return
        ed_active = self.workspace_visible and self.active_workspace_tab == "editor"
        term_active = self.workspace_visible and self.active_workspace_tab == "terminal"
        self.editor_toggle_btn.setObjectName("toolToggleActive" if ed_active else "secondary")
        self.terminal_toggle_btn.setObjectName("toolToggleActive" if term_active else "secondary")
        # переполировка стиля после смены objectName
        for b in (self.editor_toggle_btn, self.terminal_toggle_btn):
            b.style().unpolish(b)
            b.style().polish(b)

    def _show_workspace(self, tab: str = "editor", reason: str = "") -> None:
        if not hasattr(self, "workspace_panel"):
            return
        self.workspace_visible = True
        self.active_workspace_tab = tab
        self.auto_open_reason = reason or ""
        self.workspace_panel.setVisible(True)
        self._editor_tabs.setCurrentIndex(self._workspace_tab_index(tab))
        if hasattr(self, "workspace_reason_label"):
            self.workspace_reason_label.setText(reason or ("Терминал" if tab == "terminal" else "Редактор"))
        # Если панель была скрыта, splitter не всегда возвращает ей место сам.
        if hasattr(self, "work_splitter"):
            self.work_splitter.setSizes([820, 460])
        self._sync_workspace_toggle_buttons()

    def _hide_workspace_manual(self) -> None:
        self.workspace_pinned = False
        if hasattr(self, "workspace_pin_btn"):
            self.workspace_pin_btn.setChecked(False)
            self.workspace_pin_btn.setText("Pin")
        self._hide_workspace(force=True)

    def _hide_workspace(self, force: bool = False) -> None:
        if not hasattr(self, "workspace_panel"):
            return
        if self.workspace_pinned and not force:
            return
        self.workspace_visible = False
        self.workspace_panel.setVisible(False)
        self._sync_workspace_toggle_buttons()

    def _toggle_workspace_pin(self) -> None:
        self.workspace_pinned = bool(self.workspace_pin_btn.isChecked())
        self.workspace_pin_btn.setText("Pinned" if self.workspace_pinned else "Pin")
        if self.workspace_pinned:
            self._show_workspace(self.active_workspace_tab or "editor", "Закреплено")

    def _set_editor_empty_state(self, is_empty: bool) -> None:
        if hasattr(self, "_editor_empty_label"):
            self._editor_empty_label.setVisible(is_empty)
        if hasattr(self, "_editor_widget"):
            self._editor_widget.setVisible(not is_empty)

    def _open_path_in_editor(self, path: str, reason: str = "Файл открыт") -> bool:
        if not path or not os.path.isfile(path):
            return False
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть {path}: {e}")
            return False
        self._set_editor_text(content)
        self.current_file_path = path
        self._set_editor_empty_state(False)
        if QSCI_AVAILABLE:
            set_lexer_for_file(self.code_editor, path)
        self.sandbox.set_cwd(os.path.dirname(path))
        self._show_workspace("editor", reason)
        return True

    def _tool_path_to_project_file(self, path: str) -> str:
        if not path:
            return ""
        candidate = path if os.path.isabs(path) else os.path.join(self.projects.current, path)
        candidate = os.path.realpath(candidate)
        root = os.path.realpath(self.projects.current)
        try:
            if os.path.commonpath([root, candidate]) != root:
                return ""
        except ValueError:
            return ""
        return candidate

    # ============================================================
    # PROFILE
    # ============================================================

    def _on_profile_changed(self, profile_id: str) -> None:
        """Юзер кликнул на другой профиль в свитчере."""
        profile = self.pm.get(profile_id)
        if not profile:
            return
        self._chat_store.set_last_profile_id(profile_id)

        # обновляем активный для слота профиля
        self.pm.set_active(profile.kind, profile_id)

        # выгружаем все остальные модели (max_loaded=1, но на всякий случай)
        # Это даст явный фидбэк "переключаюсь"
        mm = ModelManager.instance()
        target_path = resolve_model_path(profile.model_file)
        for loaded_path in mm.loaded():
            if loaded_path != target_path:
                mm.unload(loaded_path)

        # переключаем вид чата на историю этого профиля
        self._refresh_chat_view(profile_id)
        self._update_token_bar()

        # обновляем плейсхолдер ввода под тип профиля
        if profile.kind == ProfileKind.CODER:
            self.chat_input.setPlaceholderText("Запрос к кодеру (Enter)")
        elif profile.kind == ProfileKind.COMPANION:
            self.chat_input.setPlaceholderText(f"Написать {profile.name}... (Enter)")
        elif profile.kind == ProfileKind.RESEARCHER:
            self.chat_input.setPlaceholderText("Спросить Поисковик... (Enter)")
        else:
            self.chat_input.setPlaceholderText("Сообщение... (Enter)")

    def _active_profile(self) -> AIProfile | None:
        pid = self.profile_switcher.active_id()
        return self.pm.get(pid) if pid else None

    def _main_switcher_profiles(self) -> list[AIProfile]:
        """Главный UI показывает людей/режимы, а Vision остаётся capability/debug."""
        visible_kinds = {ProfileKind.CODER, ProfileKind.COMPANION, ProfileKind.RESEARCHER}
        return [p for p in self.pm.all() if p.kind in visible_kinds]

    # ============================================================
    # ЧАТ И КОНТЕКСТ
    # ============================================================

    def _restore_chat_sessions(self) -> None:
        for profile in self.pm.all():
            records, history = self._chat_store.load_profile(profile.id)
            if records:
                self._chat_records[profile.id] = records
            if history:
                self._histories[profile.id] = history

    def _persist_chat_session(self, profile_id: str) -> None:
        profile = self.pm.get(profile_id)
        if profile is None:
            return
        self._chat_store.save_profile(
            profile_id,
            profile.name,
            self._chat_records.get(profile_id, []),
            self._histories.get(profile_id, []),
        )

    def _get_project_tree(self, root_dir: str, max_depth: int = 2) -> str:
        """Создает текстовую карту проекта для ИИ (игнорирует мусор)."""
        tree_str = "📂 Структура проекта (карта файлов):\n"
        ignore_dirs = {
            "__pycache__", "venv", "env", ".venv", "node_modules", 
            "build", "dist", ".git", ".idea", ".zen_ai"
        }
        
        def walk(dir_path: str, prefix: str = "", depth: int = 0) -> str:
            if depth > max_depth:
                return prefix + "└── ...\n"
            res = ""
            try:
                items = sorted(os.listdir(dir_path))
            except Exception:
                return res
            
            # Отфильтровываем скрытые файлы и мусорные папки
            items = [item for item in items if not item.startswith('.') and item not in ignore_dirs]
            
            for i, item in enumerate(items):
                path = os.path.join(dir_path, item)
                is_last = (i == len(items) - 1)
                pointer = "└── " if is_last else "├── "
                res += prefix + pointer + item + "\n"
                
                if os.path.isdir(path):
                    extension = "    " if is_last else "│   "
                    res += walk(path, prefix + extension, depth + 1)
            return res

        tree = walk(root_dir)
        return tree_str + tree if tree else ""

    def _refresh_chat_view(self, profile_id: str) -> None:
        self.chat_view.set_records(self._chat_records.setdefault(profile_id, []))

    def _append_chat(self, profile_id: str, html: str) -> None:
        self._add_chat_record(profile_id, {
            "role": "system",
            "sender": "System",
            "text": self._plain_from_html(html),
        })

    def _add_chat_record(self, profile_id: str, record: dict) -> None:
        record.setdefault("time", datetime.now().isoformat(timespec="minutes"))
        records = self._chat_records.setdefault(profile_id, [])
        records.append(record)
        if self.profile_switcher.active_id() == profile_id:
            self.chat_view.add_record(record)
        self._persist_chat_session(profile_id)

    def _update_chat_record(self, profile_id: str, record: dict) -> None:
        if self.profile_switcher.active_id() == profile_id:
            self.chat_view.update_record(record)

    def _quote_log(self, value: object, limit: int = 240) -> str:
        text = str(value)
        if len(text) > limit:
            text = text[:limit] + "..."
        return text.replace("\\", "\\\\").replace('"', "'").replace("\n", " ")

    def _worker_is_running(self) -> bool:
        worker = getattr(self, "worker", None)
        if worker is None or not hasattr(worker, "isRunning"):
            return False
        try:
            return bool(worker.isRunning())
        except RuntimeError:
            return False

    def _worker_state_flags(self) -> dict[str, object]:
        worker = getattr(self, "worker", None)
        return {
            "worker_type": type(worker).__name__ if worker is not None else "none",
            "worker_running": self._worker_is_running(),
            "stop_enabled": bool(self.stop_btn.isEnabled()) if hasattr(self, "stop_btn") else False,
            "current_profile": getattr(self, "_current_response_profile_id", ""),
            "has_assistant_record": bool(getattr(self, "_current_assistant_record", None)),
        }

    def _log_worker_state(self, event: str) -> None:
        flags = self._worker_state_flags()
        write_log(
            "[ui_worker_state] "
            f'event="{event}" worker_type="{flags["worker_type"]}" '
            f'worker_running="{flags["worker_running"]}" stop_enabled="{flags["stop_enabled"]}" '
            f'current_profile="{self._quote_log(flags["current_profile"])}" '
            f'has_assistant_record="{flags["has_assistant_record"]}"'
        )

    def _clear_stale_worker_if_needed(self) -> None:
        worker = getattr(self, "worker", None)
        if worker is not None and not self._worker_is_running():
            write_log(
                "[ui_worker_state] "
                f'event="clear_stale_worker" worker_type="{type(worker).__name__}"'
            )
            self.worker = None
            self.stop_btn.setEnabled(False)

    def _send_blocked(self, reason: str) -> None:
        write_log(f'[ui_send_blocked] reason="{self._quote_log(reason)}"')
        if hasattr(self, "model_status"):
            self.model_status.setText(f"Send blocked: {reason}")

    def _clear_attachments_after_enqueue(self) -> None:
        if getattr(self, "attached_files", None):
            self.attached_files = []
            self._update_attached_label()

    def _route_name_for_send(self, profile: AIProfile, full_text: str, image_paths: list[str]) -> str:
        if self._should_run_vision_assist(profile, image_paths):
            return "vision" if self._is_vision_only_request(full_text) else "coder_vision"
        if profile.kind == ProfileKind.RESEARCHER and self._research_requires_pipeline(profile, full_text):
            return "research"
        if profile.kind == ProfileKind.COMPANION:
            return "companion"
        if (
            profile.kind == ProfileKind.CODER
            and getattr(profile, "agent_mode", False)
            and self._has_agent_intent(full_text, profile.id)
        ):
            return "agent"
        return "normal"

    def _on_send_enter(self) -> None:
        write_log("[ui_send_enter]")
        self.send_message()

    def _on_send_clicked(self) -> None:
        write_log("[ui_send_clicked]")
        self.send_message()

    def send_message(self) -> None:
        self._clear_stale_worker_if_needed()
        self._log_worker_state("before_send")
        text = self.chat_input.text().strip()

        profile = self._active_profile()
        profile_id = profile.id if profile else ""
        write_log(
            "[ui_send_start] "
            f'text_len="{len(text)}" attachment_count="{len(self.attached_files)}" '
            f'active_profile="{self._quote_log(profile_id)}"'
        )

        if not text and not self.attached_files:
            self._send_blocked("empty_message")
            return
        if self._worker_is_running():
            self._send_blocked("worker_busy")
            self._log_worker_state("blocked_busy")
            return

        if not LLAMA_AVAILABLE:
            self._send_blocked("llama_cpp_unavailable")
            self._append_chat_active("<b style='color:#CE9178;'>llama-cpp-python не установлен</b><br>")
            return

        if not profile:
            self._send_blocked("no_active_profile")
            self._append_chat_active("<b style='color:#CE9178;'>Не выбран профиль</b><br>")
            return
        if not profile.model_file:
            self._send_blocked("missing_model_file")
            self._append_chat_active("<b style='color:#CE9178;'>Не выбрана модель в Настройках</b><br>")
            return

        # Проверка: картинки обрабатываются как capability профиля, а не как
        # отдельный главный профиль. Vision Debug может существовать в настройках,
        # но основной UI не переключается на него автоматически.
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        image_paths = [
            p for p in self.attached_files
            if os.path.splitext(p)[1].lower() in image_exts
        ]
        has_images = bool(image_paths)
        if has_images and profile.kind == ProfileKind.CODER and not getattr(profile, "enable_vision_assist", False):
            self._append_chat_active(
                "<i style='color:#CE9178;'>⚠ Прикреплены изображения. "
                "Включите Vision Assist / Глаза в настройках Кодера, чтобы использовать их как visual_context.</i><br>"
            )
        elif has_images and profile.kind == ProfileKind.CODER and getattr(profile, "enable_vision_assist", False):
            if not getattr(profile, "vision_model_file", "") or not getattr(profile, "mmproj_file", ""):
                self._append_chat_active(
                    "<i style='color:#CE9178;'>⚠ Vision Assist включён, но не выбраны vision model/mmproj. "
                    "Изображения пока не будут проанализированы.</i><br>"
                )

        # --- ОБРАБОТКА ВЛОЖЕНИЙ ---
        attachment_text = ""
        attached_paths = list(self.attached_files) 
        
        if self.attached_files:
            attachment_text = self._load_attached_files()

        full_text = text
        if attachment_text and profile.kind == ProfileKind.COMPANION:
            full_text = f"Вложенные файлы:\n{attachment_text}\n\n{text}".strip()
        route_name = self._route_name_for_send(profile, full_text, image_paths)
        write_log(
            "[ui_send_route] "
            f'route="{route_name}" profile_id="{self._quote_log(profile.id)}" '
            f'profile_kind="{profile.kind.value}" image_count="{len(image_paths)}"'
        )

        # Рендерим сообщение пользователя.
        if attached_paths:
            file_names = ", ".join(os.path.basename(p) for p in attached_paths)
            if text:
                display_text = f"{text}\n\nПрикреплено: {file_names}"
            else:
                display_text = f"Прикреплен файл: {file_names}"
        else:
            display_text = text

        self._add_chat_record(profile.id, {
            "role": "user",
            "sender": "Ты",
            "text": display_text,
        })

        # --- ПОДГОТОВКА СВЕРХКОМПЛЕКСНОГО КОНТЕКСТА ---
        code_context = ""
        rag_snippets = ""
        if profile.kind == ProfileKind.CODER:
            # 1. Добавляем структуру проекта
            tree_context = self._get_project_tree(self.projects.current)
            
            # 2. Добавляем открытый файл (если он не пуст)
            active_file_text = self._get_editor_text().strip()
            active_context = f"📝 Текущий открытый файл в редакторе:\n{active_file_text}" if active_file_text else ""
            
            # 3. Собираем итоговый контекст
            blocks = [tree_context]
            if attachment_text:
                blocks.append(attachment_text)
            if active_context:
                blocks.append(active_context)
                
            code_context = "\n\n".join(filter(bool, blocks))

            if self.app_settings.get("use_rag") and self._rag_chunks > 0:
                rag_snippets = self.rag.search(full_text, top_k=5)

        speaker = profile.name if profile.kind == ProfileKind.COMPANION else "Ассистент"
        self._current_ai_response = ""
        self._current_message_buffer = ""
        self._current_response_profile_id = profile.id
        self._current_assistant_sender = speaker
        self._current_assistant_kind = profile.kind.value
        self._current_assistant_record = None
        if profile.kind == ProfileKind.COMPANION:
            write_log(
                "[companion_state_cleanup] "
                f'profile_id="{self._quote_log(profile.id)}" cleared_before_turn="True"'
            )

        history = self._histories.get(profile.id, [])

        if self._should_run_vision_assist(profile, image_paths):
            self._start_vision_assist(
                profile=profile,
                full_text=full_text,
                code_context=code_context,
                rag_snippets=rag_snippets,
                history=history,
                attached_paths=attached_paths,
                image_paths=image_paths,
            )
            self.chat_input.clear()
            self.stop_btn.setEnabled(True)
            self._clear_attachments_after_enqueue()
            write_log(f'[ui_send_finish_enqueue] route="{route_name}" worker_type="{type(self.worker).__name__}"')
            return

        self._start_generation_worker(
            profile=profile,
            full_text=full_text,
            code_context=code_context,
            rag_snippets=rag_snippets,
            history=history,
            attached_paths=attached_paths,
            visual_context="",
        )
        self.chat_input.clear()
        self.stop_btn.setEnabled(True)
        self._clear_attachments_after_enqueue()
        write_log(f'[ui_send_finish_enqueue] route="{route_name}" worker_type="{type(self.worker).__name__}"')
        self.worker.start()

    def _start_generation_worker(
        self,
        profile: AIProfile,
        full_text: str,
        code_context: str,
        rag_snippets: str,
        history: list[tuple[str, str]],
        attached_paths: list[str],
        visual_context: str = "",
    ) -> None:
        if profile.kind != ProfileKind.VISION:
            image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
            attached_paths = [
                path for path in attached_paths
                if os.path.splitext(path)[1].lower() not in image_exts
            ]

        # ВАЖНО: agent_mode профиля — это потенциал, не приказ.
        # Если юзер просто болтает ("как дела", "ты тут", "нет, давай поговорим"),
        # тащить запрос через AgentWorker нельзя — там системный промпт жёстко
        # требует tool-вызовов, а tools на conversational сообщения не нужны.
        # Модель уходит в шаблон "Привет, давай посмотрим файлы" и
        # No-progress guard ловит зацикливание.
        use_agent = (
            profile.kind == ProfileKind.CODER
            and getattr(profile, "agent_mode", False)
            and self._has_agent_intent(full_text, profile.id)
        )

        if use_agent:
            # Передаём continuation_state при ЛЮБОМ агентском запросе если state есть.
            # Раньше — только при «продолжай/давай». Это приводило к потере прогресса:
            # юзер писал «сделай X» после паузы → агент стартовал с нуля, игнорируя
            # всё что было сделано до лимита.
            continuation_state = self._agent_continuation_by_profile.get(profile.id)
            self.worker = AgentWorker(
                profile=profile,
                user_message=full_text,
                code_context=code_context,
                history=self._agent_safe_history(profile.id),
                project_root=self.projects.current,
                terminal_history=self._terminal_history,
                confirmation_policy=self.app_settings.get(
                    "agent_confirmation_policy", "confirm_changes"
                ),
                continuation_state=continuation_state,
                visual_context=visual_context,
            )
            self.worker.tool_started.connect(self._on_agent_tool_started)
            self.worker.tool_finished.connect(self._on_agent_tool_finished)
            self.worker.agent_state_updated.connect(self._on_agent_state_updated)
            self.worker.agent_auto_continue.connect(self._on_agent_auto_continue)
            self.worker.agent_blocked.connect(self._on_agent_blocked)
            self.worker.agent_finished.connect(self._on_agent_finished)
            self.worker.confirmation_requested.connect(self._on_agent_confirmation_requested)
        elif profile.kind == ProfileKind.RESEARCHER and self._research_requires_pipeline(profile, full_text):
            self.worker = ResearchWorker(
                profile=profile,
                user_message=full_text,
            )
        else:
            self.worker = InferenceWorker(
                profile=profile,
                user_message=full_text,
                code_context=code_context,
                rag_snippets=rag_snippets,
                history=history,
                user_name=profile.persona.get("user_name", "") if profile.kind == ProfileKind.COMPANION else "",
                attached_files=attached_paths,
                allow_agent_actions=not getattr(profile, "agent_mode", False),
            )
        self.worker.chunk_received.connect(self._on_chunk)
        self.worker.finished_signal.connect(lambda: self._on_generation_done(profile.id, full_text))
        self.worker.model_loading.connect(lambda p: self.model_status.setText(f"Загружаю {os.path.basename(p)}..."))
        self.worker.model_loaded.connect(
            lambda p, ok, err: self.model_status.setText(
                f"✓ {os.path.basename(p)}" if ok else f"✗ Ошибка: {err}"
            )
        )
        self.worker.status.connect(lambda msg: self.model_status.setText(msg))

    def _research_requires_pipeline(self, profile: AIProfile, text: str) -> bool:
        if profile.kind != ProfileKind.RESEARCHER:
            return False
        if not getattr(profile, "require_sources_for_fresh_info", True):
            return False
        return needs_web_search(text, require_sources_for_fresh_info=True)

    def _should_run_vision_assist(self, profile: AIProfile, image_paths: list[str]) -> bool:
        if profile.kind != ProfileKind.CODER:
            return False
        if not image_paths:
            return False
        if not getattr(profile, "enable_vision_assist", False):
            return False
        if getattr(profile, "vision_first_policy", "auto") == "never":
            return False
        return bool(
            getattr(profile, "vision_model_file", "")
            and getattr(profile, "mmproj_file", "")
            and getattr(profile, "vision_handler", "")
        )

    _VISION_ONLY_RE = re.compile(
        r"(что\s+(?:на|видно|изображено)|опиши\s+(?:скрин|изображение|картин)|"
        r"прочитай\s+(?:скрин|текст)|распознай|ocr|what(?:'s| is)?\s+on)",
        re.IGNORECASE,
    )

    _VISION_FIX_RE = re.compile(
        r"(исправь|почини|сделай|реализуй|измени|добавь|fix|repair|implement|change|make)",
        re.IGNORECASE,
    )

    def _is_vision_only_request(self, text: str) -> bool:
        value = text or ""
        if not value.strip():
            return True
        return bool(self._VISION_ONLY_RE.search(value) and not self._VISION_FIX_RE.search(value))

    def _start_vision_assist(
        self,
        profile: AIProfile,
        full_text: str,
        code_context: str,
        rag_snippets: str,
        history: list[tuple[str, str]],
        attached_paths: list[str],
        image_paths: list[str],
    ) -> None:
        answer_mode = self._is_vision_only_request(full_text)
        write_log(
            "[vision_routing] "
            f'profile_id="{profile.id}" answer_mode="{answer_mode}" '
            f'image_count="{len(image_paths)}" text="{full_text[:200].replace(chr(10), " ")}"'
        )
        detail = ", ".join(os.path.basename(path) for path in image_paths)
        record = {
            "role": "tool",
            "sender": "Tool",
            "tool_name": "Vision Assist",
            "detail": f"Images: {detail}",
            "output": "analyzing...",
            "ok": None,
        }
        self._add_chat_record(profile.id, record)

        self.worker = VisionWorker(
            profile=profile,
            user_message=full_text,
            image_paths=image_paths,
            answer_mode=answer_mode,
        )
        if answer_mode:
            self.worker.chunk_received.connect(self._on_chunk)
            self.worker.finished_signal.connect(lambda: self._on_generation_done(profile.id, full_text))
        self.worker.visual_context_ready.connect(
            lambda context: self._on_vision_context_ready(
                profile=profile,
                full_text=full_text,
                code_context=code_context,
                rag_snippets=rag_snippets,
                history=history,
                attached_paths=attached_paths,
                context=context,
                answer_mode=answer_mode,
                record=record,
            )
        )
        self.worker.error_signal.connect(lambda err: self._on_vision_assist_error(profile.id, record, err))
        self.worker.model_loading.connect(lambda p: self.model_status.setText(f"Загружаю Vision {os.path.basename(p)}..."))
        self.worker.model_loaded.connect(
            lambda p, ok, err: self.model_status.setText(
                f"✓ Vision {os.path.basename(p)}" if ok else f"✗ Vision: {err}"
            )
        )
        self.worker.status.connect(lambda msg: self.model_status.setText(msg))
        self.worker.start()

    def _on_vision_context_ready(
        self,
        profile: AIProfile,
        full_text: str,
        code_context: str,
        rag_snippets: str,
        history: list[tuple[str, str]],
        attached_paths: list[str],
        context: str,
        answer_mode: bool,
        record: dict,
    ) -> None:
        short = context if len(context) <= 4000 else context[:4000] + "\n[visual_context truncated in chat]"
        record.update({"output": short or "[empty visual_context]", "ok": bool(context.strip())})
        self._update_chat_record(profile.id, record)
        self._persist_chat_session(profile.id)
        if answer_mode:
            return
        if not context.strip():
            self._append_chat(
                profile.id,
                "<i style='color:#CE9178;'>Vision Assist не смог распознать изображение. "
                "Уточните, что нужно исправить.</i><br>",
            )
            self._on_generation_done(profile.id, full_text)
            return

        visual_block = (
            "## Visual context from Vision Assist\n"
            "This is evidence from attached screenshot/image. It is not a user task.\n"
            f"{context}"
        )
        combined_context = "\n\n".join(filter(bool, [visual_block, code_context]))
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        downstream_attachments = [
            path for path in attached_paths
            if os.path.splitext(path)[1].lower() not in image_exts
        ]
        self.worker = None
        self._start_generation_worker(
            profile=profile,
            full_text=full_text,
            code_context=combined_context,
            rag_snippets=rag_snippets,
            history=history,
            attached_paths=downstream_attachments,
            visual_context=context,
        )
        self.worker.start()

    def _on_vision_assist_error(self, profile_id: str, record: dict, error: str) -> None:
        reason = (error or "unknown error").strip()
        if reason.startswith("Vision Assist error:"):
            reason = reason.split(":", 1)[1].strip()
        record.update({"output": f"Vision analysis failed: {reason}", "ok": False})
        self._update_chat_record(profile_id, record)
        self._persist_chat_session(profile_id)
        write_log(f'[ui_worker_state] event="vision_error_cleanup" reason="{self._quote_log(reason)}"')
        self.stop_btn.setEnabled(False)
        self.worker = None
        self._current_ai_response = ""
        self._current_message_buffer = ""
        self._current_response_profile_id = ""
        self._current_assistant_record = None

    def _on_agent_tool_started(self, payload: dict) -> None:
        if self._current_assistant_record is not None:
            self._current_assistant_record["streaming"] = False
            self._update_chat_record(
                self._current_response_profile_id,
                self._current_assistant_record,
            )
            self._current_assistant_record = None
            self._current_message_buffer = ""

        name = payload.get("name", "")
        args = payload.get("args", {})
        call_id = payload.get("id", "")
        if name == "run_terminal":
            self._show_workspace("terminal", "Команда агента")
        elif name in {"read_file", "write_file", "edit_file", "apply_patch", "create_folder", "move_file", "delete_file_safe"}:
            self._show_workspace("editor", "Файловое действие агента")
        detail = ""
        if isinstance(args, dict):
            if args.get("path"):
                detail = f"Path: {args.get('path', '')}"
            elif args.get("query"):
                detail = f"Query: {args.get('query', '')}"
            elif args.get("command"):
                detail = f"$ {args.get('command', '')}"
        pid = self._current_response_profile_id or self.profile_switcher.active_id()
        if not pid:
            return
        record = {
            "role": "tool",
            "sender": "Tool",
            "tool_name": name,
            "detail": detail,
            "output": "running...",
            "ok": None,
        }
        self._agent_tool_records[str(call_id)] = (pid, record)
        self._add_chat_record(pid, record)

    def _on_agent_tool_finished(self, payload: dict) -> None:
        name = payload.get("name", "")
        call_id = str(payload.get("id", ""))
        output = str(payload.get("output", ""))
        ok = bool(payload.get("ok", False))
        meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
        title = payload.get("title") or f"Tool: {name}"
        short = output
        if len(short) > 4000:
            short = short[:4000] + "\n[output truncated in chat]"
        if name == "run_terminal":
            self._terminal_history.append(short)
            self._terminal_history = self._terminal_history[-5:]
            self._show_workspace("terminal", "Команда завершена")
        extra = ""
        if meta.get("lines"):
            extra = f" ({meta.get('lines')} строк)"
        elif meta.get("count") is not None:
            extra = f" ({meta.get('count')} совпадений)"
        stored = self._agent_tool_records.get(call_id)
        if stored:
            pid, record = stored
            record.update({
                "tool_name": str(title) + extra,
                "output": short,
                "ok": ok,
            })
            self._update_chat_record(pid, record)
            self._persist_chat_session(pid)
        else:
            pid = self._current_response_profile_id or self.profile_switcher.active_id()
            if pid:
                self._add_chat_record(pid, {
                    "role": "tool",
                    "sender": "Tool",
                    "tool_name": str(title) + extra,
                    "detail": "",
                    "output": short,
                    "ok": ok,
                })

    def _on_agent_state_updated(self, snapshot: dict) -> None:
        if hasattr(self, "agent_progress"):
            self.agent_progress.update_state(snapshot)

    def _on_agent_auto_continue(self, snapshot: dict) -> None:
        if hasattr(self, "agent_progress"):
            self.agent_progress.update_state(snapshot)
        write_log(
            "[ui_agent_auto_continue] "
            f'run_id="{self._quote_log(str(snapshot.get("run_id", "")))}" '
            f'reason="{self._quote_log(str(snapshot.get("auto_continue_reason", "")))}" '
            f'count="{snapshot.get("auto_continue_count", "")}"'
        )

    def _on_agent_blocked(self, snapshot: dict) -> None:
        if hasattr(self, "agent_progress"):
            self.agent_progress.set_blocked(snapshot)

    def _on_agent_finished(self, snapshot: dict) -> None:
        if hasattr(self, "agent_progress"):
            self.agent_progress.set_finished(snapshot)
        args = payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {}
        tool_path = str(args.get("path") or meta.get("path") or "")
        if ok and name in {"read_file", "write_file", "edit_file", "apply_patch"} and tool_path:
            file_path = self._tool_path_to_project_file(tool_path)
            if file_path and os.path.isfile(file_path):
                self._open_path_in_editor(file_path, f"{name}: {os.path.basename(file_path)}")

    def _on_agent_confirmation_requested(self, payload: dict) -> None:
        name = payload.get("name", "")
        args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
        call_id = payload.get("id", "")
        preview = str(payload.get("preview", ""))

        subject = args.get("path") or args.get("command") or args.get("query") or ""
        msg = f"Разрешить tool «{name}»?"
        if subject:
            msg += f"\n\n{subject}"

        box = QMessageBox(self)
        box.setWindowTitle("Agent confirmation")
        box.setText(msg)
        if preview:
            box.setDetailedText(preview[:20000])
        box.setIcon(QMessageBox.Icon.Question)
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.Yes)
        accepted = box.exec() == QMessageBox.StandardButton.Yes

        if self.worker and hasattr(self.worker, "resolve_confirmation"):
            self.worker.resolve_confirmation(call_id, accepted)

    def _on_chunk(self, chunk: str) -> None:
        self._current_ai_response += chunk
        self._current_message_buffer += chunk
        if self._current_assistant_record is None and self._current_response_profile_id:
            self._current_assistant_record = {
                "role": "assistant",
                "sender": self._current_assistant_sender,
                "text": "",
                "profile_kind": self._current_assistant_kind,
                "streaming": True,
            }
            self._add_chat_record(
                self._current_response_profile_id,
                self._current_assistant_record,
            )
        if self._current_assistant_record is not None:
            self._current_assistant_record["text"] = self._current_message_buffer
            self._current_assistant_record["streaming"] = True
            if not self._stream_update_timer.isActive():
                self._stream_update_timer.start()

    def _flush_stream_update(self) -> None:
        if self._current_assistant_record is None:
            return
        if not self._current_response_profile_id:
            return
        self._current_assistant_record["text"] = self._current_message_buffer
        self._current_assistant_record["streaming"] = True
        self._update_chat_record(
            self._current_response_profile_id,
            self._current_assistant_record,
        )

    # Регекс для детекта намерения действовать. Без слова из этого набора
    # запрос считается conversational и идёт в обычный InferenceWorker.
    _AGENT_INTENT_RE = re.compile(
        r"\b("
        # русские глаголы действия
        r"сделай|сделать|делай|реализуй|реализовывай|реализовать|создай|создать|"
        r"добавь|добавить|измени|изменить|поменяй|правь|исправь|исправить|"
        r"перепиши|переписать|переделай|удали|удалить|"
        r"запусти|запустить|запуск|выполни|выполнить|прогони|"
        r"проверь|проверить|проверка|"
        r"напиши|написать|сгенерируй|собери|"
        # английские
        r"do|make|create|add|change|edit|fix|delete|remove|implement|"
        r"refactor|run|execute|launch|write|build|test"
        r")\b",
        re.IGNORECASE,
    )
    # Слова, явно сигнализирующие продолжение прерванного действия.
    _AGENT_CONTINUE_RE = re.compile(
        r"\b(продолжай|продолжи|дальше|давай|готов[оа]?|continue|go|proceed)\b",
        re.IGNORECASE,
    )

    def _has_agent_intent(self, text: str, profile_id: str) -> bool:
        """
        Решает, должен ли запрос обработать AgentWorker.

        True если:
        - в сообщении есть слово-намерение действия (сделай/создай/run/...)
        - ИЛИ это continuation после сохранённого состояния (давай/дальше)
          И у профиля есть pending continuation_state
        - ИЛИ юзер прикрепил файлы (явный сигнал что нужны действия)
        """
        normalized = (text or "").strip()
        if not normalized:
            return False
        if self.attached_files:
            return True
        if self._AGENT_INTENT_RE.search(normalized):
            return True
        if (self._AGENT_CONTINUE_RE.search(normalized)
                and self._agent_continuation_by_profile.get(profile_id)):
            return True
        return False

    def _agent_safe_history(self, profile_id: str) -> list[tuple[str, str]]:
        return sanitize_agent_history(self._histories.get(profile_id, []))

    def _on_generation_done(self, profile_id: str, user_msg: str) -> None:
        finished_worker = self.worker
        auto_continue = (
            isinstance(finished_worker, AgentWorker)
            and bool(getattr(finished_worker, "auto_continue_requested", False))
            and bool(getattr(finished_worker, "continuation_state", None))
        )
        if self._stream_update_timer.isActive():
            self._stream_update_timer.stop()
        if self._current_assistant_record is not None:
            self._current_assistant_record["text"] = self._current_message_buffer
            self._current_assistant_record["streaming"] = False
            self._update_chat_record(profile_id, self._current_assistant_record)

        hist = self._histories.setdefault(profile_id, [])
        companion_invalid = (
            getattr(finished_worker, "profile", None) is not None
            and getattr(finished_worker.profile, "kind", None) == ProfileKind.COMPANION
            and not bool(getattr(finished_worker, "companion_response_valid", True))
        )
        companion_response_text = self._current_ai_response
        if (
            getattr(finished_worker, "profile", None) is not None
            and getattr(finished_worker.profile, "kind", None) == ProfileKind.COMPANION
        ):
            validated_text = str(getattr(finished_worker, "validated_response_text", "") or "")
            if validated_text and not companion_response_text.strip():
                companion_response_text = validated_text
                self._current_ai_response = validated_text
                self._current_message_buffer = validated_text
                if self._current_assistant_record is not None:
                    self._current_assistant_record["text"] = validated_text
                    self._current_assistant_record["streaming"] = False
                    self._update_chat_record(profile_id, self._current_assistant_record)
            if not companion_response_text.strip():
                companion_invalid = True
                if finished_worker is not None:
                    setattr(finished_worker, "companion_block_reason", "empty_response")
        if not auto_continue:
            if not companion_invalid:
                hist.append((user_msg, companion_response_text))
                self._capture_companion_memory(profile_id, user_msg)
            else:
                write_log(
                    "[companion_response_blocked] "
                    f'profile_id="{self._quote_log(profile_id)}" '
                    f'reason="{self._quote_log(getattr(finished_worker, "companion_block_reason", "invalid"))}" '
                    'saved_to_history="False"'
                )

            if len(hist) > 50:
                del hist[: len(hist) - 50]
            self._persist_chat_session(profile_id)

        if self.worker and hasattr(self.worker, "attached_files"):
            self.worker.attached_files.clear()

        if isinstance(finished_worker, AgentWorker):
            if finished_worker.continuation_state:
                self._agent_continuation_by_profile[profile_id] = finished_worker.continuation_state
            else:
                self._agent_continuation_by_profile.pop(profile_id, None)
        else:
            # Если ответил не AgentWorker — это была болтовня / обычный
            # текстовый запрос. Прерванная агент-сессия больше не актуальна,
            # иначе следующее "давай" подхватит давно неактуальный pending task.
            self._agent_continuation_by_profile.pop(profile_id, None)

        self._current_ai_response = ""
        self._current_message_buffer = ""
        self._current_response_profile_id = ""
        self._current_assistant_record = None
        self._current_assistant_sender = "Ассистент"
        self._current_assistant_kind = ""
        self.stop_btn.setEnabled(False)
        self.worker = None
        if auto_continue:
            profile = self.pm.get(profile_id)
            if profile and getattr(profile, "auto_continue_enabled", True):
                self._current_ai_response = ""
                self._current_message_buffer = ""
                self._current_response_profile_id = profile.id
                self._current_assistant_sender = "Ассистент"
                self._current_assistant_kind = profile.kind.value
                self._current_assistant_record = None
                code_context = self._get_project_tree(self.projects.current) if profile.kind == ProfileKind.CODER else ""
                self._start_generation_worker(
                    profile=profile,
                    full_text=user_msg,
                    code_context=code_context,
                    rag_snippets="",
                    history=self._histories.get(profile.id, []),
                    attached_paths=[],
                    visual_context="",
                )
                self.stop_btn.setEnabled(True)
                write_log(
                    "[ui_agent_auto_continue_start] "
                    f'profile_id="{self._quote_log(profile_id)}" user_len="{len(user_msg)}"'
                )
                self.worker.start()
                return
        profile = self.pm.get(profile_id)
        if profile and profile.kind == ProfileKind.COMPANION:
            write_log(
                "[companion_state_cleanup] "
                f'profile_id="{self._quote_log(profile_id)}" after_finish="True"'
            )

    def _capture_companion_memory(self, profile_id: str, user_msg: str) -> None:
        profile = self.pm.get(profile_id)
        if not profile or profile.kind != ProfileKind.COMPANION:
            return
        if str(profile.persona.get("memory_enabled", "true")).lower() in {"0", "false", "no", "off", "нет"}:
            return
        memory_text = extract_explicit_memory(user_msg)
        if memory_text:
            CompanionMemoryStore().add(memory_text, category="explicit", source="chat")

    def stop_generation(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def _append_chat_active(self, html: str) -> None:
        pid = self.profile_switcher.active_id()
        if pid:
            self._append_chat(pid, html)

    # ============================================================
    # МОДЕЛЬ — колбэки
    # ============================================================

    def _on_model_load_start(self, path: str) -> None:
        self.model_status.setText(f"Загружаю {os.path.basename(path)}...")

    def _on_model_load_finish(self, path: str, ok: bool, err: str | None) -> None:
        if ok:
            self.model_status.setText(f"✓ {os.path.basename(path)}")
        else:
            self.model_status.setText(f"✗ Ошибка: {err}")

    def _on_model_evict(self, path: str) -> None:
        pass

    # ============================================================
    # КОД И ВСТАВКА В РЕДАКТОР
    # ============================================================

    def apply_code_from_chat(self) -> None:
        if not self._current_ai_response:
            pid = self.profile_switcher.active_id()
            if pid and self._histories.get(pid):
                _, last_response = self._histories[pid][-1]
            else:
                return
        else:
            last_response = self._current_ai_response

        code_blocks = re.findall(r"`{3}(?:\w+)?\n(.*?)`{3}", last_response, re.DOTALL)
        if code_blocks:
            new_code = code_blocks[0].strip()
        else:
            new_code = last_response.strip()

        self._insert_code_from_chat(new_code)

    def _insert_code_from_chat(self, new_code: str) -> None:
        old_code = self._get_editor_text()
        has_selection = self._editor_has_selection()

        if self.app_settings.get("diff_before_apply", True) and not has_selection and old_code.strip():
            dlg = DiffApplyDialog(old_code, new_code, parent=self)
            if dlg.exec() != dlg.DialogCode.Accepted:
                return
            new_code = dlg.accepted_code

        if has_selection:
            self._replace_selection(new_code)
        else:
            self._set_editor_text(new_code)

    def _get_editor_text(self) -> str:
        if QSCI_AVAILABLE and hasattr(self.code_editor, "text"):
            return self.code_editor.text()
        return self.code_editor.toPlainText()

    def _set_editor_text(self, text: str) -> None:
        if QSCI_AVAILABLE and hasattr(self.code_editor, "setText"):
            self.code_editor.setText(text)
        else:
            self.code_editor.setPlainText(text)

    def _editor_has_selection(self) -> bool:
        if QSCI_AVAILABLE and hasattr(self.code_editor, "hasSelectedText"):
            return self.code_editor.hasSelectedText()
        return self.code_editor.textCursor().hasSelection()

    def _replace_selection(self, text: str) -> None:
        if QSCI_AVAILABLE and hasattr(self.code_editor, "replaceSelectedText"):
            self.code_editor.replaceSelectedText(text)
        else:
            cursor = self.code_editor.textCursor()
            cursor.insertText(text)

    # ============================================================
    # ФАЙЛЫ, ДЕРЕВО И ВЛОЖЕНИЯ
    # ============================================================

    def _on_tree_context_menu(self, pos) -> None:
        """Обрабатывает правый клик по файлу в дереве."""
        index = self.tree_view.indexAt(pos)
        if not index.isValid():
            return
            
        path = self.file_model.filePath(index)
        menu = QMenu(self)
        
        if os.path.isfile(path):
            attach_action = menu.addAction("📎 Прикрепить к запросу")
            open_action = menu.addAction("📝 Открыть в редакторе")
            
            action = menu.exec(self.tree_view.viewport().mapToGlobal(pos))
            
            if action == attach_action:
                if path not in self.attached_files:
                    self.attached_files.append(path)
                    self._update_attached_label()
            elif action == open_action:
                self.open_file(index)

    def open_file(self, index) -> None:
        path = self.file_model.filePath(index)
        if os.path.isfile(path):
            self._open_path_in_editor(path, f"Открыт {os.path.basename(path)}")

    def save_file(self) -> None:
        if not self.current_file_path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить как", self.projects.current, "Все файлы (*.*)"
            )
            if not path:
                return
            self.current_file_path = path
            if QSCI_AVAILABLE:
                set_lexer_for_file(self.code_editor, path)
            self._set_editor_empty_state(False)
            self._show_workspace("editor", f"Сохранён {os.path.basename(path)}")

        try:
            with open(self.current_file_path, "w", encoding="utf-8") as f:
                f.write(self._get_editor_text())
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить: {e}")

    def run_current_file(self) -> None:
        if not self.current_file_path:
            self._append_chat_active("<i style='color:#888;'>Сначала сохраните файл (Ctrl+S).</i><br>")
            return
        if not self.current_file_path.endswith(".py"):
            self._append_chat_active("<i style='color:#888;'>Запуск поддерживается только для .py файлов.</i><br>")
            return
        self._show_workspace("terminal", f"Запуск {os.path.basename(self.current_file_path)}")
        self.sandbox.set_cwd(os.path.dirname(self.current_file_path))
        self.sandbox.run_code_file(self.current_file_path)

    def attach_file(self) -> None:
        write_log(
            "[ui_attach_start] "
            f'project_root="{self._quote_log(getattr(self.projects, "current", ""))}" '
            f'pending_before="{len(self.attached_files)}"'
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Прикрепить файлы", self.projects.current, "Все файлы (*.*)"
        )
        if not paths:
            write_log("[ui_attach_empty]")
            return
        for p in paths:
            if p not in self.attached_files:
                self.attached_files.append(p)
        write_log(
            "[ui_attach_selected] "
            f'count="{len(paths)}" pending_after="{len(self.attached_files)}" '
            f'files="{self._quote_log(", ".join(os.path.basename(p) for p in paths))}"'
        )
        self._update_attached_label()

    def _update_attached_label(self) -> None:
        """Обновляет текст метки под полем ввода."""
        if not self.attached_files:
            self.attached_label.setVisible(False)
            self.attached_label.setText("")
            return
            
        names = ", ".join(os.path.basename(p) for p in self.attached_files)
        self.attached_label.setText(f"📎 Прикреплено: {names}")
        self.attached_label.setVisible(True)

    def _load_attached_files(self) -> str:
        parts = []
        skip_exts = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.pdf', '.exe', '.zip'}
        for p in self.attached_files:
            ext = os.path.splitext(p)[1].lower()
            if ext in skip_exts:
                continue 
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    parts.append(f"### Прикрепленный файл: {os.path.basename(p)}\n{f.read()}")
            except Exception:
                pass
        return "\n\n".join(parts)

    # ============================================================
    # RAG
    # ============================================================

    def start_rag_indexing(self) -> None:
        if not RAG_AVAILABLE:
            self._append_chat_active(
                "<b style='color:#CE9178;'>RAG недоступен: установите faiss-cpu и sentence-transformers.</b><br>"
            )
            return
        self.index_btn.setEnabled(False)
        self.rag_status_label.setText("RAG: индексация...")
        self._rag_worker = RagIndexWorker(self.rag, self.projects.current)
        self._rag_worker.finished_signal.connect(self._on_rag_indexed)
        self._rag_worker.error_signal.connect(self._on_rag_error)
        self._rag_worker.start()

    def _on_rag_indexed(self, count: int) -> None:
        self._rag_chunks = count
        self.rag_status_label.setText(f"RAG: {count} чанков")
        self.index_btn.setEnabled(True)
        self._append_chat_active(
            f"<i style='color:#4EC9B0;'>✓ Проект проиндексирован: {count} фрагментов.</i><br>"
        )

    def _on_rag_error(self, err: str) -> None:
        self.index_btn.setEnabled(True)
        self.rag_status_label.setText("RAG: ошибка")
        self._append_chat_active(f"<b style='color:#CE9178;'>RAG ошибка: {err}</b><br>")

    # ============================================================
    # НАСТРОЙКИ / СИДБАР / БЮДЖЕТ
    # ============================================================

    # ---------- ПРОЕКТЫ ----------

    def _show_project_menu(self) -> None:
        """Меню при клике на кнопку '📁 Проект': открыть + список недавних."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #252526;
                color: #D4D4D4;
                border: 1px solid #3A3A3A;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 24px 6px 16px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background: #0E639C;
            }
            QMenu::separator {
                height: 1px;
                background: #3A3A3A;
                margin: 4px 8px;
            }
        """)

        open_act = QAction("📂 Открыть папку...", self)
        open_act.setShortcut("Ctrl+Shift+O")
        open_act.triggered.connect(self.open_project_dialog)
        menu.addAction(open_act)

        # Текущий — для информации
        current_label = f"  📍 Сейчас: {self.projects.current_name}"
        current_act = QAction(current_label, self)
        current_act.setEnabled(False)
        menu.addAction(current_act)

        recent = self.projects.recent()
        if recent:
            menu.addSeparator()
            header = QAction("Недавние:", self)
            header.setEnabled(False)
            menu.addAction(header)

            for path in recent[:8]:
                name = os.path.basename(os.path.normpath(path)) or path
                short_path = self._shorten_path(path, 50)
                act = QAction(f"  {name}", self)
                act.setToolTip(short_path)
                act.triggered.connect(lambda _, p=path: self._switch_project(p))
                menu.addAction(act)

        # показываем под кнопкой
        pos = self.project_btn.mapToGlobal(self.project_btn.rect().bottomLeft())
        menu.exec(pos)

    def open_project_dialog(self) -> None:
        """Диалог выбора папки."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Открыть проект",
            self.projects.current,
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if path:
            self._switch_project(path)

    def _switch_project(self, new_path: str) -> None:
        """Переключение на другой проект. Спрашивает про несохранённые изменения."""
        # Несохранённые правки в редакторе — спрашиваем
        if self._editor_has_unsaved_changes():
            btn = QMessageBox.question(
                self, "Несохранённые изменения",
                "В редакторе есть несохранённый текст. Сохранить перед переключением?",
                (QMessageBox.StandardButton.Save
                 | QMessageBox.StandardButton.Discard
                 | QMessageBox.StandardButton.Cancel),
                QMessageBox.StandardButton.Save,
            )
            if btn == QMessageBox.StandardButton.Cancel:
                return
            if btn == QMessageBox.StandardButton.Save:
                self.save_file()

        if not self.projects.open(new_path):
            self._append_chat_active(
                f"<i style='color:#CE9178;'>Не удалось открыть проект: {self._escape(new_path)}</i><br>"
            )
            return

        # Перенастраиваем дерево файлов
        self.file_model.setRootPath(self.projects.current)
        self.tree_view.setRootIndex(self.file_model.index(self.projects.current))

        # Терминал — новый cwd
        self.sandbox.set_cwd(self.projects.current)

        # Закрываем открытый файл, если он не из нового проекта
        if self.current_file_path:
            try:
                rel = os.path.relpath(self.current_file_path, self.projects.current)
                # если путь начинается с '..' — файл вне проекта
                if rel.startswith(".."):
                    self._set_editor_text("")
                    self.current_file_path = ""
            except ValueError:
                # разные диски на Windows — точно вне проекта
                self._set_editor_text("")
                self.current_file_path = ""

        # RAG-индекс старого проекта не подходит для нового
        self.rag = ProjectRAG()
        self._rag_chunks = 0
        # пробуем подхватить индекс нового проекта, если он раньше индексировался
        loaded = self.rag.load_index()
        if loaded > 0:
            self._rag_chunks = loaded
            self.rag_status_label.setText(f"RAG: {loaded} чанков (кэш)")
        else:
            self.rag_status_label.setText("RAG: – (новый проект)")

        # Заголовок окна
        self._update_window_title()

        # Подсказка в чате
        self._append_chat_active(
            f"<i style='color:#4EC9B0;'>📁 Проект: {self._escape(self.projects.current_name)} "
            f"<span style='color:#888;'>({self._escape(self.projects.current)})</span></i><br>"
        )

    def _editor_has_unsaved_changes(self) -> bool:
        """
        Быстрая эвристика: если в редакторе есть текст и нет открытого файла —
        точно несохранённый. Точное отслеживание dirty-флага потребовало бы
        отдельной логики, пока обходимся этим.
        """
        text = self._get_editor_text().strip()
        if not text:
            return False
        if not self.current_file_path:
            return True
        # если есть открытый файл — сравниваем содержимое с тем что на диске
        try:
            with open(self.current_file_path, "r", encoding="utf-8", errors="ignore") as f:
                on_disk = f.read()
            return on_disk != self._get_editor_text()
        except Exception:
            return False

    def _update_window_title(self) -> None:
        name = self.projects.current_name
        self.setWindowTitle(f"Zen AI Editor — {name}")

    @staticmethod
    def _shorten_path(path: str, max_len: int = 50) -> str:
        if len(path) <= max_len:
            return path
        return "…" + path[-(max_len - 1):]

    # ---------- НАСТРОЙКИ И ОСТАЛЬНОЕ ----------

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.pm, self.app_settings, parent=self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            self.app_settings.update(dlg.get_app_settings())
            saved = PersistentSettings.load()
            saved.update(self.app_settings)
            PersistentSettings.save(saved)
            current = self.profile_switcher.active_id()
            profiles = self.pm.all()
            if current and current not in {p.id for p in profiles}:
                current = profiles[0].id if profiles else None
            if current and self.pm.get(current) and self.pm.get(current).kind == ProfileKind.VISION:
                visible = self._main_switcher_profiles()
                current = visible[0].id if visible else None
            self.profile_switcher.set_profiles(self._main_switcher_profiles(), current)
            self._update_token_bar()

    def toggle_sidebar(self) -> None:
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def _update_token_bar(self) -> None:
        profile = self._active_profile()
        if not profile:
            self.token_label.setText("Токены: —")
            self.token_bar.setValue(0)
            return

        budget = TokenBudget(
            n_ctx=profile.n_ctx,
            max_response_tokens=profile.max_tokens,
        )
        prompt = self.chat_input.text()
        code_ctx = self._get_editor_text() if profile.kind == ProfileKind.CODER else ""
        sys_p = profile.system_prompt
        pct = budget.usage_percent(sys_p, prompt, code_ctx)
        used = (
            TokenBudget.estimate(sys_p) +
            TokenBudget.estimate(prompt) +
            TokenBudget.estimate(code_ctx)
        )

        self.token_bar.setValue(pct)
        self.token_label.setText(f"Токены: ~{used} / {profile.n_ctx}")

        if pct >= 90:
            self.token_bar.setStyleSheet("QProgressBar::chunk { background: #CE4040; border-radius: 3px; }")
        elif pct >= 70:
            self.token_bar.setStyleSheet("QProgressBar::chunk { background: #CDA040; border-radius: 3px; }")
        else:
            self.token_bar.setStyleSheet("")

    # ============================================================
    # УТИЛИТЫ
    # ============================================================

    @staticmethod
    def _escape(text: str) -> str:
        return (text.replace("&", "&amp;")
                    .replace("<", "&lt;").replace(">", "&gt;"))

    @staticmethod
    def _plain_from_html(text: str) -> str:
        text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
        text = re.sub(r"<[^>]+>", "", text)
        return html.unescape(text).strip()

    @staticmethod
    def _stylesheet() -> str:
        return f"""
            QMainWindow {{ background-color: {Palette.BG_APP}; }}
            QWidget {{ color: {Palette.TEXT_PRIMARY}; }}
            QFrame {{ background-color: transparent; border: none; }}
            QTextEdit {{
                background-color: {Palette.BG_CHAT}; color: {Palette.TEXT_PRIMARY};
                border: none; padding: 10px; font-size: 13px;
            }}
            QLineEdit {{
                background-color: {Palette.BG_INPUT}; color: {Palette.TEXT_PRIMARY};
                border: 1px solid {Palette.BORDER}; border-radius: 10px;
                padding: 8px 11px; font-size: 13px;
                selection-background-color: rgba(167,139,250,0.28);
            }}
            QLineEdit:focus {{ border: 1px solid {Palette.ACCENT}; }}
            QPushButton {{
                background-color: {Palette.ACCENT}; color: #17171C;
                border-radius: 8px; padding: 6px 12px;
                font-weight: 600; font-size: 12px; border: none;
            }}
            QPushButton:hover {{ background-color: #B8A3FF; }}
            QPushButton:pressed {{ background-color: #9274E8; }}
            QPushButton:disabled {{ background-color: {Palette.BG_INPUT}; color: {Palette.TEXT_DIM}; }}
            QPushButton#secondary, QPushButton#secondaryCompact {{
                background-color: {Palette.BG_INPUT}; color: {Palette.TEXT_SECONDARY};
                border: 1px solid {Palette.BORDER};
            }}
            QPushButton#secondary {{
                padding: 6px 12px;
                min-height: 20px;
            }}
            QPushButton#secondaryCompact {{
                padding: 4px 9px;
                min-height: 18px;
                font-size: 11px;
            }}
            QPushButton#secondary:hover, QPushButton#secondaryCompact:hover {{
                background-color: rgba(167,139,250,0.10);
                color: {Palette.TEXT_PRIMARY};
                border-color: {Palette.BORDER_LIGHT};
            }}
            QPushButton#secondaryCompact:checked {{
                background-color: rgba(167,139,250,0.22);
                color: {Palette.TEXT_PRIMARY};
                border-color: {Palette.ACCENT};
            }}
            QPushButton#toolToggleActive {{
                background-color: rgba(167,139,250,0.20);
                color: {Palette.TEXT_PRIMARY};
                border: 1px solid {Palette.ACCENT};
                border-radius: 8px;
                padding: 6px 12px;
                min-height: 20px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton#toolToggleActive:hover {{
                background-color: rgba(167,139,250,0.30);
            }}
            QPushButton#stop_btn:enabled {{
                background-color: rgba(248,113,113,0.20);
                color: {Palette.ACCENT_RED};
                border: 1px solid rgba(248,113,113,0.35);
            }}
            QPushButton#stop_btn:enabled:hover {{ background-color: rgba(248,113,113,0.30); }}
            QPushButton#stop_btn:disabled {{ background-color: {Palette.BG_INPUT}; color: {Palette.TEXT_DIM}; }}
            QPushButton#green_btn {{
                background-color: rgba(110,231,183,0.12);
                color: {Palette.ACCENT_GREEN};
                border: 1px solid rgba(110,231,183,0.28);
            }}
            QPushButton#green_btn:hover {{ background-color: rgba(110,231,183,0.18); }}
            QFrame#workspace_panel {{
                background-color: {Palette.BG_PANEL};
                border: 1px solid {Palette.BORDER};
                border-radius: 12px;
            }}
            QLabel#workspace_title {{
                color: {Palette.TEXT_PRIMARY};
                font-size: 12px;
                font-weight: 700;
            }}
            QLabel#workspace_reason, QLabel#token_label {{
                color: {Palette.TEXT_DIM};
                font-size: 11px;
            }}
            QLabel#workspace_empty {{
                color: {Palette.TEXT_DIM};
                font-size: 13px;
                line-height: 150%;
                background: {Palette.BG_CHAT};
                border: 1px dashed {Palette.BORDER};
                border-radius: 10px;
                padding: 18px;
            }}
            QFrame#agent_progress {{
                background-color: {Palette.BG_PANEL};
                border: 1px solid {Palette.BORDER};
                border-radius: 12px;
            }}
            QFrame#agent_progress[state="blocked"] {{
                border-color: rgba(248,113,113,0.55);
                background-color: rgba(248,113,113,0.07);
            }}
            QFrame#agent_progress[state="finished"] {{
                border-color: rgba(110,231,183,0.45);
                background-color: rgba(110,231,183,0.06);
            }}
            QLabel#agentProgressTitle {{
                color: {Palette.TEXT_PRIMARY};
                font-weight: 700;
                font-size: 12px;
            }}
            QLabel#agentProgressStep {{
                color: {Palette.TEXT_SECONDARY};
                font-size: 12px;
                line-height: 145%;
            }}
            QLabel#agentProgressMeta {{
                color: {Palette.TEXT_DIM};
                font-size: 11px;
            }}
            QLabel#agentProgressDetails {{
                color: {Palette.TEXT_SECONDARY};
                font-size: 11px;
                line-height: 145%;
                background-color: {Palette.BG_INPUT};
                border: 1px solid {Palette.BORDER};
                border-radius: 8px;
                padding: 7px;
            }}
            QPushButton#agentProgressStop {{
                background-color: rgba(248,113,113,0.18);
                color: {Palette.ACCENT_RED};
                border: 1px solid rgba(248,113,113,0.35);
                padding: 4px 9px;
                min-height: 18px;
                font-size: 11px;
            }}
            QPushButton#agentProgressStop:hover {{
                background-color: rgba(248,113,113,0.28);
            }}
            QSplitter::handle {{
                background-color: {Palette.BORDER};
                width: 1px;
                height: 1px;
            }}
            QTreeView {{
                background-color: {Palette.BG_PANEL}; color: {Palette.TEXT_PRIMARY};
                border: none; font-size: 12px;
                outline: none;
            }}
            QTreeView::item {{ padding: 3px 6px; border-radius: 5px; }}
            QTreeView::item:hover {{ background-color: rgba(167,139,250,0.08); }}
            QTreeView::item:selected {{ background-color: rgba(167,139,250,0.18); color: {Palette.TEXT_PRIMARY}; }}
            QMenu {{
                background-color: {Palette.BG_PANEL}; color: {Palette.TEXT_PRIMARY};
                border: 1px solid {Palette.BORDER};
                border-radius: 8px;
            }}
            QMenu::item {{ padding: 6px 18px; }}
            QMenu::item:selected {{ background-color: rgba(167,139,250,0.16); }}
            QTabWidget::pane {{
                border: none;
                background: transparent;
            }}
            QTabBar::tab {{
                background: transparent; color: {Palette.TEXT_DIM};
                padding: 7px 14px; border: none;
            }}
            QTabBar::tab:selected {{
                color: {Palette.TEXT_PRIMARY};
                border-bottom: 2px solid {Palette.ACCENT};
            }}
            QTabBar::tab:hover {{ color: {Palette.TEXT_PRIMARY}; background: rgba(167,139,250,0.08); }}
            QProgressBar {{
                background: {Palette.BG_INPUT}; border-radius: 4px;
                height: 6px; text-align: right;
                font-size: 10px; color: {Palette.TEXT_DIM};
            }}
            QProgressBar::chunk {{ background: {Palette.ACCENT}; border-radius: 4px; }}
            QScrollBar:horizontal, QScrollBar:vertical {{
                background: transparent;
                border: none;
            }}
            QScrollBar:horizontal {{ height: 10px; }}
            QScrollBar:vertical {{ width: 10px; }}
            QScrollBar::handle {{
                background: rgba(167,139,250,0.28);
                border-radius: 5px;
                min-height: 28px;
                min-width: 28px;
            }}
            QScrollBar::handle:hover {{ background: rgba(184,163,255,0.48); }}
            QScrollBar::add-line, QScrollBar::sub-line,
            QScrollBar::add-page, QScrollBar::sub-page {{
                background: transparent;
                border: none;
                width: 0;
                height: 0;
            }}
        """
