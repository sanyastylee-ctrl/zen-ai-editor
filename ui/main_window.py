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
    QLabel, QProgressBar, QFileDialog, QMessageBox, QMenu
)

from core.profiles import ProfileManager, ProfileKind, AIProfile
from core.model_manager import ModelManager, LLAMA_AVAILABLE
from core.paths import resolve_model_path
from core.token_budget import TokenBudget
from core.projects import ProjectManager
from core.settings import PersistentSettings
from ai.agent import AgentWorker
from ai.worker import InferenceWorker
from ui.chat import ChatView
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
        self._stream_update_timer = QTimer(self)
        self._stream_update_timer.setSingleShot(True)
        self._stream_update_timer.setInterval(50)
        self._stream_update_timer.timeout.connect(self._flush_stream_update)

        # Раздельные истории для разных профилей.
        # Ключ — profile_id, значение — list[(user, assistant)].
        self._histories: dict[str, list[tuple[str, str]]] = {}
        # Визуальные записи чата для ChatView (по профилю).
        self._chat_records: dict[str, list[dict]] = {}

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
        first_active = active_coder or active_companion
        if first_active:
            self.profile_switcher.set_profiles(self.pm.all(), first_active.id)
            self._refresh_chat_view(first_active.id)
        self._update_token_bar()

        # Заголовок и статус с именем проекта
        self._update_window_title()

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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

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

        top_bar.addStretch()

        # переключатель профилей
        self.profile_switcher = ProfileSwitcher()
        self.profile_switcher.profile_changed.connect(self._on_profile_changed)
        top_bar.addWidget(self.profile_switcher)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setObjectName("secondary")
        self.settings_btn.setFixedWidth(40)
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
        self.token_bar.setMaximumWidth(180)
        status_row.addWidget(self.token_bar)
        layout.addLayout(status_row)

        # --- ШОРТКАТЫ ---
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_file)
        QShortcut(QKeySequence("Ctrl+Shift+O"), self).activated.connect(self.open_project_dialog)

        # --- WORK SPLITTER ---
        self.work_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.work_splitter, 1)

        # чат
        self.work_splitter.addWidget(self._build_chat_zone())
        # редактор + терминал в табах
        self.work_splitter.addWidget(self._build_editor_zone())
        self.work_splitter.setSizes([420, 880])

        return right

    def _build_chat_zone(self) -> QWidget:
        zone = QWidget()
        layout = QVBoxLayout(zone)
        layout.setContentsMargins(0, 0, 10, 0)

        self.chat_view = ChatView()
        self.chat_view.insert_requested.connect(self._insert_code_from_chat)
        layout.addWidget(self.chat_view, 1)

        # ввод
        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.attach_btn = QPushButton("📎")
        self.attach_btn.setObjectName("secondary")
        self.attach_btn.setFixedWidth(40)
        self.attach_btn.setToolTip("Прикрепить файл к запросу")
        self.attach_btn.clicked.connect(self.attach_file)
        input_row.addWidget(self.attach_btn)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Сообщение... (Enter)")
        self.chat_input.returnPressed.connect(self.send_message)
        self.chat_input.textChanged.connect(self._update_token_bar)
        input_row.addWidget(self.chat_input)

        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setFixedWidth(40)
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

        tabs.addTab(editor_widget, "📝 Редактор")

        self.sandbox = SandboxWidget()
        tabs.addTab(self.sandbox, "⚡ Терминал")

        self._editor_tabs = tabs
        return tabs

    # ============================================================
    # PROFILE
    # ============================================================

    def _on_profile_changed(self, profile_id: str) -> None:
        """Юзер кликнул на другой профиль в свитчере."""
        profile = self.pm.get(profile_id)
        if not profile:
            return

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
        elif profile.kind == ProfileKind.VISION:
            self.chat_input.setPlaceholderText("Опишите, что хотите узнать о картинке (Enter)")
        else:
            self.chat_input.setPlaceholderText("Сообщение... (Enter)")

    def _active_profile(self) -> AIProfile | None:
        pid = self.profile_switcher.active_id()
        return self.pm.get(pid) if pid else None

    # ============================================================
    # ЧАТ И КОНТЕКСТ
    # ============================================================

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

    def _update_chat_record(self, profile_id: str, record: dict) -> None:
        if self.profile_switcher.active_id() == profile_id:
            self.chat_view.update_record(record)

    def send_message(self) -> None:
        text = self.chat_input.text().strip()
        
        if not text and not self.attached_files:
            return
            
        if not LLAMA_AVAILABLE:
            self._append_chat_active("<b style='color:#CE9178;'>llama-cpp-python не установлен</b><br>")
            return

        profile = self._active_profile()
        if not profile:
            self._append_chat_active("<b style='color:#CE9178;'>Не выбран профиль</b><br>")
            return
        if not profile.model_file:
            self._append_chat_active("<b style='color:#CE9178;'>Не выбрана модель в Настройках</b><br>")
            return

        # Проверка: есть ли картинки и подходит ли профиль
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        has_images = any(
            os.path.splitext(p)[1].lower() in image_exts
            for p in self.attached_files
        )
        if has_images and profile.kind != ProfileKind.VISION:
            # ищем Vision-профиль
            vision_profiles = self.pm.by_kind(ProfileKind.VISION)
            ready_vision = next(
                (p for p in vision_profiles if p.model_file and p.mmproj_file),
                None,
            )
            if ready_vision:
                btn = QMessageBox.question(
                    self, "Прикреплена картинка",
                    f"Активный профиль «{profile.name}» не понимает изображения.\n"
                    f"Переключиться на Vision-профиль «{ready_vision.name}»?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if btn == QMessageBox.StandardButton.Yes:
                    self.profile_switcher.set_active(ready_vision.id)
                    self._on_profile_changed(ready_vision.id)
                    profile = ready_vision
            else:
                self._append_chat_active(
                    "<i style='color:#CE9178;'>⚠ Прикреплены картинки, но нет настроенного Vision-профиля. "
                    "Создайте профиль типа Vision в Настройках и выберите файл модели + mmproj.</i><br>"
                )

        # --- ОБРАБОТКА ВЛОЖЕНИЙ ---
        attachment_text = ""
        attached_paths = list(self.attached_files) 
        
        if self.attached_files:
            attachment_text = self._load_attached_files()
            self.attached_files = []
            self._update_attached_label()

        full_text = text
        if attachment_text and profile.kind == ProfileKind.COMPANION:
            full_text = f"Вложенные файлы:\n{attachment_text}\n\n{text}".strip()

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

        history = self._histories.get(profile.id, [])

        if profile.kind == ProfileKind.CODER and getattr(profile, "agent_mode", False):
            self.worker = AgentWorker(
                profile=profile,
                user_message=full_text,
                code_context=code_context,
                history=history,
                project_root=self.projects.current,
                terminal_history=self._terminal_history,
                confirmation_policy=self.app_settings.get(
                    "agent_confirmation_policy", "confirm_changes"
                ),
            )
            self.worker.tool_started.connect(self._on_agent_tool_started)
            self.worker.tool_finished.connect(self._on_agent_tool_finished)
            self.worker.confirmation_requested.connect(self._on_agent_confirmation_requested)
        else:
            self.worker = InferenceWorker(
                profile=profile,
                user_message=full_text,
                code_context=code_context,
                rag_snippets=rag_snippets,
                history=history,
                user_name=profile.persona.get("user_name", "") if profile.kind == ProfileKind.COMPANION else "",
                attached_files=attached_paths,
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

        self.chat_input.clear()
        self.stop_btn.setEnabled(True)
        self.worker.start()

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

    def _on_generation_done(self, profile_id: str, user_msg: str) -> None:
        if self._stream_update_timer.isActive():
            self._stream_update_timer.stop()
        if self._current_assistant_record is not None:
            self._current_assistant_record["text"] = self._current_message_buffer
            self._current_assistant_record["streaming"] = False
            self._update_chat_record(profile_id, self._current_assistant_record)

        hist = self._histories.setdefault(profile_id, [])
        hist.append((user_msg, self._current_ai_response))

        if len(hist) > 50:
            del hist[: len(hist) - 50]

        self._current_ai_response = ""
        self._current_message_buffer = ""
        self._current_response_profile_id = ""
        self._current_assistant_record = None
        self._current_assistant_sender = "Ассистент"
        self._current_assistant_kind = ""
        self.stop_btn.setEnabled(False)
        self.worker = None

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
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось открыть {path}: {e}")
                return
            self._set_editor_text(content)
            self.current_file_path = path
            if QSCI_AVAILABLE:
                set_lexer_for_file(self.code_editor, path)
            self.sandbox.set_cwd(os.path.dirname(path))

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
        self._editor_tabs.setCurrentIndex(1)
        self.sandbox.set_cwd(os.path.dirname(self.current_file_path))
        self.sandbox.run_code_file(self.current_file_path)

    def attach_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Прикрепить файлы", self.projects.current, "Все файлы (*.*)"
        )
        if not paths:
            return
        for p in paths:
            if p not in self.attached_files:
                self.attached_files.append(p)
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
            self.profile_switcher.set_profiles(profiles, current)
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
        return """
            QMainWindow { background-color: #1A1A1F; }
            QFrame { background-color: #22232A; border: none; }
            QTextEdit {
                background-color: #1A1A1F; color: #E7E8EE;
                border: none; padding: 10px; font-size: 14px;
            }
            QLineEdit {
                background-color: #2A2D38; color: #FFFFFF;
                border: 1px solid #3B3D48; border-radius: 10px;
                padding: 8px 10px; font-size: 13px;
            }
            QPushButton {
                background-color: #A78BFA; color: #17171C;
                border-radius: 10px; padding: 7px 14px;
                font-weight: 500; font-size: 12px; border: none;
            }
            QPushButton:hover { background-color: #B8A3FF; }
            QPushButton:disabled { background-color: #3B3D48; color: #777A88; }
            QPushButton#secondary {
                background-color: #3B3D48; color: #E7E8EE;
                border: 1px solid #4A4D5C;
            }
            QPushButton#secondary:hover { background-color: #4A4D5C; }
            QPushButton#stop_btn:enabled { background-color: #8B2020; color: white; }
            QPushButton#stop_btn:enabled:hover { background-color: #A52828; }
            QPushButton#stop_btn:disabled { background-color: #3B3D48; color: #777A88; }
            QPushButton#green_btn {
                background-color: #24443B; color: #6EE7B7;
                border: 1px solid #2F6B58;
            }
            QPushButton#green_btn:hover { background-color: #2F5F50; }
            QSplitter::handle { background-color: #30313A; width: 2px; }
            QTreeView {
                background-color: #22232A; color: #E7E8EE;
                border: none; font-size: 13px;
            }
            QTreeView::item:hover { background-color: #2A2D2E; }
            QTreeView::item:selected { background-color: #37373D; }
            QMenu {
                background-color: #22232A; color: #E7E8EE;
                border: 1px solid #343642;
            }
            QMenu::item:selected {
                background-color: #4A3D73;
            }
            QTabWidget::pane { border: none; background: #1A1A1F; }
            QTabBar::tab {
                background: #2A2D38; color: #A4A7B5;
                padding: 6px 16px; border: none;
            }
            QTabBar::tab:selected {
                background: #1A1A1F; color: #E7E8EE;
                border-bottom: 2px solid #A78BFA;
            }
            QTabBar::tab:hover { background: #3B3D48; color: #E7E8EE; }
            QProgressBar {
                background: #3B3D48; border-radius: 4px;
                height: 6px; text-align: right;
                font-size: 10px; color: #A4A7B5;
            }
            QProgressBar::chunk { background: #A78BFA; border-radius: 4px; }
            QLabel#token_label { color: #A4A7B5; font-size: 11px; }
        """
