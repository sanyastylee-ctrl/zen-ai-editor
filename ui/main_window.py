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

import os
import re
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFileSystemModel, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QFrame, QTreeView, QTabWidget,
    QLabel, QProgressBar, QFileDialog, QMessageBox,
)

from core.profiles import ProfileManager, ProfileKind, AIProfile
from core.model_manager import ModelManager, LLAMA_AVAILABLE
from core.paths import resolve_model_path
from core.token_budget import TokenBudget
from ai.worker import InferenceWorker
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

        self.app_settings = {
            "use_rag": False,
            "diff_before_apply": True,
        }

        self.current_file_path: str = ""
        self.worker: InferenceWorker | None = None

        # стрим текущего ответа (для apply_code_from_chat)
        self._current_ai_response = ""

        # Раздельные истории для разных профилей.
        # Ключ — profile_id, значение — list[(user, assistant)].
        self._histories: dict[str, list[tuple[str, str]]] = {}
        # Лог сырых сообщений для отображения чата (по профилю)
        self._chat_html: dict[str, str] = {}

        self.rag = ProjectRAG()
        self._rag_worker: RagIndexWorker | None = None
        self._rag_chunks = 0

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
        self.file_model.setRootPath(os.getcwd())

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(os.getcwd()))
        self.tree_view.hideColumn(1)
        self.tree_view.hideColumn(2)
        self.tree_view.hideColumn(3)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.doubleClicked.connect(self.open_file)
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

        self.toggle_btn = QPushButton("☰ Проект")
        self.toggle_btn.setObjectName("secondary")
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        top_bar.addWidget(self.toggle_btn)

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

        # --- ШОРТКАТ ---
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_file)

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

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setPlaceholderText("Здесь будет диалог с ИИ...")
        layout.addWidget(self.chat_history, 1)

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

        self.apply_btn = QPushButton("↙ В редактор")
        self.apply_btn.setObjectName("secondary")
        self.apply_btn.setToolTip(
            "Вставить код из последнего ответа.\n"
            "Если в редакторе есть выделение — заменит его, иначе — весь файл."
        )
        self.apply_btn.clicked.connect(self.apply_code_from_chat)
        input_row.addWidget(self.apply_btn)

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
        else:
            self.chat_input.setPlaceholderText("Сообщение... (Enter)")

    def _active_profile(self) -> AIProfile | None:
        pid = self.profile_switcher.active_id()
        return self.pm.get(pid) if pid else None

    # ============================================================
    # ЧАТ
    # ============================================================

    def _refresh_chat_view(self, profile_id: str) -> None:
        self.chat_history.setHtml(self._chat_html.get(profile_id, ""))
        # прокрутка в конец
        sb = self.chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _append_chat(self, profile_id: str, html: str) -> None:
        prev = self._chat_html.get(profile_id, "")
        self._chat_html[profile_id] = prev + html
        if self.profile_switcher.active_id() == profile_id:
            self.chat_history.moveCursor(self.chat_history.textCursor().MoveOperation.End)
            self.chat_history.insertHtml(html)
            sb = self.chat_history.verticalScrollBar()
            sb.setValue(sb.maximum())

    def send_message(self) -> None:
        text = self.chat_input.text().strip()
        if not text:
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

        # рендерим юзер-сообщение
        user_html = f"<div style='margin:6px 0; color:#9CDCFE;'><b>Ты:</b> {self._escape(text)}</div>"
        self._append_chat_active(user_html)

        # подготовка контекста
        code_context = ""
        rag_snippets = ""
        if profile.kind == ProfileKind.CODER:
            code_context = self._get_editor_text()
            if self.attached_files:
                code_context += "\n\n" + self._load_attached_files()
            if self.app_settings.get("use_rag") and self._rag_chunks > 0:
                rag_snippets = self.rag.search(text, top_k=5)

        # вступление для ответа
        speaker = profile.name if profile.kind == ProfileKind.COMPANION else "Ассистент"
        self._append_chat_active(f"<div style='margin:6px 0; color:#CE9178;'><b>{speaker}:</b> </div>")
        self._current_ai_response = ""

        # история профиля
        history = self._histories.get(profile.id, [])

        self.worker = InferenceWorker(
            profile=profile,
            user_message=text,
            code_context=code_context,
            rag_snippets=rag_snippets,
            history=history,
            user_name=profile.persona.get("user_name", "") if profile.kind == ProfileKind.COMPANION else "",
        )
        self.worker.chunk_received.connect(self._on_chunk)
        self.worker.finished_signal.connect(lambda: self._on_generation_done(profile.id, text))
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

    def _on_chunk(self, chunk: str) -> None:
        self._current_ai_response += chunk
        # экранируем HTML, чтобы стрим текст не ломал разметку
        esc = self._escape(chunk)
        self.chat_history.moveCursor(self.chat_history.textCursor().MoveOperation.End)
        self.chat_history.insertHtml(esc.replace("\n", "<br>"))
        sb = self.chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_generation_done(self, profile_id: str, user_msg: str) -> None:
        # дописываем разделитель в кэш-html
        end_html = "<br><br>"
        self._chat_html[profile_id] = self._chat_html.get(profile_id, "") + end_html
        if self.profile_switcher.active_id() == profile_id:
            self.chat_history.insertHtml(end_html)

        # пушим в историю
        hist = self._histories.setdefault(profile_id, [])
        hist.append((user_msg, self._current_ai_response))

        # лимит истории, чтобы не разрасталось бесконечно (бюджет токенов всё равно обрежет)
        if len(hist) > 50:
            del hist[: len(hist) - 50]

        self._current_ai_response = ""
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
        # пользователь увидит "Загружаю..." в следующем _on_model_load_start
        pass

    # ============================================================
    # КОД И ВСТАВКА В РЕДАКТОР
    # ============================================================

    def apply_code_from_chat(self) -> None:
        """Извлекает код из последнего ответа и вставляет в редактор."""
        if not self._current_ai_response:
            # после finish мы обнуляем response — берём из истории
            pid = self.profile_switcher.active_id()
            if pid and self._histories.get(pid):
                _, last_response = self._histories[pid][-1]
            else:
                return
        else:
            last_response = self._current_ai_response

        # ищем код в ```язык ... ``` блоках, либо весь текст если их нет
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", last_response, re.DOTALL)
        if code_blocks:
            new_code = code_blocks[0].strip()
        else:
            new_code = last_response.strip()

        old_code = self._get_editor_text()
        has_selection = self._editor_has_selection()

        # diff превью при замене всего файла
        if self.app_settings.get("diff_before_apply", True) and not has_selection and old_code.strip():
            dlg = DiffApplyDialog(old_code, new_code, parent=self)
            if dlg.exec() != dlg.DialogCode.Accepted:
                return
            new_code = dlg.accepted_code

        if has_selection:
            self._replace_selection(new_code)
        else:
            self._set_editor_text(new_code)

    # ============================================================
    # ФАЙЛЫ И РЕДАКТОР
    # ============================================================

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
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить как", "", "Все файлы (*.*)")
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
        # переключиться на вкладку терминала
        self._editor_tabs.setCurrentIndex(1)
        self.sandbox.set_cwd(os.path.dirname(self.current_file_path))
        self.sandbox.run_code_file(self.current_file_path)

    def attach_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Прикрепить файлы", "", "Все файлы (*.*)")
        if not paths:
            return
        self.attached_files = paths
        names = ", ".join(os.path.basename(p) for p in paths)
        self.attached_label.setText(f"📎 Прикреплено: {names}")
        self.attached_label.setVisible(True)

    def _load_attached_files(self) -> str:
        parts = []
        for p in self.attached_files:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    parts.append(f"### {os.path.basename(p)}\n{f.read()}")
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
        self._rag_worker = RagIndexWorker(self.rag, os.getcwd())
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

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.pm, self.app_settings, parent=self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            self.app_settings.update(dlg.get_app_settings())
            # перестроить свитчер (могли добавиться новые профили или удалиться)
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
    def _stylesheet() -> str:
        return """
            QMainWindow { background-color: #1E1E1E; }
            QFrame { background-color: #252526; border: none; }
            QTextEdit {
                background-color: #1E1E1E; color: #D4D4D4;
                border: none; padding: 10px; font-size: 14px;
            }
            QLineEdit {
                background-color: #3C3C3C; color: #FFFFFF;
                border: 1px solid #555; border-radius: 6px;
                padding: 8px 10px; font-size: 13px;
            }
            QPushButton {
                background-color: #0E639C; color: white;
                border-radius: 6px; padding: 7px 14px;
                font-weight: 500; font-size: 12px; border: none;
            }
            QPushButton:hover { background-color: #1177BB; }
            QPushButton:disabled { background-color: #3A3A3A; color: #666; }
            QPushButton#secondary {
                background-color: #3A3A3A; color: #D4D4D4;
                border: 1px solid #4A4A4A;
            }
            QPushButton#secondary:hover { background-color: #4A4A4A; }
            QPushButton#stop_btn:enabled { background-color: #8B2020; color: white; }
            QPushButton#stop_btn:enabled:hover { background-color: #A52828; }
            QPushButton#stop_btn:disabled { background-color: #3A3A3A; color: #666; }
            QPushButton#green_btn {
                background-color: #1B4F1B; color: #4EC9B0;
                border: 1px solid #2A7A2A;
            }
            QPushButton#green_btn:hover { background-color: #236B23; }
            QSplitter::handle { background-color: #333; width: 2px; }
            QTreeView {
                background-color: #252526; color: #CCC;
                border: none; font-size: 13px;
            }
            QTreeView::item:hover { background-color: #2A2D2E; }
            QTreeView::item:selected { background-color: #37373D; }
            QTabWidget::pane { border: none; background: #1E1E1E; }
            QTabBar::tab {
                background: #2D2D2D; color: #888;
                padding: 6px 16px; border: none;
            }
            QTabBar::tab:selected {
                background: #1E1E1E; color: #D4D4D4;
                border-bottom: 2px solid #0E639C;
            }
            QTabBar::tab:hover { background: #3C3C3C; color: #D4D4D4; }
            QProgressBar {
                background: #3C3C3C; border-radius: 3px;
                height: 6px; text-align: right;
                font-size: 10px; color: #888;
            }
            QProgressBar::chunk { background: #0E639C; border-radius: 3px; }
            QLabel#token_label { color: #888; font-size: 11px; }
        """
