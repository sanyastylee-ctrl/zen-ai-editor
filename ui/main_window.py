import os
import re
import json
from PyQt6.QtWidgets import (QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QLineEdit, QPushButton, QFrame, QTreeView,
                             QComboBox, QFileDialog, QProgressBar, QLabel, QMessageBox,
                             QTabWidget, QSizePolicy)
from PyQt6.QtGui import QFont, QTextCursor, QShortcut, QKeySequence, QFileSystemModel
from PyQt6.QtCore import Qt, QTimer

from core.settings import PersistentSettings
from core.session import SessionManager
from core.tokens import TokenBudgetManager
from ai.model_manager import Llama
from ai.workers import LlamaCppWorker
from rag.indexer import ProjectRAG, RagIndexWorker, RAG_AVAILABLE
from sandbox.terminal import SandboxWidget
from comfy.image_gen import ComfyUIWorker
from widgets.tabs import EditorTabWidget
from widgets.find_replace import FindReplaceBar
from widgets.cheat_sheet import CheatSheetDialog
from widgets.diff_dialog import DiffApplyDialog
from ui.settings_dialog import SettingsDialog

class ZenEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zen AI Editor")
        self.resize(1400, 900)
        self.setMinimumSize(900, 600)

        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QFrame { background-color: #252526; border: none; }
            QTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: none; padding: 10px; font-size: 14px; }
            QLineEdit { background-color: #3C3C3C; color: #FFFFFF; border: 1px solid #555555; border-radius: 6px; padding: 10px; font-size: 14px; }
            QPushButton { background-color: #0E639C; color: white; border-radius: 6px; padding: 8px 15px; font-weight: bold; font-size: 13px; border: none; }
            QPushButton:hover { background-color: #1177BB; }
            QPushButton:pressed { background-color: #0A4F7E; }
            QPushButton:disabled { background-color: #3A3A3A; color: #666666; }
            QPushButton#secondary { background-color: #3C3C3C; color: #D4D4D4; border: 1px solid #555555; }
            QPushButton#secondary:hover { background-color: #4A4A4A; }
            QPushButton#secondary:disabled { background-color: #2E2E2E; color: #555555; }
            QPushButton#stop_btn:enabled { background-color: #8B2020; color: white; }
            QPushButton#stop_btn:enabled:hover { background-color: #A52828; }
            QPushButton#stop_btn:disabled { background-color: #3A3A3A; color: #666666; }
            QPushButton#green_btn { background-color: #1B4F1B; color: #4EC9B0; border: 1px solid #2A7A2A; }
            QPushButton#green_btn:hover { background-color: #236B23; }
            QComboBox { background-color: #3C3C3C; color: #FFFFFF; border: 1px solid #555555; border-radius: 6px; padding: 5px 10px; font-size: 13px; min-width: 150px; }
            QComboBox QAbstractItemView { background-color: #2D2D2D; color: #FFFFFF; selection-background-color: #0E639C; }
            QSplitter::handle { background-color: #333333; width: 2px; }
            QTreeView { background-color: #252526; color: #CCCCCC; border: none; font-size: 13px; }
            QTreeView::item:hover { background-color: #2A2D2E; }
            QTreeView::item:selected { background-color: #37373D; }
            QTabWidget::pane { border: none; background: #1E1E1E; }
            QTabBar::tab { background: #2D2D2D; color: #888888; padding: 6px 16px; border: none; min-width: 80px; }
            QTabBar::tab:selected { background: #1E1E1E; color: #D4D4D4; border-bottom: 2px solid #0E639C; }
            QTabBar::tab:hover { background: #3C3C3C; color: #D4D4D4; }
            QTabBar::close-button { image: none; }
            QProgressBar { background: #3C3C3C; border-radius: 3px; height: 6px; text-align: right; font-size: 10px; color: #888888; }
            QProgressBar::chunk { background: #0E639C; border-radius: 3px; }
            QLabel#token_label { color: #888888; font-size: 11px; }
            QLabel#token_warn  { color: #CE9178; font-size: 11px; }
            QScrollBar:vertical { background: #1E1E1E; width: 8px; border: none; }
            QScrollBar::handle:vertical { background: #424242; border-radius: 4px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #555555; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        self.app_settings = PersistentSettings.load()
        self.attached_files_content = ""
        self.available_models       = []
        self.worker                 = None
        self._chat_raw_log          = []
        self._current_ai_response   = ""
        self._comfy_worker          = None

        self.token_budget = TokenBudgetManager(n_ctx=4096, response_reserve=2048)
        self.rag          = ProjectRAG()
        self._rag_indexed = False
        self._rag_worker  = None

        self._build_ui()
        self.scan_local_models() 
        loaded = self.rag.load_index()
        if loaded > 0:
            self._rag_indexed = True
            self.rag_status_label.setText(f"RAG: {loaded} чанков (кэш)")

        self._settings_timer = QTimer(self)
        self._settings_timer.timeout.connect(lambda: PersistentSettings.save(self.app_settings))
        self._settings_timer.start(60_000)

        self._recovery_timer = QTimer(self)
        self._recovery_timer.timeout.connect(self._save_session)
        self._recovery_timer.start(30_000)

        self._try_restore_session()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)

        self.sidebar = QFrame()
        sb_layout = QVBoxLayout(self.sidebar)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.setSpacing(4)

        rag_row = QHBoxLayout()
        self.index_btn = QPushButton("⟳ Индекс RAG")
        self.index_btn.setObjectName("green_btn")
        self.index_btn.clicked.connect(self.start_rag_indexing)
        rag_row.addWidget(self.index_btn)
        self.rag_status_label = QLabel("RAG: –")
        self.rag_status_label.setObjectName("token_label")
        rag_row.addWidget(self.rag_status_label)
        sb_layout.addLayout(rag_row)

        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(os.getcwd())
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(os.getcwd()))
        self.tree_view.hideColumn(1); self.tree_view.hideColumn(2); self.tree_view.hideColumn(3)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.doubleClicked.connect(self.open_file)
        sb_layout.addWidget(self.tree_view)
        self.sidebar.hide()

        right_zone = QWidget()
        right_layout = QVBoxLayout(right_zone)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(6)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)
        
        self.toggle_btn = QPushButton("☰ Проект")
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        top_bar.addWidget(self.toggle_btn)

        self.open_folder_btn = QPushButton("📁 Открыть папку")
        self.open_folder_btn.clicked.connect(self.open_project_folder)
        top_bar.addWidget(self.open_folder_btn)

        self.save_btn = QPushButton("💾 Сохранить")
        self.save_btn.clicked.connect(self.save_file)
        top_bar.addWidget(self.save_btn)

        self.run_file_btn = QPushButton("▶ Запустить")
        self.run_file_btn.setObjectName("green_btn")
        self.run_file_btn.clicked.connect(self.run_current_file)
        top_bar.addWidget(self.run_file_btn)

        top_bar.addStretch()

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Режим: Кодер", "Режим: Ассистент"])
        self.mode_selector.currentIndexChanged.connect(self._update_token_bar)
        top_bar.addWidget(self.mode_selector)

        self.settings_btn = QPushButton("⚙ Настройки")
        self.settings_btn.clicked.connect(self.open_settings)
        top_bar.addWidget(self.settings_btn)

        help_btn = QPushButton("?")
        help_btn.setObjectName("secondary")
        help_btn.setFixedWidth(32)
        help_btn.clicked.connect(self.show_cheat_sheet)
        top_bar.addWidget(help_btn)

        right_layout.addLayout(top_bar)

        budget_row = QHBoxLayout()
        self.token_label = QLabel("Токены: 0 / 4096")
        self.token_label.setObjectName("token_label")
        budget_row.addWidget(self.token_label)
        self.token_bar = QProgressBar()
        self.token_bar.setRange(0, 100); self.token_bar.setValue(0)
        self.token_bar.setFormat("%p%")
        budget_row.addWidget(self.token_bar)
        right_layout.addLayout(budget_row)

        QShortcut(QKeySequence("Ctrl+S"),      self).activated.connect(self.save_file)
        QShortcut(QKeySequence("Ctrl+R"),      self).activated.connect(self.run_current_file)
        QShortcut(QKeySequence("Ctrl+T"),      self).activated.connect(self._new_editor_tab)
        QShortcut(QKeySequence("Ctrl+W"),      self).activated.connect(self._close_current_editor_tab)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.send_message)
        QShortcut(QKeySequence("Ctrl+L"),      self).activated.connect(lambda: self.chat_input.clear())
        QShortcut(QKeySequence("Ctrl+1"),      self).activated.connect(lambda: self.mode_selector.setCurrentIndex(0))
        QShortcut(QKeySequence("Ctrl+2"),      self).activated.connect(lambda: self.mode_selector.setCurrentIndex(1))

        self.work_splitter = QSplitter(Qt.Orientation.Horizontal)
        right_layout.addWidget(self.work_splitter)

        # --- Чат ---
        chat_zone = QWidget()
        chat_layout = QVBoxLayout(chat_zone)
        chat_layout.setContentsMargins(0, 0, 10, 0)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setPlaceholderText("Здесь будет диалог с ИИ...")

        # ВЕРХНИЙ РЯД КНОПОК ЧАТА (Вспомогательные действия)
        chat_tools_layout = QHBoxLayout()
        chat_tools_layout.setSpacing(6)
        
        self.attach_btn = QPushButton("📎 Прикрепить")
        self.attach_btn.setObjectName("secondary")
        self.attach_btn.setToolTip("Прикрепить файл к контексту")
        self.attach_btn.clicked.connect(self.attach_file)
        chat_tools_layout.addWidget(self.attach_btn)

        self.clear_chat_btn = QPushButton("🗑 Очистить")
        self.clear_chat_btn.setObjectName("secondary")
        self.clear_chat_btn.clicked.connect(self.clear_chat)
        chat_tools_layout.addWidget(self.clear_chat_btn)

        self.export_md_btn = QPushButton("⬇ Сохранить .md")
        self.export_md_btn.setObjectName("secondary")
        self.export_md_btn.clicked.connect(self.export_chat_md)
        chat_tools_layout.addWidget(self.export_md_btn)
        
        # Заставляем кнопки сжаться влево
        chat_tools_layout.addStretch()

        self.apply_btn = QPushButton("↙ Вставить код")
        self.apply_btn.setObjectName("secondary")
        self.apply_btn.setToolTip("Применить последний блок кода из чата.")
        self.apply_btn.clicked.connect(self.apply_code_from_chat)
        chat_tools_layout.addWidget(self.apply_btn)

        # НИЖНИЙ РЯД (Поле ввода и кнопка Стоп)
        input_layout = QHBoxLayout()
        input_layout.setSpacing(6)
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Спроси ИИ или дай задачу... (Enter)")
        self.chat_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.chat_input.returnPressed.connect(self.send_message)
        self.chat_input.textChanged.connect(self._update_token_bar)
        input_layout.addWidget(self.chat_input)

        self.stop_btn = QPushButton("⏹ Стоп")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)
        input_layout.addWidget(self.stop_btn)

        # Собираем чат
        chat_layout.addWidget(self.chat_history)
        chat_layout.addLayout(chat_tools_layout) # Сначала тулбар
        chat_layout.addLayout(input_layout)      # Под ним строка ввода

        right_panel = QTabWidget()
        right_panel.setTabPosition(QTabWidget.TabPosition.South)

        editor_container = QWidget()
        ec_layout = QVBoxLayout(editor_container)
        ec_layout.setContentsMargins(0, 0, 0, 0)
        ec_layout.setSpacing(0)

        self.editor_tabs = EditorTabWidget()
        self.editor_tabs.context_changed.connect(self._update_token_bar)
        ec_layout.addWidget(self.editor_tabs)

        self.find_bar = FindReplaceBar(self.editor_tabs)
        ec_layout.addWidget(self.find_bar)

        right_panel.addTab(editor_container, "📝 Редактор")

        self.sandbox = SandboxWidget()
        right_panel.addTab(self.sandbox, "⚡ Терминал")

        self.work_splitter.addWidget(chat_zone)
        self.work_splitter.addWidget(right_panel)
        self.work_splitter.setSizes([420, 880])
        self.main_splitter.addWidget(self.sidebar)
        self.main_splitter.addWidget(right_zone)
        self.main_splitter.setSizes([250, 1150])

        self._right_panel = right_panel

        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(lambda: self.find_bar.toggle(False))
        QShortcut(QKeySequence("Ctrl+H"), self).activated.connect(lambda: self.find_bar.toggle(True))
        QShortcut(QKeySequence("F1"), self).activated.connect(self.show_cheat_sheet)

    def open_project_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку проекта", os.getcwd())
        if folder:
            os.chdir(folder)
            self.file_model.setRootPath(folder)
            self.tree_view.setRootIndex(self.file_model.index(folder))
            self.sandbox.set_cwd(folder)
            self.chat_history.append(f"<i style='color:#4EC9B0;'>📁 Открыт проект: {folder}</i><br>")

    def scan_local_models(self):
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_dir): os.makedirs(models_dir)
        
        if Llama is None:
            self.chat_history.append("<b style='color:#CE9178;'>Внимание: llama-cpp-python не установлена.</b><br>")
            return
            
        self.available_models = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]
        
        if self.available_models:
            if not self.app_settings.get('coder_model'):
                self.app_settings['coder_model'] = self.available_models[0]
            if not self.app_settings.get('assistant_model'):
                self.app_settings['assistant_model'] = self.available_models[0]
        else:
            self.chat_history.append("<b style='color:#CE9178;'>Положите .gguf файлы в папку /models и перезапустите.</b><br>")

    def open_settings(self):
        self.scan_local_models() 
        dialog = SettingsDialog(self, self.app_settings, self.available_models)
        if dialog.exec():
            self.app_settings = dialog.get_settings()
            self.token_budget = TokenBudgetManager(n_ctx=self.app_settings['n_ctx'], response_reserve=self.app_settings['max_tokens'])
            self._update_token_bar()
            PersistentSettings.save(self.app_settings)
            self.chat_history.append("<i style='color:#888888;'>Настройки сохранены.</i><br>")

    def _new_editor_tab(self):
        self.editor_tabs.new_tab()

    def _close_current_editor_tab(self):
        idx = self.editor_tabs.currentIndex()
        self.editor_tabs._close_tab(idx)

    def _get_editor_text(self) -> str:
        return self.editor_tabs.get_text()

    def _set_editor_text(self, text: str, undo_safe: bool = False):
        self.editor_tabs.set_text(text, undo_safe=undo_safe)

    def _replace_selection(self, text: str):
        self.editor_tabs.replace_selection(text)

    @property
    def current_file_path(self) -> str:
        return self.editor_tabs.current_file_path()

    def _update_token_bar(self):
        prompt    = self.chat_input.text() if hasattr(self, 'chat_input') else ""
        code_ctx  = self._get_editor_text()
        mode      = "coder" if "Кодер" in self.mode_selector.currentText() else "assistant"
        sys_p     = self.app_settings.get('coder_system_prompt', '') if mode == "coder" else self.app_settings.get('system_prompt', '')

        self.token_budget = TokenBudgetManager(n_ctx=self.app_settings.get('n_ctx', 4096), response_reserve=self.app_settings.get('max_tokens', 2048))
        pct  = self.token_budget.get_usage_pct(code_ctx, prompt, sys_p)
        used = self.token_budget.estimate_tokens(sys_p + prompt + code_ctx)
        n_ctx= self.app_settings.get('n_ctx', 4096)

        self.token_bar.setValue(pct)
        self.token_label.setText(f"Токены: ~{used} / {n_ctx}")
        if pct >= 90:
            self.token_bar.setStyleSheet("QProgressBar::chunk { background: #CE4040; border-radius: 3px; }")
            self.token_label.setObjectName("token_warn")
        elif pct >= 70:
            self.token_bar.setStyleSheet("QProgressBar::chunk { background: #CDA040; border-radius: 3px; }")
            self.token_label.setObjectName("token_label")
        else:
            self.token_bar.setStyleSheet("")
            self.token_label.setObjectName("token_label")
        self.token_label.style().unpolish(self.token_label)
        self.token_label.style().polish(self.token_label)

    def start_rag_indexing(self):
        if not RAG_AVAILABLE:
            self.chat_history.append("<b style='color:#CE9178;'>RAG недоступен: установите faiss-cpu и sentence-transformers.</b><br>")
            return
        self.index_btn.setEnabled(False)
        self.rag_status_label.setText("RAG: индексация...")
        self._rag_worker = RagIndexWorker(self.rag, os.getcwd())
        self._rag_worker.finished_signal.connect(self._on_rag_indexed)
        self._rag_worker.error_signal.connect(self._on_rag_error)
        self._rag_worker.start()

    def _on_rag_indexed(self, count):
        self._rag_indexed = count > 0
        self.rag_status_label.setText(f"RAG: {count} чанков")
        self.index_btn.setEnabled(True)
        self.chat_history.append(f"<i style='color:#4EC9B0;'>✓ Проект проиндексирован: {count} фрагментов.</i><br>")

    def _on_rag_error(self, err):
        self.index_btn.setEnabled(True)
        self.rag_status_label.setText("RAG: ошибка")
        self.chat_history.append(f"<b style='color:#CE9178;'>RAG ошибка: {err}</b><br>")

    def attach_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Прикрепить файл")
        if fp:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    self.attached_files_content += f"\n--- Файл: {os.path.basename(fp)} ---\n"
                    self.attached_files_content += f.read()[:5000]
                self.chat_history.append(f"<i style='color:#888888;'>📎 {os.path.basename(fp)} загружен.</i><br>")
            except Exception as e:
                self.chat_history.append(f"<b style='color:#CE9178;'>Ошибка чтения:</b> {str(e)}<br>")

    def toggle_sidebar(self):
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def stop_generation(self):
        if self.worker and self.worker.isRunning(): self.worker.stop()

    def save_file(self):
        if not self.editor_tabs.current_file_path():
            self.save_file_as()
            return
        ok, result = self.editor_tabs.save_current()
        if ok:
            self.chat_history.append(f"<i style='color:#888888;'>Сохранено: {os.path.basename(result)}</i><br>")
            self.setWindowTitle(f"Zen AI Editor — {os.path.basename(result)}")
        else:
            self.chat_history.append(f"<b style='color:#CE9178;'>Ошибка сохранения:</b> {result}<br>")

    def save_file_as(self):
        fp, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", os.getcwd(), "Python Files (*.py);;All Files (*)")
        if fp:
            self.editor_tabs.set_current_path(fp)
            self.save_file()

    def open_file(self, index):
        fp = self.file_model.filePath(index)
        if not os.path.isfile(fp): return
        try:
            with open(fp, 'r', encoding='utf-8') as f: content = f.read()
            self.editor_tabs.open_file_tab(fp, content)
            self.setWindowTitle(f"Zen AI Editor — {os.path.basename(fp)}")
            self.sandbox.set_cwd(os.path.dirname(fp))
            self._update_token_bar()
        except Exception as e:
            self.chat_history.append(f"<b style='color:#CE9178;'>Ошибка:</b> {str(e)}")

    def run_current_file(self):
        fp = self.editor_tabs.current_file_path()
        if fp and fp.endswith('.py'):
            self._right_panel.setCurrentIndex(1) 
            self.sandbox.set_cwd(os.path.dirname(fp))
            self.sandbox.run_code_file(fp)
        elif fp:
            self.chat_history.append("<i style='color:#888;'>Запуск поддерживается только для .py файлов.</i><br>")
        else:
            self.chat_history.append("<i style='color:#888;'>Сначала сохраните файл (Ctrl+S).</i><br>")

    def _save_session(self):
        SessionManager.save(self.editor_tabs)

    def _try_restore_session(self):
        session = SessionManager.load()
        if not session or not session.get('tabs'): return
        has_unsaved = any(t.get('recovery') for t in session['tabs'])
        if not has_unsaved: return
        reply = QMessageBox.question(
            self, "Восстановление сессии", "Найдены несохранённые изменения предыдущей сессии.\nВосстановить?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            SessionManager.clear()
            return
        first_restored = False
        for tab_info in session.get('tabs', []):
            fp, rec = tab_info.get('file_path', ''), tab_info.get('recovery')
            content_text = None
            if rec and os.path.exists(rec):
                try:
                    with open(rec, 'r', encoding='utf-8') as f: content_text = f.read()
                except Exception: pass
            elif fp and os.path.exists(fp):
                try:
                    with open(fp, 'r', encoding='utf-8') as f: content_text = f.read()
                except Exception: pass
            if content_text is None: continue
            
            if not first_restored:
                self.editor_tabs.open_file_tab(fp, content_text) if fp else self.editor_tabs.set_text(content_text)
                first_restored = True
            else:
                self.editor_tabs.new_tab(fp, content_text)
        if first_restored:
            self.chat_history.append("<i style='color:#4EC9B0;'>✓ Сессия восстановлена из авто-бэкапа.</i><br>")

    def closeEvent(self, event):
        self._save_session()
        PersistentSettings.save(self.app_settings)
        SessionManager.clear() 
        event.accept()

    def _try_comfyui(self, ai_response: str):
        if not self.app_settings.get('comfyui_enabled'): return
        try:
            match = re.search(r'\{[\s\S]*?"comfyui"[\s\S]*?\}', ai_response)
            if not match: return
            data = json.loads(match.group())
            comfy = data.get("comfyui", data)
            positive = comfy.get("positive", comfy.get("prompt", ""))
            negative = comfy.get("negative", "")
            if not positive: return

            self.chat_history.append(f"<i style='color:#4EC9B0;'>🎨 Обнаружен ComfyUI запрос: отправка в {self.app_settings.get('comfyui_url')}...</i><br>")
            self._comfy_worker = ComfyUIWorker(
                base_url=self.app_settings.get('comfyui_url', ''), positive=positive, negative=negative,
                steps=self.app_settings.get('comfyui_steps', 20), cfg=self.app_settings.get('comfyui_cfg', 7.0)
            )
            self._comfy_worker.image_ready.connect(self._on_comfy_image)
            self._comfy_worker.status_signal.connect(lambda m: self.chat_history.append(f"<i style='color:#888;'>{m}</i><br>"))
            self._comfy_worker.error_signal.connect(lambda e: self.chat_history.append(f"<b style='color:#CE9178;'>{e}</b><br>"))
            self._comfy_worker.start()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def _on_comfy_image(self, local_path: str):
        url = local_path.replace("\\", "/")
        self.chat_history.append(f"<br><b style='color:#4EC9B0;'>🖼 ComfyUI:</b><br><img src='file:///{url}' style='max-width:400px; border-radius:6px;'><br>")
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def send_message(self):
        if self.worker and self.worker.isRunning(): return
        text = self.chat_input.text().strip()
        if not text: return

        mode      = "coder" if "Кодер" in self.mode_selector.currentText() else "assistant"
        model_key = 'coder_model' if mode == "coder" else 'assistant_model'
        model_file= self.app_settings.get(model_key)

        if not model_file:
            self.chat_history.append("<b style='color:#CE9178;'>Не выбрана модель в настройках.</b><br>")
            return

        sys_p       = self.app_settings.get('coder_system_prompt', '') if mode == "coder" else self.app_settings.get('system_prompt', '')
        code_ctx_raw= self._get_editor_text() if mode == "coder" else self.attached_files_content

        self.token_budget = TokenBudgetManager(n_ctx=self.app_settings.get('n_ctx', 4096), response_reserve=self.app_settings.get('max_tokens', 2048))
        code_context, was_trimmed = self.token_budget.trim_context(code_ctx_raw, text, sys_p)
        if was_trimmed:
            self.chat_history.append("<i style='color:#CDA040;'>⚠ Контекст обрезан — превышен бюджет токенов.</i><br>")

        if self.app_settings.get('use_rag') and self._rag_indexed:
            rag_ctx = self.rag.search(text, top_k=3)
            if rag_ctx:
                code_context = f"[RAG — релевантные фрагменты]\n{rag_ctx}\n\n[Текущий файл]\n{code_context}"
                self.chat_history.append("<i style='color:#888;'>🔍 RAG: найден контекст проекта.</i><br>")

        self.chat_history.append(f"<b style='color:#569CD6;'>Ты:</b> {text}<br>")
        self._chat_raw_log.append(f"USER: {text}")
        self.chat_input.clear()

        self.chat_input.setDisabled(True)
        self.mode_selector.setDisabled(True)
        self.stop_btn.setEnabled(True)

        if mode == "assistant":
            self.attached_files_content = ""

        model_path = os.path.join(os.getcwd(), "models", model_file)
        self.chat_history.append(f"<b style='color:#4EC9B0;'>[{mode} | {model_file}]:</b> ")
        self._current_ai_response = ""

        self.worker = LlamaCppWorker(
            prompt=text, code_context=code_context, model_path=model_path,
            system_prompt=self.app_settings.get('system_prompt', ''),
            coder_system_prompt=self.app_settings.get('coder_system_prompt', ''),
            temperature=self.app_settings.get('temperature', 0.7),
            max_tokens=self.app_settings.get('max_tokens', 2048), mode=mode
        )
        self.worker.chunk_received.connect(self.update_chat)
        self.worker.status_signal.connect(self.add_status_msg)
        self.worker.finished_signal.connect(self.unlock_ui)
        self.worker.start()

    def unlock_ui(self):
        self.chat_input.setDisabled(False)
        self.mode_selector.setDisabled(False)
        self.stop_btn.setEnabled(False)
        self.chat_input.setFocus()
        self.chat_history.append("<br>")
        self._chat_raw_log.append(f"AI: {self._current_ai_response}")
        self._update_token_bar()
        self._try_comfyui(self._current_ai_response)

    def add_status_msg(self, msg):
        self.chat_history.append(msg)

    def update_chat(self, chunk):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        self.chat_history.setTextCursor(cursor)
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())
        self._current_ai_response += chunk

    def clear_chat(self):
        reply = QMessageBox.question(
            self, "Очистить чат", "Очистить всю историю диалога?\n(raw_log для поиска кода тоже будет сброшен)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.chat_history.clear()
            self._chat_raw_log.clear()
            self._current_ai_response = ""

    def export_chat_md(self):
        if not self._chat_raw_log:
            self.chat_history.append("<i style='color:#888;'>Нет истории для экспорта.</i><br>")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Экспорт чата", os.getcwd(), "Markdown (*.md);;All Files (*)")
        if not fp: return
        lines = ["# Zen AI Editor — экспорт диалога\n\n"]
        for entry in self._chat_raw_log:
            if entry.startswith("USER: "):
                lines.append(f"## 👤 Пользователь\n\n{entry[6:]}\n\n")
            elif entry.startswith("AI: "):
                lines.append(f"## 🤖 ИИ\n\n{entry[4:]}\n\n---\n\n")
        try:
            with open(fp, 'w', encoding='utf-8') as f: f.writelines(lines)
            self.chat_history.append(f"<i style='color:#4EC9B0;'>✓ Экспорт: {os.path.basename(fp)}</i><br>")
        except Exception as e:
            self.chat_history.append(f"<b style='color:#CE9178;'>Ошибка: {e}</b><br>")

    def show_cheat_sheet(self):
        CheatSheetDialog(self).exec()

    def apply_code_from_chat(self):
        full_log = "\n".join(self._chat_raw_log)
        blocks   = re.findall(r'```[a-zA-Z]*\n(.*?)```', full_log, re.DOTALL)
        if not blocks:
            self.chat_history.append("<br><i style='color:#888888;'>[Блоки кода не найдены]</i><br>")
            self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())
            return

        new_code = blocks[-1].strip()

        if self.editor_tabs.has_selection():
            old_sel = self.editor_tabs.selected_text()
            if self.app_settings.get('diff_before_apply', True):
                dlg = DiffApplyDialog(old_sel, new_code, parent=self)
                if dlg.exec():
                    self._replace_selection(dlg.accepted_code)
                    self.chat_history.append("<br><i style='color:#888888;'>[Выделенный код заменён с diff]</i><br>")
                else:
                    self.chat_history.append("<br><i style='color:#888888;'>[Применение отменено]</i><br>")
            else:
                self._replace_selection(new_code)
                self.chat_history.append("<br><i style='color:#888888;'>[Выделенный код заменён]</i><br>")
        else:
            old_code = self._get_editor_text()
            if self.app_settings.get('diff_before_apply', True) and old_code.strip():
                dlg = DiffApplyDialog(old_code, new_code, parent=self)
                if dlg.exec():
                    self._set_editor_text(dlg.accepted_code, undo_safe=True)
                    self.chat_history.append("<br><i style='color:#888888;'>[Код применён с diff]</i><br>")
                else:
                    self.chat_history.append("<br><i style='color:#888888;'>[Применение отменено]</i><br>")
            else:
                self._set_editor_text(new_code, undo_safe=True)
                self.chat_history.append("<br><i style='color:#888888;'>[Код перенесён в редактор]</i><br>")

        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())