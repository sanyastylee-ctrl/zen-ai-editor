from PyQt6.QtWidgets import (QDialog, QFormLayout, QComboBox, QTextEdit, 
                             QDoubleSpinBox, QSpinBox, QCheckBox, QLabel, 
                             QLineEdit, QPushButton)

class SettingsDialog(QDialog):
    def __init__(self, parent=None, current_settings=None, available_models=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки ИИ")
        self.setMinimumWidth(500)
        self.setStyleSheet("""
            QDialog { background-color: #252526; color: #D4D4D4; }
            QLabel { color: #D4D4D4; font-size: 13px; }
            QComboBox, QTextEdit, QDoubleSpinBox, QSpinBox, QLineEdit {
                background-color: #3C3C3C; color: #FFFFFF;
                border: 1px solid #555555; border-radius: 4px; padding: 5px;
            }
            QPushButton { background-color: #0E639C; color: white;
                border-radius: 4px; padding: 6px 15px; font-weight: bold; }
            QPushButton:hover { background-color: #1177BB; }
            QCheckBox { color: #D4D4D4; }
            QGroupBox { color: #888; border: 1px solid #444; border-radius:4px;
                margin-top: 8px; padding: 8px; font-size: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #666; }
        """)
        self.settings = current_settings or {}
        layout = QFormLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        self.coder_combo = QComboBox()
        self.assistant_combo = QComboBox()
        if available_models:
            self.coder_combo.addItems(available_models)
            self.assistant_combo.addItems(available_models)
            if self.settings.get('coder_model') in available_models:
                self.coder_combo.setCurrentText(self.settings['coder_model'])
            if self.settings.get('assistant_model') in available_models:
                self.assistant_combo.setCurrentText(self.settings['assistant_model'])

        self.sys_prompt_edit = QTextEdit()
        self.sys_prompt_edit.setMaximumHeight(65)
        self.sys_prompt_edit.setPlainText(
            self.settings.get('system_prompt', 'Ты — полезный ИИ-ассистент. Отвечай по существу.')
        )
        self.coder_prompt_edit = QTextEdit()
        self.coder_prompt_edit.setMaximumHeight(65)
        self.coder_prompt_edit.setPlainText(
            self.settings.get('coder_system_prompt', 'Ты — опытный программист. Пиши чистый рабочий код. Отвечай кратко.')
        )

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0); self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(self.settings.get('temperature', 0.7))

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(256, 8192); self.max_tokens_spin.setSingleStep(256)
        self.max_tokens_spin.setValue(self.settings.get('max_tokens', 2048))

        self.n_ctx_spin = QSpinBox()
        self.n_ctx_spin.setRange(512, 32768); self.n_ctx_spin.setSingleStep(512)
        self.n_ctx_spin.setValue(self.settings.get('n_ctx', 4096))

        self.diff_check = QCheckBox("Показывать diff перед вставкой кода")
        self.diff_check.setChecked(self.settings.get('diff_before_apply', True))

        self.rag_check = QCheckBox("Использовать RAG при запросах (если проиндексирован)")
        self.rag_check.setChecked(self.settings.get('use_rag', False))

        layout.addRow("Модель Кодера:", self.coder_combo)
        layout.addRow("Модель Ассистента:", self.assistant_combo)
        layout.addRow("Промпт Ассистента:", self.sys_prompt_edit)
        layout.addRow("Промпт Кодера:", self.coder_prompt_edit)
        layout.addRow("Температура:", self.temp_spin)
        layout.addRow("Макс. токенов:", self.max_tokens_spin)
        layout.addRow("n_ctx модели:", self.n_ctx_spin)
        layout.addRow("", self.diff_check)
        layout.addRow("", self.rag_check)

        comfy_label = QLabel("── ComfyUI / Stable Diffusion ──")
        comfy_label.setStyleSheet("color:#4EC9B0; font-size:12px; font-weight:bold; margin-top:6px;")
        layout.addRow(comfy_label)

        self.comfyui_check = QCheckBox("Включить интеграцию ComfyUI")
        self.comfyui_check.setChecked(self.settings.get('comfyui_enabled', False))
        layout.addRow("", self.comfyui_check)

        self.comfyui_url_edit = QLineEdit()
        self.comfyui_url_edit.setPlaceholderText("http://127.0.0.1:8188")
        self.comfyui_url_edit.setText(self.settings.get('comfyui_url', 'http://127.0.0.1:8188'))
        layout.addRow("ComfyUI URL:", self.comfyui_url_edit)

        self.comfy_steps_spin = QSpinBox()
        self.comfy_steps_spin.setRange(1, 100)
        self.comfy_steps_spin.setValue(self.settings.get('comfyui_steps', 20))
        layout.addRow("Шагов (sampling):", self.comfy_steps_spin)

        self.comfy_cfg_spin = QDoubleSpinBox()
        self.comfy_cfg_spin.setRange(1.0, 30.0); self.comfy_cfg_spin.setSingleStep(0.5)
        self.comfy_cfg_spin.setValue(self.settings.get('comfyui_cfg', 7.0))
        layout.addRow("CFG Scale:", self.comfy_cfg_spin)

        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self.accept)
        layout.addRow("", save_btn)

    def get_settings(self):
        return {
            'coder_model':        self.coder_combo.currentText(),
            'assistant_model':    self.assistant_combo.currentText(),
            'system_prompt':      self.sys_prompt_edit.toPlainText(),
            'coder_system_prompt':self.coder_prompt_edit.toPlainText(),
            'temperature':        self.temp_spin.value(),
            'max_tokens':         self.max_tokens_spin.value(),
            'n_ctx':              self.n_ctx_spin.value(),
            'diff_before_apply':  self.diff_check.isChecked(),
            'use_rag':            self.rag_check.isChecked(),
            'comfyui_enabled':    self.comfyui_check.isChecked(),
            'comfyui_url':        self.comfyui_url_edit.text().strip(),
            'comfyui_steps':      self.comfy_steps_spin.value(),
            'comfyui_cfg':        self.comfy_cfg_spin.value(),
        }