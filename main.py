"""
Zen AI Editor — точка входа.

Запуск:
    python main.py

Перед первым запуском:
1. Установите зависимости: pip install -r requirements.txt
2. Скачайте модели в формате GGUF и положите в папку models/:
   - Qwen2.5-Coder-32B-Instruct-Q3_K_M.gguf (для профиля Кодер)
   - Hermes-3-Llama-3.1-8B.Q4_K_M.gguf      (для профиля Алиса)
3. Запустите, откройте Настройки (⚙), выберите файл модели для каждого профиля.
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QStyleFactory

from ui.main_window import ZenEditor


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = ZenEditor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
