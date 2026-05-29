# Zen AI Editor

Локальный AI-редактор с двумя профилями: **Кодер** (Qwen2.5-Coder) для работы с кодом и **Компаньон** (Hermes 3) для свободного диалога.

## Архитектура

```
zen-ai-editor/
├── main.py                 — точка входа
├── core/
│   ├── profiles.py         — AIProfile, ProfileManager (JSON в %APPDATA%/ZenAI/settings/)
│   ├── model_manager.py    — менеджер моделей с LRU-кэшем
│   ├── token_budget.py     — обрезка контекста под n_ctx
│   ├── chat_templates.py   — ChatML/Llama3/Mistral/Gemma/DeepSeek
│   └── paths.py            — resolve_model_path, list_available_models
├── ai/
│   ├── prompt_builder.py   — единая сборка финального промпта
│   └── worker.py           — QThread инференс с стримингом
├── rag/
│   └── project_rag.py      — faiss-индексация проекта
├── sandbox/
│   └── terminal.py         — встроенный терминал
├── widgets/
│   └── code_editor.py      — QScintilla + Diff dialog
├── ui/
│   ├── main_window.py      — главное окно
│   ├── profile_switcher.py — переключатель Кодер ↔ Алиса
│   ├── profile_editor.py   — редактор одного профиля (вкладки)
│   ├── persona_editor.py   — редактор персоны компаньона
│   └── settings_dialog.py  — главные настройки
└── models/                 — сюда .gguf файлы (создаётся автоматически)
```

## Установка

```bash
# 1. Зависимости
pip install -r requirements.txt

# 2. Под GPU (RTX 50xx — CUDA 12.4) пересоберите llama-cpp:
pip install llama-cpp-python --extra-index-url \
    https://abetlen.github.io/llama-cpp-python/whl/cu124 --upgrade --force-reinstall
```

## Модели

Положите GGUF-файлы в папку `models/`:

| Профиль | Модель | Квантизация | VRAM |
|---|---|---|---|
| Кодер | `Qwen2.5-Coder-32B-Instruct-Q3_K_M.gguf` | Q3_K_M | ~15 GB |
| Алиса | `Hermes-3-Llama-3.1-8B.Q4_K_M.gguf` | Q4_K_M | ~5 GB |

Запустите → откройте Настройки (⚙) → выберите файл для каждого профиля → Сохранить.

## Работа с моделями

Одна модель загружена за раз. При клике на другой профиль текущая выгружается, новая загружается. Загрузка с NVMe занимает ~5-8 секунд для 32B, ~2-3 секунды для 8B.

История чата хранится **раздельно** для каждого профиля — переключение не сбрасывает диалог.

## Настройка персонажа

Профиль Компаньона → вкладка **Персона**. Поля (имя, возраст, характер, манера речи, биография, настроение, отношения) подставляются в системный промпт через переменные `{character_name}`, `{personality}` и т.д.

Дефолтный промпт уже содержит инструкции «не AI, не ассистент, без disclaimer'ов, говори по-русски, не выходи из роли». Менять под себя — вкладка **Промпт**.

## Кодер с RAG

Сайдбар → **⟳ Индекс RAG** → проект индексируется через faiss + MiniLM (кэш в `%APPDATA%\ZenAI\memory\`).
Включить RAG для кодера: Настройки → **Использовать RAG для кодера**.

## Хоткеи

| Действие | Сочетание |
|---|---|
| Сохранить файл | Ctrl+S |
| Отправить в чат | Enter |
| Остановить генерацию | кнопка ⏹ |

## Данные приложения

Постоянные данные хранятся вне открытого проекта, в `%APPDATA%\ZenAI\`:

- `chats/` — JSON-сессии чатов;
- `sessions/` — состояние последнего профиля и недавних проектов;
- `memory/` — кэш RAG по проектам;
- `settings/` — настройки и профили.

При первом запуске после обновления существующие файлы из `~/.zen_ai/` и
проектный RAG-кэш читаются через одноразовое копирование в новое хранилище.

## Сборка Windows

Для portable-сборки в режиме `onedir`:

```bash
pip install pyinstaller
pyinstaller ZenAI.spec --clean
```

Результат появится в `dist/ZenAI/`. Папка `models/` с `.gguf` должна находиться
рядом с `ZenAI.exe`; модели не включаются в сборку из-за их размера.
