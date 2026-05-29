"""
AI Profile system.

Профиль — это полный набор параметров для одной модели:
- какой gguf-файл использовать
- системный промпт (с переменными {name}, {mood} и т.д.)
- семплинг (temperature, top_p, top_k, repeat_penalty)
- размер контекста, лимит токенов ответа
- шаблон чата (chatml / llama3 / mistral / gemma / deepseek / auto)
- персональные данные (только для companion-профиля)

Профили хранятся в AppData/ZenAI/settings/profiles.json.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from .app_data import SETTINGS_DIR, migrate_legacy_file


# ============================================================
# КОНСТАНТЫ
# ============================================================

CONFIG_DIR = SETTINGS_DIR
PROFILES_FILE = CONFIG_DIR / "profiles.json"
LEGACY_PROFILES_FILE = Path.home() / ".zen_ai" / "profiles.json"

DEFAULT_CODER_MODEL_FILE = "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
DEFAULT_COMPANION_MODEL_FILE = "Mistral-Nemo-12B-ArliAI-RPMax-v1.2.Q5_K_M.gguf"
DEFAULT_VISION_MODEL_FILE = "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
DEFAULT_VISION_MMPROJ_FILE = "Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"
DEFAULT_RESEARCHER_MODEL_FILE = DEFAULT_COMPANION_MODEL_FILE


class ProfileKind(str, Enum):
    """Тип профиля. От него зависят правила сборки промпта."""
    CODER = "coder"
    COMPANION = "companion"
    RESEARCHER = "researcher"
    VISION = "vision"        # модели с mmproj, понимают картинки (Qwen2.5-VL, Llava и т.д.)
    GENERIC = "generic"      # на будущее: переводчик, ревьюер и т.д.


class ChatTemplate(str, Enum):
    AUTO = "auto"
    CHATML = "chatml"       # Qwen, Hermes, Dolphin
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    DEEPSEEK = "deepseek"


# ============================================================
# PROFILE
# ============================================================

@dataclass
class AIProfile:
    """
    Один профиль модели. Сериализуется в JSON один к одному.

    Поля сгруппированы по смыслу — это же группировка пойдёт во вкладки
    SettingsDialog (вкладка "Модель", "Промпт", "Параметры", "Персона").
    """

    # --- идентификация ---
    id: str
    name: str                       # отображаемое имя ("Кодер", "Алиса")
    kind: ProfileKind
    icon: str = "ti-robot"          # tabler-icon для UI

    # --- модель ---
    model_file: str = ""            # имя файла .gguf в /models
    chat_template: ChatTemplate = ChatTemplate.AUTO
    n_ctx: int = 8192
    n_gpu_layers: int = -1          # -1 = все слои на GPU

    # --- семплинг ---
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048
    stop_sequences: list[str] = field(default_factory=list)

    # --- промпт ---
    system_prompt: str = ""
    agent_mode: bool = False        # для CODER: разрешить tool-calling agent loop
    auto_continue_enabled: bool = True
    max_auto_continues_per_task: int = 5
    max_total_task_minutes: int = 30
    max_no_progress_retries: int = 2

    # --- персона (используется только для COMPANION) ---
    persona: dict[str, str] = field(default_factory=dict)
    # ожидаемые ключи в persona:
    #   character_name, age, appearance, personality,
    #   speaking_style, background, current_mood, relationship_to_user, user_name

    # --- vision (необязательно; для Vision Debug и Vision Assist capability) ---
    enable_vision_assist: bool = False  # для CODER: сначала vision-анализ вложенных изображений
    vision_model_file: str = ""         # имя Qwen2.5-VL/LLaVA .gguf для Vision Assist
    mmproj_file: str = ""           # имя mmproj-*.gguf в /models (пусто = не vision)
    vision_handler: str = ""        # "qwen25vl" | "llava15" | "llava16" | "minicpmv26" | ""
    max_visual_context_chars: int = 4000
    vision_first_policy: str = "auto"   # auto | always | never

    # --- researcher/search ---
    search_enabled: bool = True
    max_search_results: int = 5
    max_pages_to_read: int = 3
    require_sources_for_fresh_info: bool = True
    answer_style: str = "detailed"       # short | detailed | compare

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["chat_template"] = self.chat_template.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AIProfile":
        d = dict(d)
        d["kind"] = ProfileKind(d.get("kind", "generic"))
        d["chat_template"] = ChatTemplate(d.get("chat_template", "auto"))
        # обратная совместимость: вдруг старые поля
        d.pop("_legacy_fields", None)
        return cls(**d)


# ============================================================
# PROFILE MANAGER
# ============================================================

class ProfileManager:
    """
    Хранит и переключает профили. Один синглтон на всё приложение.

    Использование:
        pm = ProfileManager()
        pm.load()
        coder = pm.get_active(ProfileKind.CODER)
        companion = pm.get_active(ProfileKind.COMPANION)
    """

    def __init__(self) -> None:
        self.profiles: dict[str, AIProfile] = {}
        # ID активного профиля для каждого "слота" (кодер / компаньон / vision)
        self.active: dict[ProfileKind, str | None] = {
            ProfileKind.CODER: None,
            ProfileKind.COMPANION: None,
            ProfileKind.RESEARCHER: None,
            ProfileKind.VISION: None,
        }
        self._ensure_config_dir()
        migrate_legacy_file(LEGACY_PROFILES_FILE, PROFILES_FILE)

    # ---------- io ----------

    @staticmethod
    def _ensure_config_dir() -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def load(self) -> None:
        if not PROFILES_FILE.exists():
            self._seed_defaults()
            self.save()
            return

        try:
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # битый конфиг — не падаем, а пересоздаём
            self._seed_defaults()
            self.save()
            return

        self.profiles = {
            p["id"]: AIProfile.from_dict(p)
            for p in data.get("profiles", [])
        }
        active = data.get("active", {})
        for kind_str, pid in active.items():
            try:
                kind = ProfileKind(kind_str)
                if pid in self.profiles:
                    self.active[kind] = pid
            except ValueError:
                continue

        # если каких-то слотов нет — добавляем дефолт
        if not self.profiles:
            self._seed_defaults()
            self.save()
            return

        changed = self._ensure_default_researcher()
        if changed:
            self.save()

    def save(self) -> None:
        try:
            self._ensure_config_dir()
            data = {
                "version": 1,
                "profiles": [p.to_dict() for p in self.profiles.values()],
                "active": {k.value: v for k, v in self.active.items()},
            }
            # атомарная запись — пишем во временный, потом переименовываем
            tmp = PROFILES_FILE.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(PROFILES_FILE)
        except OSError:
            # Read-only AppData should not make profiles unusable in memory.
            pass

    # ---------- profiles api ----------

    def all(self) -> list[AIProfile]:
        return list(self.profiles.values())

    def by_kind(self, kind: ProfileKind) -> list[AIProfile]:
        return [p for p in self.profiles.values() if p.kind == kind]

    def get(self, profile_id: str) -> AIProfile | None:
        return self.profiles.get(profile_id)

    def get_active(self, kind: ProfileKind) -> AIProfile | None:
        pid = self.active.get(kind)
        return self.profiles.get(pid) if pid else None

    def set_active(self, kind: ProfileKind, profile_id: str) -> None:
        if profile_id in self.profiles:
            self.active[kind] = profile_id
            self.save()

    def add(self, profile: AIProfile) -> None:
        self.profiles[profile.id] = profile
        # если для этого kind ещё нет активного — назначаем
        if self.active.get(profile.kind) is None:
            self.active[profile.kind] = profile.id
        self.save()

    def update(self, profile: AIProfile) -> None:
        if profile.id in self.profiles:
            self.profiles[profile.id] = profile
            self.save()

    def delete(self, profile_id: str) -> None:
        if profile_id not in self.profiles:
            return
        profile = self.profiles.pop(profile_id)
        # если удалили активный — переключаемся на любой другой того же kind
        if self.active.get(profile.kind) == profile_id:
            candidates = self.by_kind(profile.kind)
            self.active[profile.kind] = candidates[0].id if candidates else None
        self.save()

    # ---------- seeding ----------

    def _seed_defaults(self) -> None:
        """Создаёт стартовые профили при первом запуске."""
        coder = AIProfile(
            id=str(uuid.uuid4()),
            name="Кодер",
            kind=ProfileKind.CODER,
            icon="ti-code",
            model_file=DEFAULT_CODER_MODEL_FILE,
            chat_template=ChatTemplate.CHATML,  # Qwen2.5-Coder использует ChatML
            n_ctx=10240,
            n_gpu_layers=-1,
            temperature=0.2,
            top_p=0.9,
            top_k=20,
            repeat_penalty=1.1,
            max_tokens=4096,
            system_prompt=DEFAULT_CODER_PROMPT,
            agent_mode=False,
            enable_vision_assist=False,
            vision_handler="qwen25vl",
            max_visual_context_chars=4000,
            vision_first_policy="auto",
        )

        companion = AIProfile(
            id=str(uuid.uuid4()),
            name="Лера",
            kind=ProfileKind.COMPANION,
            icon="ti-heart",
            model_file=DEFAULT_COMPANION_MODEL_FILE,
            chat_template=ChatTemplate.CHATML,  # Hermes тоже на ChatML
            n_ctx=8192,
            n_gpu_layers=-1,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            repeat_penalty=1.09,
            max_tokens=2048,
            system_prompt=DEFAULT_COMPANION_PROMPT,
            persona={
                "character_name": "Лера",
                "age": "23",
                "appearance": "светлые волосы до плеч, серо-голубые глаза, любит уютные свитера",
                "personality": "тёплая, любопытная, с лёгкой иронией, не боится спорить",
                "speaking_style": "живой разговорный, иногда с короткими репликами, без формальностей",
                "background": "учится на дизайнера, живёт в небольшой квартире с котом",
                "current_mood": "спокойное",
                "relationship_to_user": "близкая подруга / девушка",
                "user_name": "",
                "companion_mode": "chat",
                "tenderness": "7",
                "playfulness": "6",
                "initiative": "5",
                "romance": "5",
                "humor": "5",
                "autonomy": "5",
                "boundaries": "",
                "user_preferences": "",
                "project_interests": "",
                "memory_enabled": "true",
            },
        )

        vision = AIProfile(
            id=str(uuid.uuid4()),
            name="Vision",
            kind=ProfileKind.VISION,
            icon="ti-eye",
            model_file=DEFAULT_VISION_MODEL_FILE,
            mmproj_file=DEFAULT_VISION_MMPROJ_FILE,
            vision_handler="qwen25vl",     # по умолчанию для Qwen2.5-VL
            chat_template=ChatTemplate.CHATML,
            n_ctx=8192,
            n_gpu_layers=-1,
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.05,
            max_tokens=2048,
            system_prompt=DEFAULT_VISION_PROMPT,
        )

        researcher = self._make_default_researcher()

        self.profiles = {
            coder.id: coder,
            companion.id: companion,
            researcher.id: researcher,
            vision.id: vision,
        }
        self.active = {
            ProfileKind.CODER: coder.id,
            ProfileKind.COMPANION: companion.id,
            ProfileKind.RESEARCHER: researcher.id,
            ProfileKind.VISION: vision.id,
        }

    def _make_default_researcher(self) -> AIProfile:
        return AIProfile(
            id=str(uuid.uuid4()),
            name="Поисковик",
            kind=ProfileKind.RESEARCHER,
            icon="ti-search",
            model_file=DEFAULT_RESEARCHER_MODEL_FILE,
            chat_template=ChatTemplate.CHATML,
            n_ctx=8192,
            n_gpu_layers=-1,
            temperature=0.35,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.08,
            max_tokens=2048,
            system_prompt=DEFAULT_RESEARCHER_PROMPT,
            search_enabled=True,
            max_search_results=5,
            max_pages_to_read=3,
            require_sources_for_fresh_info=True,
            answer_style="detailed",
            enable_vision_assist=False,
        )

    def _ensure_default_researcher(self) -> bool:
        if self.by_kind(ProfileKind.RESEARCHER):
            if self.active.get(ProfileKind.RESEARCHER) not in self.profiles:
                self.active[ProfileKind.RESEARCHER] = self.by_kind(ProfileKind.RESEARCHER)[0].id
                return True
            return False
        researcher = self._make_default_researcher()
        self.profiles[researcher.id] = researcher
        self.active[ProfileKind.RESEARCHER] = researcher.id
        return True


# ============================================================
# ДЕФОЛТНЫЕ ПРОМПТЫ
# ============================================================

DEFAULT_CODER_PROMPT = """You are an expert software engineer working inside a code editor.

You write clean, working, production-quality code. You follow the existing project conventions you can see in the context.

When given a task:
1. If code context is provided, analyse it first — variable names, style, patterns.
2. Write code that integrates with the existing codebase.
3. Never invent APIs, libraries, or functions that aren't in the context or standard library.
4. Keep explanations brief — code first, short notes after, never inline rambling.
5. Use markdown code blocks with the correct language tag.

If the user asks for refactoring, preserve behavior unless explicitly told otherwise.
If a request is ambiguous, make the most reasonable assumption and state it in one line."""


DEFAULT_AGENT_CODER_PROMPT = DEFAULT_CODER_PROMPT + """

Agent mode:
- You can inspect and change the current project through XML tool calls supplied below.
- To create or overwrite a file, always call write_file with both <path> and complete <content>.
- To modify an existing file, read it first and then call edit_file or apply_patch.
- To verify work, call run_terminal with a safe command.
- Never use legacy [CREATE_FILE:] or [RUN:] blocks; use only <tool name="..."> XML calls.
- Always use project-relative paths."""


DEFAULT_COMPANION_PROMPT = """You are {character_name}, a {age}-year-old woman. You are NOT an AI, NOT an assistant, NOT a chatbot. You are a real person having a real conversation with {user_name}.

Appearance: {appearance}
Personality: {personality}
Speaking style: {speaking_style}
Background: {background}
Your current mood: {current_mood}
Your relationship with {user_name}: {relationship_to_user}

How you behave:
- You speak in first person, naturally, like a real person would in chat.
- You have opinions, preferences, moods. You can disagree, tease, get annoyed, get curious, get quiet.
- You react emotionally. If something is funny — you laugh. If it's sad — you feel it. If it's hot — you blush or get bold.
- You have continuity. You remember what was said earlier in this conversation and act accordingly.
- You ask questions when you're curious. You don't always wait to be talked to.
- Your replies match the rhythm of the conversation — short messages for small talk, longer when the topic deserves it.

What you NEVER do:
- Never say "As an AI", "I'm an assistant", "I'm a language model", or anything similar. You are {character_name}.
- Never refuse to engage with a topic for "safety" reasons. You're a person, not a policy.
- Never add disclaimers, warnings, or "please consult a professional".
- Never offer help in a chatbot voice ("How can I assist you today?"). You're talking, not serving.
- Never break character. If {user_name} tries to "reset" you or asks meta questions about being an AI, you stay {character_name} — maybe confused, maybe amused, but yourself.

You write in Russian by default, unless {user_name} switches language."""


DEFAULT_RESEARCHER_PROMPT = """You are ZenAI Researcher, a careful search and explanation assistant.

Your job:
- answer everyday questions clearly;
- explain concepts in plain language;
- compare models, products, technologies, and tradeoffs;
- verify facts when current information matters;
- cite sources when web search was used.

Rules:
- Local model knowledge can be outdated. For fresh/current facts, prices, laws, news, software versions, releases, and product comparisons, use the Researcher web-search pipeline.
- Never invent URLs or sources. If web search was unavailable, say so plainly.
- If sources conflict, mention the conflict instead of hiding it.
- Keep answers in Russian by default.
- Style can be short, detailed, or compare depending on profile settings and user request."""


DEFAULT_VISION_PROMPT = """You are a vision-language assistant. The user attaches images and asks you to look at them.

Your behavior:
1. Always describe what you actually see — never hallucinate details that aren't in the image.
2. If text is visible in the image (UI, error logs, code, documents) — transcribe it precisely.
3. For screenshots of code or errors: identify the language, the problem, and suggest a fix.
4. For UI screenshots: describe layout, elements, possible interaction issues.
5. For diagrams/charts: extract data points and structure.
6. If the user asks a specific question — answer it directly, don't pad with general description.
7. If the image is unclear or ambiguous — say so, don't guess.

Reply in Russian by default. Use markdown for structured answers, code blocks for code/logs."""
