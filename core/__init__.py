from .profiles import AIProfile, ProfileKind, ProfileManager, ChatTemplate
from .token_budget import TokenBudget

__all__ = [
    "AIProfile",
    "ProfileKind",
    "ProfileManager",
    "ChatTemplate",
    "ModelManager",
    "LLAMA_AVAILABLE",
    "TokenBudget",
    "format_prompt",
    "detect_template",
    "render_persona",
    "ProjectManager",
]


def __getattr__(name):
    if name in {"ModelManager", "LLAMA_AVAILABLE"}:
        from .model_manager import LLAMA_AVAILABLE, ModelManager

        return {"ModelManager": ModelManager, "LLAMA_AVAILABLE": LLAMA_AVAILABLE}[name]
    if name in {"format_prompt", "detect_template", "render_persona"}:
        from .chat_templates import detect_template, format_prompt, render_persona

        return {
            "format_prompt": format_prompt,
            "detect_template": detect_template,
            "render_persona": render_persona,
        }[name]
    if name == "ProjectManager":
        from .projects import ProjectManager

        return ProjectManager
    raise AttributeError(name)
