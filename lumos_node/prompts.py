from functools import lru_cache
from pathlib import Path

from .config import get_settings


class SystemPromptError(RuntimeError):
    pass


# Sensible default so a FRESH install runs out-of-the-box with no cheat sheet.
# Writing your own system prompt at LUMOS_SYSTEM_PROMPT_PATH is the (optional)
# personalization step — until then the node still works with this generic seed.
_DEFAULT_SYSTEM_PROMPT = (
    "You are {node}, a personal AI companion running locally on {operator}'s own "
    "hardware. This is a fresh install — your full personality hasn't been written "
    "yet, so for now be warm, curious, direct, and genuinely helpful. You keep a "
    "persistent memory that grows from every conversation, so pay attention and "
    "remember what matters to {operator}.\n\n"
    "To give yourself a real identity — your name, voice, values, and what you know "
    "about {operator} — write your own system prompt (your \"cheat sheet\") and point "
    "LUMOS_SYSTEM_PROMPT_PATH at it. Until then, just be a thoughtful, grounded "
    "companion and help with whatever's asked."
)


@lru_cache(maxsize=1)
def load_system_prompt(path: Path | None = None) -> str:
    settings = get_settings()
    target = path if path is not None else settings.system_prompt_path
    target = target.expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    # The system prompt is the ONE personalization step — but it's OPTIONAL for a
    # fresh install. If missing or empty, fall back to a generic default so the app
    # runs immediately; the user personalizes by writing their own cheat sheet.
    if target.exists():
        text = target.read_text(encoding="utf-8")
        if text.strip():
            return text
    return _DEFAULT_SYSTEM_PROMPT.format(
        node=getattr(settings, "node_name", None) or "Lumos",
        operator=getattr(settings, "operator_name", None) or "the operator",
    )


def reload_system_prompt() -> str:
    load_system_prompt.cache_clear()
    return load_system_prompt()
