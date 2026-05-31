"""Lumos OS desktop launcher — the PyInstaller entry point.

Keeps config + memory in a per-user data dir so the packaged app runs from
anywhere (including a read-only install location like Program Files), then starts
the server which serves the HUD and opens the browser. Double-click → setup
wizard (first run) → chat.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _user_data_dir() -> Path:
    """OS-appropriate per-user data dir for config (.env) + memory (cache)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    d = Path(base) / "LumosOS"
    (d / "cache").mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    data = _user_data_dir()
    # setdefault → respects an explicit override (dev), else uses the user-data dir.
    os.environ.setdefault("LUMOS_CONFIG_DIR", str(data))
    os.environ.setdefault("LUMOS_CACHE_DIR", str(data / "cache"))
    from lumos_node.cli import app as cli_app

    sys.argv = ["lumos", "serve"]
    cli_app()


if __name__ == "__main__":
    main()
