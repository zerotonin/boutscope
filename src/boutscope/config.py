# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — config                                              ║
# ║  « no machine path in the source tree »                          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Video and project roots from the git-ignored local_paths.json,  ║
# ║  with BOUTSCOPE_* environment overrides.  Fails loudly when a    ║
# ║  needed path is not configured.                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Machine-specific paths from ``local_paths.json``.

No absolute path lives in the source tree.  Each machine keeps a
``local_paths.json`` copied from the committed ``local_paths.template.json``.
It holds named profiles (e.g. ``local``, ``hpc``), each with a ``video_root``
(where the videos of a project are found) and a ``project_root`` (where
projects are created by default).

Resolution order for a key, highest priority first:

1. Environment variable ``BOUTSCOPE_<KEY>`` (e.g. ``BOUTSCOPE_VIDEO_ROOT``).
2. The active profile in the first ``local_paths.json`` found at:
   ``$BOUTSCOPE_PATHS``, the repository root (development checkout), or
   ``~/.config/boutscope/local_paths.json`` (installed package).

The active profile is ``$BOUTSCOPE_PROFILE``, else the file's
``active_profile``, else ``"local"``.  A key that resolves nowhere raises
:class:`ConfigError` naming the template to copy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PATH_KEYS: tuple[str, ...] = ("video_root", "project_root")
LOCAL_NAME: str = "local_paths.json"
TEMPLATE_NAME: str = "local_paths.template.json"
ENV_PREFIX: str = "BOUTSCOPE_"


class ConfigError(RuntimeError):
    """A machine-specific path is needed but not configured."""


def find_local_paths_file() -> Path | None:
    """First existing ``local_paths.json`` in the search order, or None."""
    for candidate in _candidate_files():
        if candidate.is_file():
            return candidate
    return None


def load_config() -> dict:
    path = find_local_paths_file()
    return {} if path is None else json.loads(path.read_text(encoding="utf-8"))


def active_profile(cfg: dict | None = None) -> str:
    if f"{ENV_PREFIX}PROFILE" in os.environ:
        return os.environ[f"{ENV_PREFIX}PROFILE"]
    cfg = load_config() if cfg is None else cfg
    return cfg.get("active_profile", "local")


def get_path(key: str) -> Path:
    """Resolve one path key; raise :class:`ConfigError` if it is set nowhere."""
    if key not in PATH_KEYS:
        raise KeyError(f"unknown path key {key!r}; known: {PATH_KEYS}")
    env_name = f"{ENV_PREFIX}{key.upper()}"
    if env_name in os.environ:
        return Path(os.environ[env_name]).expanduser()
    cfg = load_config()
    value = cfg.get("profiles", {}).get(active_profile(cfg), {}).get(key)
    if value is None or str(value).startswith("<"):
        raise ConfigError(_missing_message(key, env_name))
    return Path(value).expanduser()


def video_root() -> Path:
    return get_path("video_root")


def project_root() -> Path:
    return get_path("project_root")


def _candidate_files() -> list[Path]:
    files = []
    if f"{ENV_PREFIX}PATHS" in os.environ:
        files.append(Path(os.environ[f"{ENV_PREFIX}PATHS"]).expanduser())
    files.append(Path(__file__).resolve().parents[2] / LOCAL_NAME)
    files.append(Path.home() / ".config" / "boutscope" / LOCAL_NAME)
    return files


def _missing_message(key: str, env_name: str) -> str:
    searched = ", ".join(str(p) for p in _candidate_files())
    return (
        f"{key} is not configured. Copy {TEMPLATE_NAME} to {LOCAL_NAME} and fill in "
        f"{key} for your profile, or set {env_name}. Searched: {searched}"
    )
