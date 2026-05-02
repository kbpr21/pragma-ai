"""User-level configuration persisted to ``~/.pragma/config.json``.

This module is the single source of truth for the LLM provider, model,
API key, and base URL chosen by the user via ``pragma connect``.

It deliberately lives separately from :class:`pragma.config.PragmaConfig`,
which models *runtime / library* configuration (knowledge-base directory,
hop depth, fact thresholds, etc.). User-config is **interactive,
machine-local, and security-sensitive**, so it has its own concerns:

* Stored in the platform-appropriate user directory:
  ``~/.pragma/config.json`` on POSIX,
  ``%USERPROFILE%\\.pragma\\config.json`` on Windows.
* On POSIX the file is created with mode ``0o600`` (owner read/write
  only) so other users on the machine cannot read the API key.
* Atomic writes (write-temp + rename) so a crash mid-write cannot
  produce a corrupt config that breaks every subsequent ``pragma``
  command.
* The path can be overridden with the ``PRAGMA_USER_CONFIG`` environment
  variable, which makes the wizard trivially testable in tmpdirs and
  lets advanced users keep multiple configs.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["UserConfig", "user_config_path", "load", "save", "clear"]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def user_config_path() -> Path:
    """Return the absolute path to the user's pragma config file.

    Honours the ``PRAGMA_USER_CONFIG`` env var (used by tests and by users
    who want to keep multiple isolated configs).
    """
    override = os.environ.get("PRAGMA_USER_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".pragma" / "config.json"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class UserConfig:
    """The persisted user-level config.

    All fields are optional so a partially-configured state is
    representable -- e.g. the wizard may have saved provider+model but
    not yet the api_key, or the user may have ``unset`` a field.
    """

    provider: Optional[str] = None
    """One of ``ollama``, ``openai``, ``anthropic``, ``groq``, ``inception``."""

    model: Optional[str] = None
    """Model identifier to pass to the provider (e.g. ``gpt-4o-mini``)."""

    api_key: Optional[str] = None
    """API key for the chosen provider. ``None`` for Ollama."""

    base_url: Optional[str] = None
    """Override base URL. ``None`` falls back to the provider's default."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Forwards-compatible bag for fields added in future versions; we
    preserve unknown keys instead of dropping them on round-trip."""

    # ------------------------------------------------------------------
    # (de)serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Render to a JSON-friendly dict, dropping ``None`` values so
        the file stays minimal and diff-friendly."""
        d = asdict(self)
        extra = d.pop("extra", {}) or {}
        d = {k: v for k, v in d.items() if v is not None}
        # Merge extra at the end so first-class fields take precedence.
        merged = {**extra, **d}
        return merged

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserConfig":
        """Construct from a parsed JSON object, preserving unknown keys
        in ``extra`` so we round-trip cleanly across version upgrades."""
        if not isinstance(data, dict):
            return cls()
        known = {"provider", "model", "api_key", "base_url"}
        kwargs = {k: data.get(k) for k in known}
        kwargs["extra"] = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs)

    def is_complete(self) -> bool:
        """True if this config is sufficient to instantiate a provider.

        For Ollama we only need ``provider`` + ``model``; for everything
        else we additionally need ``api_key``.
        """
        if not self.provider or not self.model:
            return False
        if self.provider == "ollama":
            return True
        return bool(self.api_key)


# ---------------------------------------------------------------------------
# Read / write helpers
# ---------------------------------------------------------------------------


def load(path: Optional[Path] = None) -> UserConfig:
    """Load the user config from disk. Returns an empty ``UserConfig``
    if the file is missing or malformed -- never raises -- so callers
    can treat "no config yet" and "bad config" identically and prompt
    the user to (re-)run ``pragma connect``.
    """
    p = path or user_config_path()
    if not p.exists():
        return UserConfig()
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return UserConfig()
    return UserConfig.from_dict(data)


def save(cfg: UserConfig, path: Optional[Path] = None) -> Path:
    """Atomically write *cfg* to disk and return the file path.

    On POSIX the file is chmod'd ``0o600`` so other users on the
    machine cannot read the API key. The write goes to a temp file in
    the same directory and is then ``os.replace``'d into place; this is
    atomic on every platform we support and avoids leaving a
    half-written config behind on power loss.
    """
    p = path or user_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n"

    # Write to a sibling temp file then atomic-rename into place.
    fd, tmp_name = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        # Tighten perms before the rename so there is never a window in
        # which the final path is world-readable.
        try:
            os.chmod(tmp_name, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            # Best effort -- some filesystems (FAT, network mounts) do
            # not support chmod. Not fatal.
            pass
        os.replace(tmp_name, p)
    except Exception:
        # Clean up the temp file on any error so we don't litter
        # ``~/.pragma`` with stale ``.config.*.tmp`` files.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return p


def clear(path: Optional[Path] = None) -> bool:
    """Delete the user-config file. Returns True if a file was removed,
    False if there was nothing to remove."""
    p = path or user_config_path()
    if p.exists():
        p.unlink()
        return True
    return False
