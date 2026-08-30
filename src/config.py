"""Configuration loader for TTS Baseline project."""

import os
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Hierarchical configuration object backed by a YAML file.

    Attributes are accessible via dot-notation on nested dicts that are
    promoted to :class:`Config` instances transparently.
    """

    def __init__(self, data: dict) -> None:
        self._data: dict = data

    # ------------------------------------------------------------------
    # Attribute access helpers
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* or *default* if absent."""
        value = self._data.get(key, default)
        if isinstance(value, dict):
            return Config(value)
        return value

    def to_dict(self) -> dict:
        """Return the underlying plain dictionary."""
        return self._data

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self._data!r})"


def load_config(config_path: str | Path | None = None) -> Config:
    """Load YAML configuration from *config_path*.

    If *config_path* is ``None`` the function searches for
    ``configs/config.yaml`` relative to the project root (two levels above
    this file).

    Parameters
    ----------
    config_path:
        Explicit path to a YAML config file, or ``None`` to use the default.

    Returns
    -------
    Config
        Populated :class:`Config` instance.

    Raises
    ------
    FileNotFoundError
        If the resolved config file does not exist.
    """
    if config_path is None:
        # Project root is two directories above src/config.py
        project_root = Path(__file__).parent.parent
        config_path = project_root / "configs" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    # Expand environment-variable overrides: TTS_<SECTION>_<KEY>=value
    env_overrides: dict = {}
    prefix = "TTS_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("_", 1)
        if len(parts) == 2:
            section, key = parts
            env_overrides.setdefault(section, {})[key] = env_val

    merged = _deep_merge(raw, env_overrides)
    return Config(merged)
