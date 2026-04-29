"""Load and validate application configuration.

Responsibilities
----------------
* Load ``.env`` (if present) so ``${VAR}`` references resolve.
* Read a YAML file from disk.
* Recursively resolve ``${VAR}`` and ``${VAR:-default}`` placeholders against
  ``os.environ``.
* Validate the result against :class:`AppConfig`.

Anything stricter (e.g. requiring specific env vars) is the schema's job.

Encoding handling
-----------------
Both ``.env`` and ``config.yaml`` are read via :mod:`encoding` which
transparently handles the common Windows Notepad encodings (UTF-16 LE,
UTF-8 with BOM) and emits a friendly error if a file cannot be decoded.
"""

from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from investment_copilot.config.encoding import (
    FileEncodingError,
    detect_encoding_label,
    read_text_robust,
)
from investment_copilot.config.schema import AppConfig

logger = logging.getLogger(__name__)

# Matches ${VAR} or ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


def _resolve_env_in_string(value: str) -> str:
    """Replace ``${VAR}`` / ``${VAR:-default}`` in ``value``.

    Raises ``ConfigError`` if a referenced variable has no value and no default.
    """

    def repl(match: re.Match[str]) -> str:
        var_name, default = match.group(1), match.group(2)
        env_value = os.environ.get(var_name)
        if env_value is not None and env_value != "":
            return env_value
        if default is not None:
            return default
        raise ConfigError(
            f"Environment variable '{var_name}' is referenced in config "
            f"but is not set and has no default."
        )

    return _ENV_PATTERN.sub(repl, value)


def _resolve_env(node: Any) -> Any:
    """Recursively resolve env-var placeholders inside any YAML-shaped value."""
    if isinstance(node, str):
        return _resolve_env_in_string(node)
    if isinstance(node, dict):
        return {k: _resolve_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_env(v) for v in node]
    return node


def _load_env_file(env_path: Path) -> None:
    """Load a dotenv file with encoding tolerance and inject into ``os.environ``.

    Differs from :func:`dotenv.load_dotenv` in that we read the file
    ourselves (with encoding detection) and feed a decoded stream into
    ``dotenv_values``. This avoids the ``UnicodeDecodeError`` users hit
    when Notepad saves ``.env`` as UTF-16 LE.

    Existing environment variables are not overwritten (matches
    ``load_dotenv(override=False)`` semantics).
    """
    try:
        text = read_text_robust(env_path)
    except FileEncodingError as exc:
        raise ConfigError(str(exc)) from exc

    label = detect_encoding_label(env_path)
    if label not in {"UTF-8", "empty"}:
        logger.warning(
            ".env was decoded as %s. Re-save it as UTF-8 (without BOM) "
            "to silence this warning.",
            label,
        )

    values = dotenv_values(stream=io.StringIO(text))
    for key, value in values.items():
        if value is None:
            continue
        os.environ.setdefault(key, value)


def load_config(
    config_path: str | Path = "config.yaml",
    *,
    env_file: str | Path | None = ".env",
) -> AppConfig:
    """Load, resolve, and validate the application configuration.

    Parameters
    ----------
    config_path:
        Path to ``config.yaml``.
    env_file:
        Path to a dotenv file. If it exists, it is loaded before placeholder
        resolution. Pass ``None`` to skip dotenv loading entirely (useful in
        tests or in a deployed environment that injects env vars directly).
    """
    if env_file is not None:
        env_path = Path(env_file)
        if env_path.is_file():
            _load_env_file(env_path)

    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")

    try:
        text = read_text_robust(cfg_path)
    except FileEncodingError as exc:
        raise ConfigError(str(exc)) from exc

    label = detect_encoding_label(cfg_path)
    if label not in {"UTF-8", "empty"}:
        logger.warning(
            "%s was decoded as %s. Re-save it as UTF-8 (without BOM) "
            "to silence this warning.",
            cfg_path,
            label,
        )

    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {cfg_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

    resolved = _resolve_env(raw)

    try:
        return AppConfig.model_validate(resolved)
    except Exception as exc:  # pydantic.ValidationError -> ConfigError
        raise ConfigError(f"Invalid configuration: {exc}") from exc
