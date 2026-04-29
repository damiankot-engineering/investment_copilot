"""Configuration loading and validation."""

from investment_copilot.config.encoding import (
    FileEncodingError,
    detect_encoding_label,
    read_text_robust,
)
from investment_copilot.config.loader import ConfigError, load_config
from investment_copilot.config.schema import AppConfig

__all__ = [
    "AppConfig",
    "ConfigError",
    "FileEncodingError",
    "detect_encoding_label",
    "load_config",
    "read_text_robust",
]
