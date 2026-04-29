"""Build :class:`LLMClient` from configuration."""

from __future__ import annotations

from investment_copilot.config.schema import LLMConfig
from investment_copilot.infrastructure.llm.base import LLMClient
from investment_copilot.infrastructure.llm.groq_client import GroqClient


def build_llm_client(cfg: LLMConfig) -> LLMClient:
    """Construct the LLM client selected by ``cfg.provider``."""
    if cfg.provider == "groq":
        return GroqClient(
            api_key=cfg.api_key,
            default_model=cfg.model_analysis,
            default_temperature=cfg.temperature,
            default_max_tokens=cfg.max_tokens,
            request_timeout_s=cfg.request_timeout_s,
        )
    raise ValueError(f"Unsupported LLM provider: {cfg.provider!r}")
