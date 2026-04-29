"""LLM client interface.

The Protocol below is the only thing the application depends on. Concrete
adapters (Groq, OpenAI, Anthropic, …) live alongside it; swapping is one
file plus a factory branch.

The structured-output path :meth:`LLMClient.complete_structured` is the
core surface used by the copilot. It guarantees a typed ``BaseModel``
return or raises :class:`LLMError` — callers never deal with raw strings
or JSON parsing.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base class for all LLM-client failures."""


class LLMRateLimitError(LLMError):
    """Raised after exhausting retries on rate-limit or transient server errors."""


class LLMAuthError(LLMError):
    """Raised on credential / auth failures (no retry)."""


class LLMValidationError(LLMError):
    """Raised when the model's JSON cannot be coerced into the target schema."""


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic LLM access. Synchronous; structured-output focused."""

    name: str

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[T],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Return a validated instance of ``response_schema``.

        Implementations must:
        * pass a JSON-schema description of ``response_schema`` to the
          model so it knows the target shape;
        * enable JSON mode (or equivalent) so the response is parseable;
        * validate the response against the schema and raise
          :class:`LLMValidationError` on failure (after at most one
          self-correction retry).
        """
        ...

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return free-form text. Provided as a low-level escape hatch.

        Prefer :meth:`complete_structured` for anything routed into the rest
        of the application.
        """
        ...
