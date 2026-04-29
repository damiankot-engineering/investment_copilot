"""Groq adapter implementing :class:`LLMClient`.

Uses the official ``groq`` SDK's chat-completions endpoint with JSON mode
enabled for structured outputs. The response schema (a Pydantic model) is
serialized to JSON Schema and embedded in the system prompt so the model
knows the target shape; the response is then validated against the same
schema on the way out.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from investment_copilot.infrastructure.llm.base import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMValidationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# --- Tunables (overridable via constructor) ---------------------------------


_MAX_NETWORK_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 30.0


# --- Adapter ----------------------------------------------------------------


class GroqClient:
    """Groq chat-completions adapter."""

    name: str = "groq"

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str,
        default_temperature: float = 0.3,
        default_max_tokens: int = 2048,
        request_timeout_s: int = 60,
        client: Any | None = None,
        max_network_retries: int = _MAX_NETWORK_RETRIES,
    ) -> None:
        if not api_key:
            raise LLMAuthError("Groq api_key is required")
        self._api_key = api_key
        self._default_model = default_model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._timeout_s = request_timeout_s
        self._max_network_retries = max(1, int(max_network_retries))
        self._client = client  # injected for tests; lazily built otherwise

    # -- LLMClient surface --------------------------------------------------

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
        schema = _pydantic_to_json_schema(response_schema)
        system_full = _augment_system_for_json(system_prompt, schema)

        messages = [
            {"role": "system", "content": system_full},
            {"role": "user", "content": user_prompt},
        ]

        text = self._call_with_retries(
            messages=messages,
            model=model or self._default_model,
            temperature=self._coerce_temperature(temperature),
            max_tokens=max_tokens or self._default_max_tokens,
            json_mode=True,
        )

        try:
            return _parse_into_schema(text, response_schema)
        except LLMValidationError as exc:
            logger.warning("Groq JSON failed validation; requesting self-correction.")
            corrective = (
                "Your previous response did not conform to the required JSON "
                "schema. Validation error:\n"
                f"{exc}\n\n"
                "Return ONLY a JSON object that strictly matches the schema. "
                "Do not include any prose, code fences, or commentary."
            )
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": corrective})
            text = self._call_with_retries(
                messages=messages,
                model=model or self._default_model,
                temperature=self._coerce_temperature(temperature),
                max_tokens=max_tokens or self._default_max_tokens,
                json_mode=True,
            )
            return _parse_into_schema(text, response_schema)

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_with_retries(
            messages=messages,
            model=model or self._default_model,
            temperature=self._coerce_temperature(temperature),
            max_tokens=max_tokens or self._default_max_tokens,
            json_mode=False,
        )

    # -- Internals ----------------------------------------------------------

    def _coerce_temperature(self, t: float | None) -> float:
        return self._default_temperature if t is None else float(t)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from groq import Groq  # local import; SDK optional at import time
        except ImportError as exc:  # pragma: no cover - ergonomic
            raise LLMError(
                "The 'groq' package is not installed. Run `uv pip install groq`."
            ) from exc
        self._client = Groq(api_key=self._api_key, timeout=self._timeout_s)
        return self._client

    def _call_with_retries(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(1, self._max_network_retries + 1):
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                category = _classify_groq_error(exc)
                if category == "auth":
                    raise LLMAuthError(f"Groq auth error: {exc}") from exc
                if category == "fatal":
                    raise LLMError(f"Groq error: {exc}") from exc

                # transient / rate-limit
                last_exc = exc
                if attempt >= self._max_network_retries:
                    break
                delay = _backoff_delay(attempt)
                logger.warning(
                    "Groq transient error (%s) attempt %d/%d, sleeping %.2fs",
                    type(exc).__name__,
                    attempt,
                    self._max_network_retries,
                    delay,
                )
                time.sleep(delay)
                continue

            return _extract_first_choice_text(response)

        raise LLMRateLimitError(
            f"Groq retries exhausted ({self._max_network_retries}): {last_exc}"
        ) from last_exc


# --- helpers ----------------------------------------------------------------


def _pydantic_to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the model's JSON Schema, with $defs inlined where Groq prefers."""
    return model.model_json_schema()


def _augment_system_for_json(system_prompt: str, schema: dict[str, Any]) -> str:
    schema_block = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        f"{system_prompt}\n\n"
        "You MUST respond with a single valid JSON object that conforms "
        "exactly to the following JSON Schema. Do not include code fences, "
        "explanations, or any text outside the JSON object.\n\n"
        f"```json\n{schema_block}\n```"
    )


def _parse_into_schema(text: str, schema: type[T]) -> T:
    cleaned = _strip_json_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMValidationError(f"Response was not valid JSON: {exc}") from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise LLMValidationError(f"JSON did not match schema: {exc}") from exc


def _strip_json_fences(text: str) -> str:
    """Defensively strip ``` ```json fences in case the model emits them."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _extract_first_choice_text(response: Any) -> str:
    """Pull the assistant message out of an OpenAI-shaped response."""
    try:
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None)
    except (AttributeError, IndexError) as exc:
        raise LLMError(f"Unexpected Groq response shape: {response!r}") from exc
    if content is None:
        raise LLMError("Groq response had no message content")
    return str(content)


def _classify_groq_error(exc: Exception) -> str:
    """Classify a Groq SDK exception as 'auth', 'transient', or 'fatal'.

    The Groq SDK is OpenAI-shaped; we match by class name to avoid a hard
    dependency on it at import time, which keeps unit tests SDK-free.
    """
    cls_name = type(exc).__name__
    if cls_name in {"AuthenticationError", "PermissionDeniedError"}:
        return "auth"
    if cls_name in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
    }:
        return "transient"
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in (401, 403):
            return "auth"
        if status == 429 or status >= 500:
            return "transient"
        if 400 <= status < 500:
            return "fatal"
    # Network-y standard library errors should retry
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient"
    return "fatal"


def _backoff_delay(attempt: int) -> float:
    base = min(_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)
    jitter = random.uniform(0, base * 0.25)
    return base + jitter
