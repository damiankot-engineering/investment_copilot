"""Tests for ``GroqClient`` — SDK is fully stubbed; no network calls."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from investment_copilot.infrastructure.llm.base import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMValidationError,
)
from investment_copilot.infrastructure.llm.groq_client import GroqClient


# --- Test schema -----------------------------------------------------------


class Reply(BaseModel):
    summary: str = Field(min_length=1)
    score: int = Field(ge=0, le=10)
    risks: list[str] = Field(default_factory=list)


# --- Helpers ---------------------------------------------------------------


def _stub_completion(content: str) -> SimpleNamespace:
    """Build an OpenAI-shaped response with a single choice."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _stub_client(*responses):
    """Create a fake Groq client whose `chat.completions.create` returns ``responses`` in order.

    A response can be a string (turned into a stub completion), a SimpleNamespace,
    or an Exception instance to be raised.
    """
    client = MagicMock()
    side_effects = []
    for r in responses:
        if isinstance(r, Exception):
            side_effects.append(r)
        elif isinstance(r, str):
            side_effects.append(_stub_completion(r))
        else:
            side_effects.append(r)
    client.chat.completions.create.side_effect = side_effects
    return client


def _make(client) -> GroqClient:
    return GroqClient(
        api_key="sk-test",
        default_model="llama-3.3-70b-versatile",
        client=client,
    )


# --- Constructor -----------------------------------------------------------


def test_requires_api_key() -> None:
    with pytest.raises(LLMAuthError):
        GroqClient(api_key="", default_model="x")


# --- Happy path: structured output ----------------------------------------


def test_complete_structured_returns_validated_model() -> None:
    payload = json.dumps({"summary": "ok", "score": 7, "risks": ["a", "b"]})
    client = _stub_client(payload)
    groq = _make(client)

    out = groq.complete_structured(
        system_prompt="you are a copilot",
        user_prompt="analyze",
        response_schema=Reply,
    )

    assert isinstance(out, Reply)
    assert out.summary == "ok"
    assert out.score == 7
    assert out.risks == ["a", "b"]
    # JSON mode kwargs were passed
    call = client.chat.completions.create.call_args
    assert call.kwargs["response_format"] == {"type": "json_object"}
    # Schema appears in the system prompt
    sys_msg = call.kwargs["messages"][0]["content"]
    assert "JSON Schema" in sys_msg
    assert '"summary"' in sys_msg


def test_complete_structured_strips_code_fences() -> None:
    payload = "```json\n" + json.dumps({"summary": "x", "score": 3}) + "\n```"
    groq = _make(_stub_client(payload))
    out = groq.complete_structured(
        system_prompt="s",
        user_prompt="u",
        response_schema=Reply,
    )
    assert out.summary == "x"


def test_complete_structured_self_corrects_on_validation_error() -> None:
    bad = json.dumps({"summary": "", "score": 5})  # min_length=1 fails
    good = json.dumps({"summary": "ok", "score": 5})
    client = _stub_client(bad, good)
    groq = _make(client)

    out = groq.complete_structured(
        system_prompt="s",
        user_prompt="u",
        response_schema=Reply,
    )

    assert out.summary == "ok"
    assert client.chat.completions.create.call_count == 2
    # The corrective second call must include the assistant's bad output
    # plus a user message asking for valid JSON.
    second_call_msgs = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    roles = [m["role"] for m in second_call_msgs]
    assert roles[-2:] == ["assistant", "user"]
    assert "did not conform" in second_call_msgs[-1]["content"]


def test_complete_structured_raises_after_failed_self_correction() -> None:
    bad1 = json.dumps({"summary": "", "score": 5})
    bad2 = "still not JSON"
    groq = _make(_stub_client(bad1, bad2))
    with pytest.raises(LLMValidationError):
        groq.complete_structured(
            system_prompt="s",
            user_prompt="u",
            response_schema=Reply,
        )


def test_complete_structured_raises_on_invalid_json() -> None:
    bad1 = "not json at all"
    bad2 = "still not json"
    groq = _make(_stub_client(bad1, bad2))
    with pytest.raises(LLMValidationError, match="not valid JSON"):
        groq.complete_structured(
            system_prompt="s",
            user_prompt="u",
            response_schema=Reply,
        )


# --- complete_text ---------------------------------------------------------


def test_complete_text_returns_raw_string() -> None:
    client = _stub_client("hello world")
    groq = _make(client)
    out = groq.complete_text(system_prompt="s", user_prompt="u")
    assert out == "hello world"
    # JSON mode is NOT enabled
    assert "response_format" not in client.chat.completions.create.call_args.kwargs


# --- Retries ---------------------------------------------------------------


def _named_exc(name: str, message: str = "boom") -> Exception:
    """Forge an exception whose class name matches Groq SDK error names."""
    cls = type(name, (Exception,), {})
    return cls(message)


def test_transient_error_is_retried_and_eventually_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(
        "investment_copilot.infrastructure.llm.groq_client.time.sleep", lambda *_: None
    )
    payload = json.dumps({"summary": "ok", "score": 1})
    client = _stub_client(_named_exc("RateLimitError"), payload)
    groq = _make(client)

    out = groq.complete_structured(
        system_prompt="s",
        user_prompt="u",
        response_schema=Reply,
    )
    assert out.summary == "ok"
    assert client.chat.completions.create.call_count == 2


def test_transient_error_exhausts_retries(monkeypatch) -> None:
    monkeypatch.setattr(
        "investment_copilot.infrastructure.llm.groq_client.time.sleep", lambda *_: None
    )
    client = _stub_client(
        _named_exc("APIConnectionError"),
        _named_exc("APIConnectionError"),
        _named_exc("APIConnectionError"),
    )
    groq = GroqClient(
        api_key="sk-test",
        default_model="x",
        client=client,
        max_network_retries=3,
    )
    with pytest.raises(LLMRateLimitError):
        groq.complete_text(system_prompt="s", user_prompt="u")


def test_auth_error_does_not_retry() -> None:
    client = _stub_client(_named_exc("AuthenticationError", "bad key"))
    groq = _make(client)
    with pytest.raises(LLMAuthError):
        groq.complete_text(system_prompt="s", user_prompt="u")
    assert client.chat.completions.create.call_count == 1


def test_status_code_429_is_transient(monkeypatch) -> None:
    monkeypatch.setattr(
        "investment_copilot.infrastructure.llm.groq_client.time.sleep", lambda *_: None
    )
    err = Exception("rate limited")
    err.status_code = 429  # type: ignore[attr-defined]
    payload = json.dumps({"summary": "ok", "score": 1})
    client = _stub_client(err, payload)
    groq = _make(client)
    out = groq.complete_structured(
        system_prompt="s", user_prompt="u", response_schema=Reply
    )
    assert out.summary == "ok"


def test_status_code_400_is_fatal() -> None:
    err = Exception("bad request")
    err.status_code = 400  # type: ignore[attr-defined]
    client = _stub_client(err)
    groq = _make(client)
    with pytest.raises(LLMError):
        groq.complete_text(system_prompt="s", user_prompt="u")
    assert client.chat.completions.create.call_count == 1


# --- Response shape edge cases --------------------------------------------


def test_unexpected_response_shape_raises() -> None:
    bad_response = SimpleNamespace(choices=[])  # no message
    client = MagicMock()
    client.chat.completions.create.return_value = bad_response
    groq = _make(client)
    with pytest.raises(LLMError):
        groq.complete_text(system_prompt="s", user_prompt="u")


def test_none_content_raises() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    groq = _make(client)
    with pytest.raises(LLMError, match="no message content"):
        groq.complete_text(system_prompt="s", user_prompt="u")


# --- Defaults & overrides -------------------------------------------------


def test_overrides_pass_through() -> None:
    payload = json.dumps({"summary": "ok", "score": 1})
    client = _stub_client(payload)
    groq = _make(client)

    groq.complete_structured(
        system_prompt="s",
        user_prompt="u",
        response_schema=Reply,
        model="custom-model",
        temperature=0.7,
        max_tokens=500,
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "custom-model"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 500
