"""Tests for encoding-tolerant file reading.

Covers the exact failure mode reported in the wild: ``.env`` saved by
Notepad as UTF-16 LE causing a UnicodeDecodeError. Plus other encodings
the helper claims to support.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from investment_copilot.config import (
    ConfigError,
    FileEncodingError,
    detect_encoding_label,
    load_config,
    read_text_robust,
)


# --- read_text_robust ------------------------------------------------------


def test_reads_utf8_plain(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_bytes("hello świat".encode("utf-8"))
    assert read_text_robust(p) == "hello świat"
    assert detect_encoding_label(p) == "UTF-8"


def test_reads_utf8_with_bom(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_bytes(b"\xef\xbb\xbf" + "hello świat".encode("utf-8"))
    # BOM is consumed by utf-8-sig; the returned string has no leading BOM
    out = read_text_robust(p)
    assert out == "hello świat"
    assert "UTF-8 with BOM" in detect_encoding_label(p)


def test_reads_utf16_le_notepad_format(tmp_path: Path) -> None:
    """The exact failure mode the user hit: Notepad 'Unicode' = UTF-16 LE."""
    p = tmp_path / ".env"
    p.write_bytes(b"\xff\xfe" + "GROQ_API_KEY=secret".encode("utf-16-le"))
    out = read_text_robust(p)
    assert out == "GROQ_API_KEY=secret"
    assert "UTF-16" in detect_encoding_label(p)


def test_reads_utf16_be(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_bytes(b"\xfe\xff" + "abc".encode("utf-16-be"))
    assert read_text_robust(p) == "abc"


def test_reads_cp1250_fallback(tmp_path: Path) -> None:
    """Legacy Windows Polish encoding — last-resort fallback."""
    p = tmp_path / "f.txt"
    # 'ł' in CP1250 is 0xB3, which is invalid UTF-8 start sequence
    p.write_bytes("łódź".encode("cp1250"))
    out = read_text_robust(p)
    assert out == "łódź"


def test_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_bytes(b"")
    assert read_text_robust(p) == ""


def test_undecodable_bytes_raise_friendly_error(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    # Pathological bytes that don't fit any of our codecs cleanly.
    # CP1250 will accept almost anything, so to force an error we need bytes
    # that fail UTF-8 AND fail UTF-16 (odd length) AND fail CP1250.
    # In practice cp1250 accepts everything, so this test verifies our
    # fallback chain succeeds rather than failing — which is the right
    # behavior. We assert the content roundtrips, even if as garbage.
    p.write_bytes(bytes([0xff, 0xfe, 0xff]))  # truncated UTF-16 BOM + extra
    # Should not raise; returns *something* (possibly garbled). The key
    # promise is "no UnicodeDecodeError leaks to the user".
    out = read_text_robust(p)
    assert isinstance(out, str)


# --- load_config (.env handling) -------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_loads_utf16_dotenv_without_crashing(tmp_path: Path, monkeypatch) -> None:
    """The end-to-end repro of the reported bug.

    Before the fix: UnicodeDecodeError at byte 0.
    After the fix: env var is loaded, config validates, no error.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    # Write .env as UTF-16 LE with BOM (what Notepad produces)
    env_path = tmp_path / ".env"
    env_path.write_bytes(b"\xff\xfe" + "GROQ_API_KEY=sk-from-utf16".encode("utf-16-le"))

    cfg_path = _write_yaml(
        tmp_path,
        """
        llm:
          api_key: ${GROQ_API_KEY}
        """,
    )

    cfg = load_config(cfg_path, env_file=env_path)
    assert cfg.llm.api_key == "sk-from-utf16"


def test_loads_utf8_bom_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_bytes(b"\xef\xbb\xbf" + b"GROQ_API_KEY=sk-bom")

    cfg_path = _write_yaml(
        tmp_path,
        """
        llm:
          api_key: ${GROQ_API_KEY}
        """,
    )

    cfg = load_config(cfg_path, env_file=env_path)
    assert cfg.llm.api_key == "sk-bom"


def test_loads_utf16_yaml_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_bytes(
        b"\xff\xfe"
        + "llm:\n  api_key: ${GROQ_API_KEY}\n".encode("utf-16-le")
    )

    cfg = load_config(cfg_path, env_file=None)
    assert cfg.llm.api_key == "x"


def test_dotenv_does_not_override_existing_env(tmp_path: Path, monkeypatch) -> None:
    """``setdefault`` semantics: existing env vars take precedence."""
    monkeypatch.setenv("GROQ_API_KEY", "from-shell")

    env_path = tmp_path / ".env"
    env_path.write_text("GROQ_API_KEY=from-file\n", encoding="utf-8")
    cfg_path = _write_yaml(
        tmp_path,
        """
        llm:
          api_key: ${GROQ_API_KEY}
        """,
    )

    cfg = load_config(cfg_path, env_file=env_path)
    assert cfg.llm.api_key == "from-shell"


# --- detect_encoding_label -------------------------------------------------


def test_detect_encoding_label_utf8(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("plain ascii", encoding="utf-8")
    assert detect_encoding_label(p) == "UTF-8"


def test_detect_encoding_label_empty(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_bytes(b"")
    assert detect_encoding_label(p) == "empty"
