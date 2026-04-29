"""Encoding-tolerant text file reading.

Notepad on Windows has a long history of saving files in UTF-16 LE or
UTF-8-with-BOM when the user picks "Unicode" or "UTF-8" from the encoding
dropdown — both of which cause naive UTF-8 readers to fail at byte zero.

This module reads any of those encodings transparently and gives a
diagnostic error message that points the user at the actual fix instead
of a raw ``UnicodeDecodeError`` traceback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# Order matters: BOMs are checked first, then encodings without BOM.
# Each tuple is (encoding_name, BOM-bytes-or-None, friendly_label).
_ENCODINGS_TO_TRY: Final[tuple[tuple[str, bytes | None, str], ...]] = (
    ("utf-8-sig", b"\xef\xbb\xbf", "UTF-8 with BOM"),
    ("utf-16",    b"\xff\xfe",     "UTF-16 LE (Windows Notepad 'Unicode')"),
    ("utf-16",    b"\xfe\xff",     "UTF-16 BE"),
    ("utf-8",     None,            "UTF-8"),
    ("cp1250",    None,            "CP1250 (legacy Windows Polish)"),
)


class FileEncodingError(RuntimeError):
    """Raised when a file cannot be decoded by any supported encoding."""


def read_text_robust(path: Path | str) -> str:
    """Read a text file, transparently handling common Windows encodings.

    Tries (in order): UTF-8 with BOM, UTF-16 LE/BE (BOM-detected), UTF-8,
    CP1250 as a last-resort fallback. Returns the decoded string with BOMs
    stripped.

    Raises
    ------
    FileEncodingError
        If no supported encoding can decode the file. The message includes
        a hint about how to re-save the file as UTF-8.
    """
    p = Path(path)
    raw = p.read_bytes()

    if not raw:
        return ""

    # Try BOM-prefixed encodings first when the BOM matches.
    for encoding, bom, _label in _ENCODINGS_TO_TRY:
        if bom is not None and raw.startswith(bom):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue  # try next

    # No BOM matched — try plain encodings.
    for encoding, bom, _label in _ENCODINGS_TO_TRY:
        if bom is not None:
            continue
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise FileEncodingError(
        f"Cannot decode {p} with any supported encoding. "
        f"Re-save the file as UTF-8 (without BOM). "
        f"In Notepad: File → Save As → Encoding: 'UTF-8'. "
        f"In VS Code: bottom-right encoding indicator → 'Save with Encoding' → 'UTF-8'."
    )


def detect_encoding_label(path: Path | str) -> str:
    """Return a human-readable label for the file's encoding.

    Useful for diagnostics and the warning shown when a non-UTF-8 file is
    successfully decoded but should be re-saved.
    """
    p = Path(path)
    raw = p.read_bytes()
    if not raw:
        return "empty"
    for _encoding, bom, label in _ENCODINGS_TO_TRY:
        if bom is not None and raw.startswith(bom):
            return label
    # Plain UTF-8 if it decodes cleanly without a BOM.
    try:
        raw.decode("utf-8")
        return "UTF-8"
    except UnicodeDecodeError:
        pass
    return "unknown"
