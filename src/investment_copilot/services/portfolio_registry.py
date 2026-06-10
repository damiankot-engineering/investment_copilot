"""Registry over the user's portfolio YAML files.

Multi-portfolio support. The configured ``portfolio.path`` is the **default**
portfolio (id ``"default"``); any additional portfolios live as ``<id>.yaml``
files under ``portfolio.dir`` (default ``portfolios/``), where the filename
stem is the id.

The registry is the single place that maps an id → file path and performs
file-level CRUD. It reuses :func:`load_portfolio` / :func:`save_portfolio`, so
all domain validation (ticker normalization, FIFO, no-duplicate tickers) still
runs on every write. Deletes are **soft** — the file is moved to
``<dir>/.trash/`` rather than removed — because portfolio files are
hand-authored and irreplaceable.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from investment_copilot.domain.portfolio import Portfolio
from investment_copilot.services.portfolio_service import (
    PortfolioError,
    load_portfolio,
    save_portfolio,
)

logger = logging.getLogger(__name__)

DEFAULT_ID: str = "default"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_ACCOUNT_TYPES: tuple[str, ...] = ("standard", "ike", "ikze")
_UNSET: object = object()  # sentinel: distinguishes "field omitted" from "set to None"


class PortfolioNotFoundError(KeyError):
    """Raised when a portfolio id does not resolve to a file. API → 404."""


class PortfolioRegistryError(ValueError):
    """Raised on an invalid id or a create/duplicate name clash. API → 400/409."""


@dataclass(frozen=True, slots=True)
class PortfolioRef:
    """A discoverable portfolio: its id, label, file path, and a cheap summary."""

    id: str
    name: str | None
    path: Path
    is_default: bool
    n_holdings: int
    account_type: str = "standard"


def validate_portfolio_id(pid: str) -> str:
    """Normalize + validate a user-supplied portfolio id (filename stem)."""
    pid = (pid or "").strip().lower()
    if pid == DEFAULT_ID:
        raise PortfolioRegistryError("'default' is reserved")
    if not _ID_RE.match(pid):
        raise PortfolioRegistryError(
            "id must be 1–64 chars: lowercase letters, digits, '-' or '_', "
            "starting with a letter or digit"
        )
    return pid


class PortfolioRegistry:
    """Discover and manage portfolio YAML files."""

    def __init__(self, portfolios_dir: Path | str, default_path: Path | str) -> None:
        self._dir = Path(portfolios_dir)
        self._default = Path(default_path)

    # ---------------------------------------------------------------- Discovery

    @staticmethod
    def _read_meta(path: Path) -> tuple[str | None, int, str]:
        """Cheap ``(name, n_holdings, account_type)`` read — never raises."""
        if not path.is_file():
            return None, 0, "standard"
        try:
            from investment_copilot.config.encoding import read_text_robust

            raw = yaml.safe_load(read_text_robust(path)) or {}
            if not isinstance(raw, dict):
                return None, 0, "standard"
            name = raw.get("name")
            holdings = raw.get("holdings") or []
            account = raw.get("account_type") or "standard"
            return (
                str(name) if name else None,
                len(holdings) if isinstance(holdings, list) else 0,
                str(account),
            )
        except Exception as exc:  # noqa: BLE001 - listing must not fault on one bad file
            logger.warning("Could not read portfolio meta from %s: %s", path, exc)
            return None, 0, "standard"

    def _ref(self, pid: str, path: Path, *, is_default: bool) -> PortfolioRef:
        name, n, account = self._read_meta(path)
        return PortfolioRef(
            id=pid, name=name, path=path, is_default=is_default,
            n_holdings=n, account_type=account,
        )

    def list(self) -> list[PortfolioRef]:
        """Default entry first, then ``portfolios/*.yaml`` sorted by id."""
        refs = [self._ref(DEFAULT_ID, self._default, is_default=True)]
        if self._dir.is_dir():
            for p in sorted(self._dir.glob("*.yaml")):
                stem = p.stem.lower()
                if stem == DEFAULT_ID or stem.startswith("."):
                    continue  # never shadow the default; skip dotfiles
                refs.append(self._ref(stem, p, is_default=False))
        return refs

    def resolve(self, pid: str | None) -> PortfolioRef:
        """Map an id to a :class:`PortfolioRef`. ``None``/empty → default."""
        if not pid or pid.lower() == DEFAULT_ID:
            return self._ref(DEFAULT_ID, self._default, is_default=True)
        target = pid.lower()
        for ref in self.list():
            if ref.id == target:
                return ref
        raise PortfolioNotFoundError(f"Unknown portfolio id: {pid!r}")

    def path_for(self, pid: str | None) -> Path:
        return self.resolve(pid).path

    # -------------------------------------------------------------------- CRUD

    def _path_for_new(self, pid: str) -> Path:
        path = self._dir / f"{pid}.yaml"
        if path.exists():
            raise PortfolioRegistryError(f"Portfolio {pid!r} already exists")
        return path

    def create(
        self,
        pid: str,
        *,
        name: str | None = None,
        base_currency: str = "PLN",
        account_type: str = "standard",
    ) -> PortfolioRef:
        """Create a new empty portfolio file ``<dir>/<pid>.yaml``."""
        pid = validate_portfolio_id(pid)
        if account_type not in _ACCOUNT_TYPES:
            raise PortfolioRegistryError(f"Invalid account_type: {account_type!r}")
        path = self._path_for_new(pid)
        portfolio = Portfolio(
            name=name or None,
            account_type=account_type,
            base_currency=base_currency,
            holdings=[],
        )
        save_portfolio(portfolio, path)
        logger.info("Created portfolio %s at %s", pid, path)
        return self._ref(pid, path, is_default=False)

    def update_meta(
        self,
        pid: str,
        *,
        name: str | None | object = _UNSET,
        account_type: str | object = _UNSET,
    ) -> PortfolioRef:
        """Patch a portfolio's metadata in place (label and/or account type).

        Only the keyword arguments actually passed are written — the rest keep
        the ``_UNSET`` sentinel — so changing the account type never clears the
        name and vice-versa. Works for the **default** portfolio too, since
        :meth:`resolve` maps it to the configured ``portfolio.path``.
        """
        ref = self.resolve(pid)
        if not ref.path.is_file():
            raise PortfolioNotFoundError(f"Portfolio file missing: {ref.path}")
        updates: dict[str, object] = {}
        if name is not _UNSET:
            updates["name"] = name or None
        if account_type is not _UNSET:
            if account_type not in _ACCOUNT_TYPES:
                raise PortfolioRegistryError(f"Invalid account_type: {account_type!r}")
            updates["account_type"] = account_type
        if updates:
            portfolio = load_portfolio(ref.path)
            save_portfolio(portfolio.model_copy(update=updates), ref.path)
            logger.info("Updated portfolio %s meta: %s", ref.id, sorted(updates))
        return self._ref(ref.id, ref.path, is_default=ref.is_default)

    def rename(self, pid: str, name: str | None) -> PortfolioRef:
        """Set a portfolio's display label (id/filename are unchanged)."""
        return self.update_meta(pid, name=name)

    def duplicate(
        self, src_id: str, new_id: str, *, name: str | None = None
    ) -> PortfolioRef:
        """Copy an existing portfolio's holdings into a new ``<new_id>.yaml``."""
        src = self.resolve(src_id)
        new_id = validate_portfolio_id(new_id)
        path = self._path_for_new(new_id)
        portfolio = load_portfolio(src.path)
        label = name if name is not None else f"{src.name or src.id} (kopia)"
        save_portfolio(portfolio.model_copy(update={"name": label or None}), path)
        logger.info("Duplicated portfolio %s -> %s", src.id, new_id)
        return self._ref(new_id, path, is_default=False)

    def delete(self, pid: str) -> None:
        """Soft-delete: move ``<id>.yaml`` to ``<dir>/.trash/``. Never the default."""
        ref = self.resolve(pid)
        if ref.is_default:
            raise PortfolioRegistryError("The default portfolio cannot be deleted")
        if not ref.path.is_file():
            raise PortfolioNotFoundError(f"Portfolio file missing: {ref.path}")
        trash = self._dir / ".trash"
        trash.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        dest = trash / f"{ref.id}-{ts}.yaml"
        try:
            shutil.move(str(ref.path), str(dest))
        except OSError as exc:
            raise PortfolioError(f"Failed to move {ref.path} to trash: {exc}") from exc
        logger.info("Soft-deleted portfolio %s -> %s", ref.id, dest)
