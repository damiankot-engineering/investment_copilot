"""Tests for the multi-portfolio registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from investment_copilot.services.portfolio_registry import (
    DEFAULT_ID,
    PortfolioNotFoundError,
    PortfolioRegistry,
    PortfolioRegistryError,
    validate_portfolio_id,
)
from investment_copilot.services.portfolio_service import load_portfolio


def _make(tmp_path: Path, *, with_default: bool = True) -> PortfolioRegistry:
    default_path = tmp_path / "portfolio.yaml"
    if with_default:
        default_path.write_text(
            "name: Główny\nbase_currency: PLN\nholdings: []\n", encoding="utf-8"
        )
    return PortfolioRegistry(portfolios_dir=tmp_path / "portfolios", default_path=default_path)


# --- discovery / resolve ----------------------------------------------------


def test_list_default_only(tmp_path) -> None:
    reg = _make(tmp_path)
    refs = reg.list()
    assert [r.id for r in refs] == [DEFAULT_ID]
    assert refs[0].is_default is True
    assert refs[0].name == "Główny"


def test_list_default_when_file_absent(tmp_path) -> None:
    reg = _make(tmp_path, with_default=False)
    refs = reg.list()
    assert refs[0].id == DEFAULT_ID and refs[0].n_holdings == 0 and refs[0].name is None


def test_create_then_list_and_resolve(tmp_path) -> None:
    reg = _make(tmp_path)
    ref = reg.create("ike", name="IKE")
    assert ref.id == "ike" and ref.name == "IKE" and not ref.is_default
    # id is the filename stem
    assert ref.path == tmp_path / "portfolios" / "ike.yaml"

    ids = [r.id for r in reg.list()]
    assert ids == [DEFAULT_ID, "ike"]  # default first, then sorted

    assert reg.resolve("ike").path == ref.path
    assert reg.resolve(None).id == DEFAULT_ID
    assert reg.resolve("default").id == DEFAULT_ID


def test_resolve_unknown_raises(tmp_path) -> None:
    reg = _make(tmp_path)
    with pytest.raises(PortfolioNotFoundError):
        reg.resolve("nope")


def test_id_derived_from_stem(tmp_path) -> None:
    reg = _make(tmp_path)
    (tmp_path / "portfolios").mkdir(parents=True)
    (tmp_path / "portfolios" / "spek.yaml").write_text(
        "base_currency: USD\nholdings: []\n", encoding="utf-8"
    )
    assert reg.resolve("spek").id == "spek"
    assert any(r.id == "spek" for r in reg.list())


# --- validation -------------------------------------------------------------


def test_validate_id_rejects_reserved_and_bad(tmp_path) -> None:
    assert validate_portfolio_id("IKE") == "ike"  # normalized lower
    for bad in ["default", "", "a b", "Ćma", "-x", "x" * 65]:
        with pytest.raises(PortfolioRegistryError):
            validate_portfolio_id(bad)


def test_create_duplicate_id_conflicts(tmp_path) -> None:
    reg = _make(tmp_path)
    reg.create("ike")
    with pytest.raises(PortfolioRegistryError):
        reg.create("ike")


# --- rename / duplicate / delete -------------------------------------------


def test_rename_sets_label_only(tmp_path) -> None:
    reg = _make(tmp_path)
    reg.create("ike", name="IKE")
    reg.rename("ike", "IKE Plus")
    assert reg.resolve("ike").name == "IKE Plus"
    # file/id unchanged
    assert reg.resolve("ike").path == tmp_path / "portfolios" / "ike.yaml"


# --- account type (IKE/IKZE tax exemption) ---------------------------------


def test_set_account_type_on_default(tmp_path) -> None:
    reg = _make(tmp_path)
    assert reg.resolve(DEFAULT_ID).account_type == "standard"
    ref = reg.update_meta(DEFAULT_ID, account_type="ike")
    assert ref.account_type == "ike"
    # persisted + label preserved (changing the type must not wipe the name)
    again = reg.resolve(DEFAULT_ID)
    assert again.account_type == "ike" and again.name == "Główny"
    assert load_portfolio(reg.path_for(DEFAULT_ID)).is_tax_exempt is True


def test_set_account_type_preserves_name(tmp_path) -> None:
    reg = _make(tmp_path)
    reg.create("ike", name="IKE")
    reg.update_meta("ike", account_type="ikze")
    ref = reg.resolve("ike")
    assert ref.account_type == "ikze" and ref.name == "IKE"


def test_update_meta_rejects_bad_account_type(tmp_path) -> None:
    reg = _make(tmp_path)
    with pytest.raises(PortfolioRegistryError):
        reg.update_meta(DEFAULT_ID, account_type="roth")


def test_update_meta_name_only_keeps_account_type(tmp_path) -> None:
    reg = _make(tmp_path)
    reg.create("ike", name="IKE", account_type="ike")
    reg.update_meta("ike", name="IKE Plus")  # touch name only
    ref = reg.resolve("ike")
    assert ref.name == "IKE Plus" and ref.account_type == "ike"


def test_duplicate_copies_holdings(tmp_path) -> None:
    reg = _make(tmp_path)
    src = reg.create("ike", name="IKE")
    # add a holding to the source file
    src.path.write_text(
        "name: IKE\nbase_currency: PLN\nholdings:\n"
        "- ticker: cdr.pl\n  thesis: t\n"
        "  transactions:\n  - date: '2024-01-02'\n    action: BUY\n"
        "    shares: 10\n    price_per_share: 100\n",
        encoding="utf-8",
    )
    dup = reg.duplicate("ike", "ike2")
    assert dup.id == "ike2"
    copied = load_portfolio(dup.path)
    assert [h.ticker for h in copied.holdings] == ["cdr.pl"]
    assert copied.name == "IKE (kopia)"


def test_delete_is_soft_and_guards_default(tmp_path) -> None:
    reg = _make(tmp_path)
    reg.create("ike")
    reg.delete("ike")
    assert [r.id for r in reg.list()] == [DEFAULT_ID]
    # file moved to .trash, not gone
    trash = list((tmp_path / "portfolios" / ".trash").glob("ike-*.yaml"))
    assert len(trash) == 1
    # default cannot be deleted
    with pytest.raises(PortfolioRegistryError):
        reg.delete(DEFAULT_ID)


def test_trash_files_are_not_listed(tmp_path) -> None:
    reg = _make(tmp_path)
    reg.create("ike")
    reg.delete("ike")
    # .trash dir + dotfiles must never appear as portfolios
    assert [r.id for r in reg.list()] == [DEFAULT_ID]
