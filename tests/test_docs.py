"""The documentation site, and the seams it is built across.

`mkdocs build --strict` in CI catches broken links and missing annotations, but
it runs on a full install of the docs toolchain. These are the checks that need
no toolchain and that catch the failure modes strictness cannot see: a module
that exists but appears in no API page, a nav entry pointing at nothing, and a
link-rewrite rule that has quietly stopped matching anything.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
MKDOCS = REPO / "mkdocs.yml"

# Modules with nothing a reader needs: pure re-export shims and the entry point.
UNDOCUMENTED_BY_DESIGN = {
    "nullres.__init__", "nullres.__main__", "nullres.cli",
    "nullres.backtest.__init__", "nullres.data.__init__",
    "nullres.features.__init__", "nullres.labels.__init__",
    "nullres.models.__init__", "nullres.strategies.__init__",
    "nullres.validation.__init__",
}


def _hooks():
    spec = importlib.util.spec_from_file_location("nullres_docs_hooks",
                                                  DOCS / "hooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _api_pages_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8")
                     for p in sorted((DOCS / "api").glob("*.md")))


def _module_names() -> list[str]:
    out = []
    for path in sorted((REPO / "nullres").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO).with_suffix("")
        out.append(".".join(rel.parts))
    return out


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_every_module_appears_in_the_api_reference():
    """A module absent from the site is a module nobody will find.

    The reference is assembled by hand — `::: nullres.crosssec` and so on — so
    adding a module does not add a page for it. This is what says so.
    """
    pages = _api_pages_text()
    missing = [
        name for name in _module_names()
        if name not in UNDOCUMENTED_BY_DESIGN
        and f"::: {name}" not in pages
        # A package's contents may be documented through its submodules.
        and not name.endswith(".__init__")
    ]
    assert not missing, (
        f"modules with no page in docs/api/: {missing}. Add `::: <module>` to "
        f"the right page, or list it in UNDOCUMENTED_BY_DESIGN with a reason."
    )


def test_documented_modules_all_exist():
    """The reverse: a `:::` directive naming a module that has been moved."""
    referenced = set(re.findall(r"^::: (nullres[\w.]*)", _api_pages_text(),
                                flags=re.M))
    known = set(_module_names())
    dangling = sorted(m for m in referenced if m not in known
                      and f"{m}.__init__" not in known)
    assert not dangling, f"api pages reference modules that do not exist: {dangling}"


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def test_every_nav_entry_points_at_a_real_file():
    """mkdocs --strict catches this too, but only with the toolchain installed."""
    text = MKDOCS.read_text(encoding="utf-8")
    nav = text[text.index("\nnav:"):]
    targets = re.findall(r":\s*([\w/.-]+\.md)\s*$", nav, flags=re.M)
    assert targets, "no nav entries found — did mkdocs.yml change shape?"
    missing = [t for t in targets if not (DOCS / t).exists()]
    assert not missing, f"nav points at missing files: {missing}"


def test_every_docs_page_is_in_the_nav():
    """An orphan page builds and is then reachable only by guessing its URL."""
    nav = MKDOCS.read_text(encoding="utf-8")
    orphans = [
        p.relative_to(DOCS).as_posix()
        for p in sorted(DOCS.rglob("*.md"))
        if p.relative_to(DOCS).as_posix() not in nav
    ]
    assert not orphans, f"pages absent from the mkdocs nav: {orphans}"


# ---------------------------------------------------------------------------
# The root-document seam
# ---------------------------------------------------------------------------

def test_root_documents_are_read_not_copied():
    """Two homes for a measured figure is the drift this project keeps fixing."""
    hooks = _hooks()
    for page, source in hooks.SOURCES.items():
        assert (REPO / source).exists(), f"{source} is gone but {page} feeds off it"
        stub = (DOCS / page).read_text(encoding="utf-8").strip()
        assert len(stub.splitlines()) <= 2 and "hooks.py" in stub, (
            f"docs/{page} has grown real content; it is generated from "
            f"{source} at build time and anything written here is discarded"
        )


def test_the_link_rewrites_still_have_something_to_rewrite():
    """A rule that matches nothing is either dead or about to silently break.

    `README.md` links to `docs/<page>.md`, which is correct from the repo root
    and wrong once the file is rendered from inside `docs/`. If those links
    ever stop existing the hook is dead code; if they change shape, the site
    fills up with 404s that only `--strict` would catch.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    research = (REPO / "RESEARCH.md").read_text(encoding="utf-8")
    assert "](docs/" in readme, "README no longer links into docs/"
    assert "](RESEARCH.md" in readme, "README no longer links to RESEARCH.md"
    assert "](docs/" in research, "RESEARCH no longer links into docs/"
    assert any("](../RESEARCH.md" in p.read_text(encoding="utf-8")
               for p in DOCS.glob("*.md")), "no docs page links back to RESEARCH"


@pytest.mark.parametrize("before,after", [
    ("see [rules](docs/00-the-rules.md) now", "see [rules](00-the-rules.md) now"),
    ("[leak](docs/02-leakage.md#7-hindsight)", "[leak](02-leakage.md#7-hindsight)"),
    ("full numbers in [R](RESEARCH.md)", "full numbers in [R](research.md)"),
    ("nothing to do here", "nothing to do here"),
])
def test_root_link_rewriting(before, after):
    assert _hooks()._rewrite_from_root(before) == after


def test_docs_pages_link_back_to_the_generated_research_page():
    hooks = _hooks()

    class _File:
        src_uri = "05-graveyard.md"

    class _Page:
        file = _File()

    out = hooks.on_page_markdown("see [the numbers](../RESEARCH.md#x)",
                                 _Page(), None, None)
    assert out == "see [the numbers](research.md#x)"
