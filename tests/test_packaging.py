"""Every module on disk must be importable — and must actually be committed.

Written after `.gitignore` contained a bare `data/`, which git matches at any
depth. That silently excluded `nullres/data/` from every commit. Locally
everything passed; CI failed with four `ModuleNotFoundError: No module named
'nullres.data'` errors, which is a confusing way to learn about a .gitignore bug.

These tests turn that into one obvious failure with the cause in the message.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "nullres"


def _module_paths() -> list[Path]:
    return sorted(
        p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts
    )


def _dotted(path: Path) -> str:
    rel = path.relative_to(REPO).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def test_every_module_imports():
    """Catches a subpackage missing its __init__.py, or a broken import chain."""
    failures = []
    for path in _module_paths():
        name = _dotted(path)
        if name.endswith("__main__"):
            continue          # executes on import by design
        try:
            importlib.import_module(name)
        except Exception as exc:                      # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n  " + "\n  ".join(failures)


def test_every_subpackage_has_an_init():
    missing = [
        d.relative_to(REPO).as_posix()
        for d in PACKAGE.rglob("*")
        if d.is_dir() and "__pycache__" not in d.parts
        and not (d / "__init__.py").exists()
    ]
    assert not missing, f"directories without __init__.py: {missing}"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="not a git checkout")
def test_no_source_file_is_gitignored():
    """The actual bug: source files silently excluded by an unanchored pattern.

    `git check-ignore` is the authority here — reimplementing gitignore matching
    would just reproduce the misunderstanding that caused the problem.
    """
    sources = [
        p.relative_to(REPO).as_posix()
        for p in [*_module_paths(), *sorted((REPO / "tests").rglob("*.py")),
                  *sorted((REPO / "configs").glob("*.toml"))]
    ]
    ignored = _git("check-ignore", *sources).strip()
    assert not ignored, (
        "these source files are excluded by .gitignore and will never be "
        "committed:\n  " + ignored.replace("\n", "\n  ") +
        "\n\nA pattern without a leading slash matches at every depth — "
        "anchor it (`/data/` not `data/`)."
    )


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="not a git checkout")
def test_every_source_file_is_tracked_or_staged():
    """A file that exists locally but is untracked will break CI, not your laptop."""
    tracked = set(_git("ls-files").split())
    known = tracked | set(_git("diff", "--cached", "--name-only").split())
    untracked = [
        p.relative_to(REPO).as_posix()
        for p in _module_paths()
        if p.relative_to(REPO).as_posix() not in known
    ]
    assert not untracked, (
        f"source files not tracked by git: {untracked}. "
        f"They exist on this machine and nowhere else."
    )
