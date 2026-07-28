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


def test_no_unaudited_fit_sites():
    """Pin the set of places that can train a model.

    Every `.fit()` is somewhere a model could be handed data from the future,
    so the docs have always claimed the call sites are few and audited. The
    claim was wrong: it named one, while `crosssec.fit_predict_panel` had its
    own — in the path that produced the strongest result in the project. None
    of them leak (each purges independently), but nothing would have flagged a
    fourth appearing.

    Add a site only after satisfying yourself it is fed purged, in-fold data,
    then list it here.
    """
    import re

    allowed = {
        ("nullres/models/classifier.py", "fit_predict_walk_forward"),
        ("nullres/models/classifier.py", "feature_importance"),
        ("nullres/crosssec.py", "fit_predict_panel"),
    }

    found = set()
    for path in _module_paths():
        rel = path.relative_to(REPO).as_posix()
        function = "<module>"
        for line in path.read_text(encoding="utf-8").splitlines():
            if match := re.match(r"\s*def\s+(\w+)", line):
                function = match.group(1)
            # `.fit(` on an estimator, not `.fit_predict_*` helper definitions
            # and not the `make_pipeline`/docstring mentions.
            if re.search(r"\w\.fit\(", line):
                found.add((rel, function))

    assert found == allowed, (
        f"the set of .fit() call sites changed.\n"
        f"  new/unaudited: {sorted(found - allowed)}\n"
        f"  gone:          {sorted(allowed - found)}\n"
        f"Each site is a place a model can be trained on data it should not "
        f"see. Confirm the new one receives purged, in-fold data only, then "
        f"add it to `allowed` here and to the docstring in "
        f"nullres/models/classifier.py."
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
