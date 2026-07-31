"""Step selection in `scripts/reproduce_all.py`.

This script is how the documents get reconciled against the code, and its two
filters decide what runs. Both match on substring, which is what makes
`--only xsec` useful and what made `--skip xsec-wide` quietly remove two steps
once `xsec-wide-37` existed. A skip leaves nothing behind to notice, so an
over-broad or no-op filter is invisible unless something says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from reproduce_all import STEPS, _matches, main  # noqa: E402


def _slugs():
    return [s[0] for s in STEPS]


def test_slugs_are_unique():
    assert len(_slugs()) == len(set(_slugs()))


def test_every_step_has_a_positive_time_estimate():
    """The printed budget is a planning number; zero would make it a lie."""
    bad = [(s[0], s[3]) for s in STEPS if not isinstance(s[3], int) or s[3] < 1]
    assert not bad, f"steps with a missing or non-positive estimate: {bad}"


# ---------------------------------------------------------------------------
# Exact beats substring
# ---------------------------------------------------------------------------

def test_an_exact_slug_selects_only_that_step():
    """`xsec-wide` must not drag in `xsec-wide-37`.

    That pair is the whole reason this rule exists: one is a 26-minute run
    needing open-interest archives, the other is the 25-minute run behind the
    project's "honest version" figure, and the syntax could not tell them apart.
    """
    assert _matches("xsec-wide", "xsec-wide")
    assert not _matches("xsec-wide", "xsec-wide-37")


def test_a_partial_string_still_matches_broadly():
    hits = [s for s in _slugs() if _matches("xsec", s)]
    assert hits == ["xsec-narrow", "xsec-narrow-37", "xsec-wide", "xsec-wide-37"]


def test_prefix_pairs_are_all_addressable_individually():
    """Any slug that is a prefix of another must still be selectable alone."""
    slugs = _slugs()
    prefixes = [a for a in slugs if any(b != a and b.startswith(a) for b in slugs)]
    assert prefixes, "no prefix pairs left — this test is guarding nothing"
    for slug in prefixes:
        selected = [s for s in slugs if _matches(slug, s)]
        assert selected == [slug], f"{slug!r} also selected {selected}"


# ---------------------------------------------------------------------------
# Filters that select nothing must fail, not proceed
# ---------------------------------------------------------------------------

def test_only_with_no_match_fails(capsys):
    assert main(["--only", "does-not-exist", "--list"]) == 1
    assert "no step matches" in capsys.readouterr().out


def test_skip_with_no_match_fails_rather_than_running_everything(capsys):
    """The dangerous direction.

    A typo in `--skip` used to mean the expensive step you were trying to avoid
    ran anyway, silently, with exit 0.
    """
    assert main(["--skip", "xsec-wid3", "--list"]) == 1
    out = capsys.readouterr().out
    assert "matches no step" in out and "nothing was skipped" in out


def test_skip_reports_what_it_removed(capsys):
    assert main(["--skip", "xsec", "--list"]) == 0
    out = capsys.readouterr().out
    assert "skipping 4 step(s)" in out
    for slug in ("xsec-narrow", "xsec-narrow-37", "xsec-wide", "xsec-wide-37"):
        assert slug in out


def test_skip_that_removes_everything_fails(capsys):
    """Reachable only by combining the filters — no substring hits every slug."""
    assert main(["--only", "xsec", "--skip", "xsec", "--list"]) == 1
    assert "removed every step" in capsys.readouterr().out


@pytest.mark.parametrize("argv,expected", [
    (["--only", "xsec-wide"], 1),
    (["--only", "xsec"], 4),
    (["--only", "robust"], 6),
    (["--skip", "xsec-wide"], len(STEPS) - 1),
    (["--skip", "xsec"], len(STEPS) - 4),
])
def test_filter_arithmetic(argv, expected, capsys):
    assert main([*argv, "--list"]) == 0
    out = capsys.readouterr().out
    assert f"{expected} step(s)" in out, out.splitlines()[:2]


# ---------------------------------------------------------------------------
# The claims the documents rest on must have a step behind them
# ---------------------------------------------------------------------------

def test_every_documented_command_has_a_step():
    """RESEARCH.md and the graveyard quote figures from specific invocations.

    Each of these was, at some point, a number in a document that no command
    regenerated — `features-deriv` for the `ls_accounts` claim, `xsec-wide-37`
    for the "honest version" Sharpe of 1.80. Losing a step here means a claim
    silently goes back to being unverifiable.
    """
    required = {
        "features-deriv",   # RESEARCH §2: ls_accounts is the top feature
        "xsec-wide-37",     # RESEARCH §3.2 / graveyard: the honest version
        "xsec-narrow-37",   # the width control, matched at 37 features
        "ablate-deriv",     # RESEARCH §2: the matched-sample ablation
        "null-run",         # the calibration control
    }
    missing = required - set(_slugs())
    assert not missing, f"documented claims lost their reproducing step: {missing}"
