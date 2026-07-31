"""Regenerate every number this repository publishes, in one unattended run.

RESEARCH.md, docs/05-graveyard.md and README.md quote figures from sixteen
different commands. Reproducing them by hand means remembering all sixteen and
their flags, which is exactly the kind of remembering the run ledger exists to
replace. This is the ledger's equivalent for the documents.

    python scripts/reproduce_all.py                 # everything, resumable
    python scripts/reproduce_all.py --list          # what it would run
    python scripts/reproduce_all.py --only xsec     # just the panel work
    python scripts/reproduce_all.py --force         # redo completed steps

Each step's full output is written to `reports/reproduce/<slug>.txt`, so the
documents can be reconciled against captured text rather than against a
terminal scrollback. Steps whose output already exists are skipped, which makes
the whole thing resumable after an interruption or a crash — the wide panel
alone takes half an hour and has been OOM-killed before.

**A failing step does not stop the run.** One config with missing data should
not cost you the other fifteen results; failures are collected and reported at
the end, and the exit code is non-zero if any occurred.

Two steps deserve comment.

  THE SWEEPS produce no figure anyone quotes. They are here because the trial
  count that deflates every Sharpe has to contain them: two 25-cell sweeps are
  fifty looks at the data whether or not a number from them is ever published.
  They are what let `prior_trials` be retired to 0 rather than estimated, and
  skipping them would put the ledger back below the exposure it describes.

  THE WIDE PANEL needs open-interest metrics for ~123 symbols, which is a
  multi-hour download — see `scripts/precache_metrics.py`. Without them it will
  refuse to start rather than silently fall back to a narrower feature set.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "reproduce"

# (slug, why it exists, argv after `-m nullres`, minutes)
#
# The last field is MEASURED, not guessed, and rounded up to the minute. It was
# guessed once and never revisited: the batteries were estimated at 6-8 minutes
# against an actual 8-14 SECONDS, because those numbers predate the context
# reuse that stopped `robust` calling `prepare()` per grid cell. The total came
# out 99 minutes against a real 19.
#
# That would be harmless if it erred one way, but it did not. The two xsec steps
# were the only ones UNDER-estimated — 4 minutes against 8, 3 against 5 — so the
# headline was three times too pessimistic while the step you actually sit and
# wait for was twice too optimistic. Re-measure after any change that moves them.
STEPS: list[tuple[str, str, list[str], int]] = [
    # --- the harness is not lying to us -------------------------------------
    ("audit-1h", "leak checks, 1h", ["audit", "-c", "configs/btc_1h.toml"], 1),
    ("audit-4h", "leak checks, 4h", ["audit", "-c", "configs/btc_4h.toml"], 1),
    ("audit-1d", "leak checks, 1d", ["audit", "-c", "configs/btc_1d.toml"], 1),
    ("audit-deriv", "leak checks, derivatives",
     ["audit", "-c", "configs/btc_4h_deriv.toml"], 1),
    ("null-run", "the calibration control: no edge in a random walk",
     ["run", "-c", "configs/null.toml"], 1),
    ("null-audit", "the same control through the CLI",
     ["audit", "-c", "configs/null.toml"], 2),

    # --- RESEARCH section 1: single-asset tables ----------------------------
    ("run-1h", "RESEARCH §1, 1h table", ["run", "-c", "configs/btc_1h.toml"], 1),
    ("run-4h", "RESEARCH §1, 4h table", ["run", "-c", "configs/btc_4h.toml"], 1),
    ("run-1d", "RESEARCH §1, 1d table", ["run", "-c", "configs/btc_1d.toml"], 1),
    ("run-deriv", "derivatives config, all strategies",
     ["run", "-c", "configs/btc_4h_deriv.toml"], 1),
    ("budget-1h", "the cost budget, modelled vs measured",
     ["budget", "-c", "configs/btc_1h.toml"], 1),
    ("budget-4h", "cost budget at 4h", ["budget", "-c", "configs/btc_4h.toml"], 1),
    ("budget-1d", "cost budget at 1d", ["budget", "-c", "configs/btc_1d.toml"], 1),
    ("features-4h", "permutation importance, 4h",
     ["features", "-c", "configs/btc_4h.toml"], 1),

    # --- multiple-testing exposure the ledger must contain ------------------
    ("sweep-1h", "25-cell threshold sweep (for the ledger, not for a figure)",
     ["sweep", "-c", "configs/btc_1h.toml", "--strategy", "ml_meta"], 1),
    ("sweep-4h", "25-cell threshold sweep at 4h",
     ["sweep", "-c", "configs/btc_4h.toml", "--strategy", "ml_meta"], 1),

    # --- RESEARCH section 2: derivatives -----------------------------------
    ("ablate-deriv", "RESEARCH §2, matched-sample ablation",
     ["ablate", "-c", "configs/btc_4h_deriv.toml"], 1),

    # --- the falsification batteries ---------------------------------------
    ("robust-donchian", "graveyard: donchian 4h",
     ["robust", "-c", "configs/btc_4h.toml", "--strategy", "donchian"], 1),
    ("robust-voltarget", "graveyard: volatility targeting",
     ["robust", "-c", "configs/btc_4h.toml", "--strategy", "vol_target"], 1),
    ("robust-smacross", "sma_cross battery",
     ["robust", "-c", "configs/btc_4h.toml", "--strategy", "sma_cross"], 1),
    ("robust-meanrev", "mean_reversion battery",
     ["robust", "-c", "configs/btc_4h.toml", "--strategy", "mean_reversion"], 1),
    ("robust-mldirection", "RESEARCH §2 battery, ml_direction",
     ["robust", "-c", "configs/btc_4h_deriv.toml", "--strategy", "ml_direction",
      "--transfer-start", "2021-12"], 2),
    ("robust-mlmeta", "RESEARCH §2 battery, ml_meta",
     ["robust", "-c", "configs/btc_4h_deriv.toml", "--strategy", "ml_meta",
      "--transfer-start", "2021-12"], 2),

    # --- RESEARCH section 3: cross-sectional -------------------------------
    ("xsec-narrow", "RESEARCH §3.1, 11 symbols, 46 features",
     ["xsec", "-c", "configs/xsec_4h.toml", "--verify"], 5),
    ("xsec-narrow-37", "the width control: same panel at 37 features",
     ["xsec", "-c", "configs/xsec_4h.toml", "--set", "data.metrics=false"], 3),
    ("xsec-wide", "RESEARCH §3.2, 136 symbols — needs precached metrics",
     ["xsec", "-c", "configs/xsec_4h.toml", "--universe", "2021-12",
      "--top-n", "40", "--verify"], 26),

    # --- what the ledger now says ------------------------------------------
    ("log", "the ledger after all of the above", ["log", "--limit", "60"], 1),
]


def run_step(slug: str, argv: list[str], force: bool) -> tuple[str, float]:
    """Execute one step. Returns (status, seconds)."""
    target = OUT / f"{slug}.txt"
    if target.exists() and not force:
        return "skipped", 0.0

    started = time.time()
    process = subprocess.run(
        [sys.executable, "-u", "-m", "nullres", *argv],
        cwd=REPO, capture_output=True, text=True,
    )
    elapsed = time.time() - started

    target.write_text(
        f"$ python -m nullres {' '.join(argv)}\n"
        f"# exit {process.returncode} after {elapsed:.0f}s\n\n"
        f"{process.stdout}\n{process.stderr}",
        encoding="utf-8",
    )
    return ("ok" if process.returncode == 0 else "FAILED"), elapsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None, metavar="TEXT",
                        help="run steps whose slug contains TEXT")
    parser.add_argument("--skip", default=None, metavar="TEXT",
                        help="skip steps whose slug contains TEXT — for running "
                             "the wide panel on the machine that holds its "
                             "metrics and everything else where there is RAM")
    parser.add_argument("--force", action="store_true",
                        help="redo steps that already have captured output")
    parser.add_argument("--list", action="store_true",
                        help="show the steps and exit")
    args = parser.parse_args(argv)

    steps = [s for s in STEPS if not args.only or args.only in s[0]]
    if args.skip:
        steps = [s for s in steps if args.skip not in s[0]]
    if not steps:
        print(f"no step matches {args.only!r}")
        return 1

    if args.list:
        print(f"{len(steps)} step(s), ~{sum(s[3] for s in steps)} min total\n")
        for slug, why, cmd, minutes in steps:
            print(f"  {slug:<20}~{minutes:>3} min  {why}")
            print(f"  {'':<20}          nullres {' '.join(cmd)}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    budget = sum(s[3] for s in steps)
    print(f"{len(steps)} steps, ~{budget} min estimated. Output -> "
          f"{OUT.relative_to(REPO)}/\n")

    results, started = [], time.time()
    for i, (slug, why, cmd, _) in enumerate(steps, start=1):
        print(f"[{i}/{len(steps)}] {slug:<20}{why}", end="", flush=True)
        status, elapsed = run_step(slug, cmd, args.force)
        note = "already done" if status == "skipped" else f"{elapsed:.0f}s"
        print(f"  -> {status} ({note})", flush=True)
        results.append((slug, status, elapsed))

    failed = [slug for slug, status, _ in results if status == "FAILED"]
    ran = [r for r in results if r[1] != "skipped"]
    print(f"\n{'=' * 70}")
    print(f"{len(ran)} run, {len(results) - len(ran)} skipped, "
          f"{len(failed)} failed, in {(time.time() - started) / 60:.0f} min")

    if failed:
        print("\nfailed steps (see their capture for the traceback):")
        for slug in failed:
            print(f"  {slug:<20}{(OUT / f'{slug}.txt').relative_to(REPO)}")
        print("\nRe-run just one with: --only <slug> --force")
        return 1

    print("\nEvery step completed. Reconcile the documents against "
          f"{OUT.relative_to(REPO)}/,")
    print("then check the trial count in `nullres log` against the figure")
    print("RESEARCH.md deflates against.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
