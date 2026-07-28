"""The evidence layer beneath the graveyard.

Two things are being kept, and keeping them apart is the whole design:

    runs/*.json          machine-written, append-only, never hand-edited.
                         What was measured, under exactly which config, at
                         which commit. Boring by design.

    docs/05-graveyard.md hand-written. WHY it died and what it means. That
                         judgement is the actual work and stays human.

Neither half works alone. Prose without evidence rots — six months on, nobody
can reproduce "mean AUC 0.5443". Evidence without prose is a spreadsheet that
teaches nothing: no log entry will ever contain "one bear market wearing a
trend-following costume". Graveyard entries cite run ids; run records point
back at the graveyard.

The one capability that genuinely needs the machine layer is recognising that
you are about to re-run something you already killed. Markdown cannot do that.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNS_DIR = "runs"

# Changing these does not make an experiment a different experiment, so they are
# excluded from the fingerprint and from distance comparisons.
COSMETIC = {"name", "out_dir", "cache_dir", "prior_trials"}


@dataclass
class RunRecord:
    """One execution of one command against one config."""

    id: str
    timestamp: str
    command: str
    config_name: str
    config_hash: str
    git_sha: str
    git_dirty: bool
    config: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None          # KILLED | SURVIVED | None
    notes: str = ""
    # How many distinct parameter combinations this run evaluated. A `run` over
    # five strategies is five; a 25-cell sweep is twenty-five. This is the
    # multiple-testing exposure the run added, and aggregating it across the
    # ledger is the only honest input to `deflated_sharpe`.
    #
    # None means the record predates the field. It is NOT the same as 1, and
    # defaulting it to 1 silently counted a 23-cell robustness battery as a
    # single look — an undercount, in the flattering direction, invisible
    # because the dataclass default filled the hole. `unrecorded_variants`
    # counts these so the gap can be reported rather than absorbed.
    variants: int | None = None

    @property
    def short_id(self) -> str:
        return self.id[:8]


def flatten_config(cfg: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dataclasses to {"data.symbol": "BTCUSDT", ...}."""
    out: dict[str, Any] = {}
    if is_dataclass(cfg):
        items = [(f.name, getattr(cfg, f.name)) for f in fields(cfg)]
    elif isinstance(cfg, dict):
        items = list(cfg.items())
    else:
        return {prefix.rstrip("."): cfg}

    for key, value in items:
        path = f"{prefix}{key}"
        if is_dataclass(value) or isinstance(value, dict):
            out.update(flatten_config(value, f"{path}."))
        elif isinstance(value, list):
            out[path] = ",".join(map(str, value))
        else:
            out[path] = value
    return out


def _significant(flat: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in flat.items()
            if k.rsplit(".", 1)[-1] not in COSMETIC}


def config_hash(cfg: Any) -> str:
    """Stable fingerprint of everything that makes this a distinct experiment."""
    payload = json.dumps(_significant(flatten_config(cfg)), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def config_distance(a: Any, b: Any) -> tuple[int, list[str]]:
    """How many meaningful parameters differ, and which.

    Used to answer "is this config a near-miss of something already killed".
    A distance of 0 means you are re-running an identical experiment; 1-3 means
    you are tuning around one that may already be dead.
    """
    fa, fb = _significant(flatten_config(a)), _significant(flatten_config(b))
    keys = set(fa) | set(fb)
    differing = sorted(k for k in keys if str(fa.get(k)) != str(fb.get(k)))
    return len(differing), differing


def _status_path(line: str) -> str:
    """The path out of one `git status --porcelain` line."""
    path = line[3:].strip()
    if " -> " in path:                      # a rename; judge the destination
        path = path.split(" -> ", 1)[1].strip()
    return path.strip('"')


def _git_state(repo: Path, runs_dir: str = RUNS_DIR) -> tuple[str, bool]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    sha = run("rev-parse", "--short", "HEAD") or "unknown"

    # `git_dirty` answers one question: was the CODE that produced this result
    # uncommitted, so that the recorded SHA does not reproduce it? The ledger's
    # own output must not count toward that. Writing a record dirties the tree,
    # so every run after the first in a session reported dirty against pristine
    # code — the flag was true almost always, which is the same as saying
    # nothing. Changes under `runs/` are therefore ignored.
    prefix = runs_dir.rstrip("/") + "/"
    dirty = any(
        not _status_path(line).startswith(prefix)
        for line in run("status", "--porcelain").splitlines()
        if line.strip()
    )
    return sha, dirty


def count_trials(runs: list[RunRecord], prior: int = 0) -> int:
    """Distinct parameter combinations explored, across every recorded run.

    This is deliberately GLOBAL rather than scoped to one config. The question
    multiple-testing correction asks is "how many things did you look at before
    reporting this one", and a researcher who would have published whichever of
    six configs worked has tried all six — not one.

    It counts DISTINCT experiments, not executions. Summing every record made
    the correction a function of how often commands were run: repeating one
    `xsec` five times took the count 230 -> 270 and quietly lowered every
    reported deflated Sharpe without a single new hypothesis being tested, which
    also means a number quoted in the docs could not be reproduced later. Each
    (config fingerprint, command) pair therefore contributes once, at the
    largest variant count seen for it — a re-run is the same look, and a wider
    sweep of the same config is a bigger one.

    `prior` declares trials that predate the ledger. Undercounting is the
    failure this whole function exists to fix, so an honest estimate of past
    work belongs here rather than a zero.
    """
    seen: dict[tuple[str, str], int] = {}
    for record in runs:
        key = (record.config_hash, record.command)
        declared = 1 if record.variants is None else max(record.variants, 1)
        seen[key] = max(seen.get(key, 0), declared)
    return prior + sum(seen.values())


def unrecorded_variants(runs: list[RunRecord]) -> int:
    """Records written before `variants` existed, each counted as a single look.

    Reported rather than repaired: the ledger is append-only and back-filling a
    guess would be worse than naming the gap. A non-zero count means the true
    multiple-testing exposure is HIGHER than `count_trials` returns.
    """
    return sum(1 for r in runs if r.variants is None)


def record_run(cfg: Any, command: str, metrics: dict[str, Any] | None = None,
               verdict: str | None = None, notes: str = "", variants: int = 1,
               runs_dir: str = RUNS_DIR, repo: Path | None = None) -> RunRecord:
    """Append one record. Never overwrites: the log is a ledger, not a cache."""
    repo = repo or Path.cwd()
    now = datetime.now(timezone.utc)
    sha, dirty = _git_state(repo)
    chash = config_hash(cfg)

    # Deterministic id from (config, command, time) so two runs never collide.
    seed = f"{chash}|{command}|{now.isoformat()}"
    run_id = hashlib.sha256(seed.encode()).hexdigest()[:12]

    record = RunRecord(
        id=run_id,
        timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        command=command,
        config_name=getattr(cfg, "name", "unnamed"),
        config_hash=chash,
        git_sha=sha,
        git_dirty=dirty,
        config=flatten_config(cfg),
        metrics=metrics or {},
        verdict=verdict,
        notes=notes,
        variants=max(int(variants), 1),      # always written; never left unset
    )

    out = Path(runs_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{now:%Y%m%d-%H%M%S}-{run_id[:8]}.json"
    path.write_text(json.dumps(asdict(record), indent=2, default=str))
    return record


def load_runs(runs_dir: str = RUNS_DIR) -> list[RunRecord]:
    """Every record on disk, newest last. Corrupt files are skipped, not fatal."""
    out = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        try:
            raw = json.loads(path.read_text())
            out.append(RunRecord(**raw))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


def find_similar(cfg: Any, runs: list[RunRecord], max_distance: int = 3,
                 verdict: str | None = "KILLED") -> list[tuple[int, list[str], RunRecord]]:
    """Past runs whose config is within `max_distance` parameters of this one.

    This is the reason the machine layer exists. Nobody re-reads a 294-line
    markdown file before every experiment, so eighteen months from now the dead
    end gets re-run. A config comparison does not forget.
    """
    hits = []
    for record in runs:
        if verdict and record.verdict != verdict:
            continue
        distance, differing = config_distance(cfg, record.config)
        if distance <= max_distance:
            hits.append((distance, differing, record))
    return sorted(hits, key=lambda h: h[0])


def format_warning(hits: list[tuple[int, list[str], RunRecord]]) -> str:
    """Render the near-miss warning. Empty string when there is nothing to say."""
    if not hits:
        return ""
    lines = [
        f"  WARNING: this config is within {hits[0][0]} parameter(s) of "
        f"{len(hits)} run(s) already marked KILLED."
    ]
    for distance, differing, record in hits[:3]:
        changed = ", ".join(differing[:4]) or "nothing — this is an exact re-run"
        lines.append(f"    {record.timestamp[:10]}  {record.config_name:<16} "
                     f"{record.command:<8} [{record.short_id}]  differs by: {changed}")
    lines.append("    See docs/05-graveyard.md before spending time on this.")
    return "\n".join(lines)
