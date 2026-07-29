"""The evidence ledger.

Its only job is to remember accurately, so the tests are about identity: does
a meaningfully different config get a different fingerprint, and does a
cosmetically different one get the same fingerprint?
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nullres.config import load_config
from nullres.runlog import (
    config_distance,
    config_hash,
    find_similar,
    flatten_config,
    format_warning,
    load_runs,
    record_run,
)


@pytest.fixture
def cfg():
    return load_config("configs/btc_4h.toml")


def test_flatten_walks_nested_dataclasses(cfg):
    flat = flatten_config(cfg)
    assert flat["data.symbol"] == "BTCUSDT"
    assert flat["sizing.min_hold"] == cfg.sizing.min_hold
    assert flat["cost.fee_bps"] == cfg.cost.fee_bps
    # Lists are collapsed, not dropped — the strategy set is part of the identity.
    assert "sma_cross" in flat["strategies"]


def test_cosmetic_changes_do_not_change_the_fingerprint(cfg):
    """Renaming an experiment does not make it a different experiment."""
    other = copy.deepcopy(cfg)
    other.name = "a-different-name"
    other.out_dir = "somewhere/else"
    other.data.cache_dir = "/tmp/cache"
    assert config_hash(other) == config_hash(cfg)


def test_meaningful_changes_do_change_the_fingerprint(cfg):
    for path, value in [("sizing.min_hold", 999), ("cost.fee_bps", 42.0),
                        ("label.horizon", 7), ("data.symbol", "ETHUSDT")]:
        other = copy.deepcopy(cfg)
        section, key = path.split(".")
        setattr(getattr(other, section), key, value)
        assert config_hash(other) != config_hash(cfg), f"{path} was ignored"


def test_distance_counts_and_names_the_differences(cfg):
    other = copy.deepcopy(cfg)
    other.sizing.min_hold = 999
    other.cost.fee_bps = 42.0

    distance, differing = config_distance(cfg, other)
    assert distance == 2
    assert differing == ["cost.fee_bps", "sizing.min_hold"]

    assert config_distance(cfg, copy.deepcopy(cfg))[0] == 0


def test_records_accumulate_rather_than_overwrite(cfg, tmp_path):
    """A ledger, not a cache. Two runs of the same config keep both records."""
    for _ in range(3):
        record_run(cfg, "run", runs_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*.json"))) == 3

    runs = load_runs(str(tmp_path))
    assert len(runs) == 3
    assert len({r.id for r in runs}) == 3, "run ids collided"
    assert len({r.config_hash for r in runs}) == 1, "same config, same fingerprint"


def test_record_captures_provenance(cfg, tmp_path):
    record = record_run(cfg, "robust", metrics={"sharpe": 0.5},
                        verdict="KILLED", runs_dir=str(tmp_path))
    assert record.verdict == "KILLED"
    assert record.git_sha and record.git_sha != ""
    assert record.timestamp.endswith("Z")
    assert record.config["data.symbol"] == "BTCUSDT"

    on_disk = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert on_disk["metrics"]["sharpe"] == 0.5


def test_trial_count_sums_variants_not_runs(cfg, tmp_path):
    """A 25-cell sweep is 25 trials, not one.

    Counting runs instead of variants is how `n_trials=6` got reported for a
    project that had explored over two hundred parameter combinations.
    """
    from nullres.runlog import count_trials

    record_run(cfg, "run", variants=6, runs_dir=str(tmp_path))
    record_run(cfg, "sweep", variants=25, runs_dir=str(tmp_path))
    record_run(cfg, "robust", variants=24, runs_dir=str(tmp_path))

    runs = load_runs(str(tmp_path))
    assert len(runs) == 3
    assert count_trials(runs) == 55
    assert count_trials(runs, prior=200) == 255


def test_trial_count_never_undercounts_to_zero(cfg, tmp_path):
    """A record with no declared variants still counts as one look."""
    from nullres.runlog import count_trials

    record_run(cfg, "run", variants=0, runs_dir=str(tmp_path))
    assert count_trials(load_runs(str(tmp_path))) == 1


def test_rerunning_the_same_experiment_is_not_a_new_trial(cfg, tmp_path):
    """Otherwise the correction measures keystrokes instead of hypotheses.

    Summing every record made `deflated_sharpe` a function of how often commands
    were run: repeating one `xsec` five times took the count 230 -> 270 and
    lowered every reported result without a single new hypothesis being tested.
    It also means a deflated Sharpe quoted in the docs cannot be reproduced —
    the number drifts down every time anyone runs anything.
    """
    from nullres.runlog import count_trials

    for _ in range(5):
        record_run(cfg, "run", variants=7, runs_dir=str(tmp_path))

    runs = load_runs(str(tmp_path))
    assert len(runs) == 5, "the ledger still keeps every execution"
    assert count_trials(runs) == 7, "five identical runs are one look, not five"


def test_the_run_about_to_happen_folds_into_the_dedupe(cfg, tmp_path):
    """Otherwise the fix for re-runs leaks back in through the `extra` term.

    `trials_so_far` adds the pending run's variants so the correction accounts
    for work not yet recorded. Added on top, that meant re-running an already
    recorded experiment nudged its own count up by its own size every time — a
    published deflated Sharpe drifting a little with each verification, which is
    the exact bug the dedupe exists to prevent. Folded into the same dedupe, a
    re-run contributes nothing and only a wider sweep raises the count.
    """
    from nullres.runlog import config_hash, count_trials

    record_run(cfg, "xsec", variants=6, runs_dir=str(tmp_path))
    runs = load_runs(str(tmp_path))
    fingerprint = config_hash(cfg)

    assert count_trials(runs) == 6
    assert count_trials(runs, pending=(fingerprint, "xsec", 6)) == 6, \
        "re-running the same experiment is not new exposure"
    assert count_trials(runs, pending=(fingerprint, "xsec", 20)) == 20, \
        "a WIDER sweep of the same config is more exposure"
    assert count_trials(runs, pending=(fingerprint, "sweep", 25)) == 31, \
        "a different command on the same config is new exposure"


def test_a_different_config_or_command_is_a_new_trial(cfg, tmp_path):
    from nullres.runlog import count_trials

    record_run(cfg, "run", variants=7, runs_dir=str(tmp_path))
    record_run(cfg, "sweep", variants=25, runs_dir=str(tmp_path))   # same config
    tweaked = copy.deepcopy(cfg)
    tweaked.sizing.min_hold = 999
    record_run(tweaked, "run", variants=7, runs_dir=str(tmp_path))  # new config

    assert count_trials(load_runs(str(tmp_path))) == 7 + 25 + 7


def test_widening_a_sweep_raises_the_count_for_that_experiment(cfg, tmp_path):
    """Re-running the same config with MORE cells did explore more."""
    from nullres.runlog import count_trials

    record_run(cfg, "sweep", variants=9, runs_dir=str(tmp_path))
    record_run(cfg, "sweep", variants=25, runs_dir=str(tmp_path))
    assert count_trials(load_runs(str(tmp_path))) == 25


def test_calibration_runs_are_not_multiple_testing_exposure(tmp_path):
    """Verifying the instrument must not penalise the results.

    `configs/null.toml` runs the whole pipeline on a random walk to prove the
    harness finds nothing there. It tests no hypothesis about any market, so
    counting it as a trial would mean every check of the instrument deflated
    every real result a little further — a direct penalty for being careful.
    """
    from nullres.runlog import count_trials

    real = load_config("configs/btc_4h.toml")
    null = load_config("configs/null.toml")
    assert null.data.source == "synthetic"

    record_run(real, "run", variants=7, runs_dir=str(tmp_path))
    before = count_trials(load_runs(str(tmp_path)))

    record_run(null, "run", variants=6, runs_dir=str(tmp_path))
    assert count_trials(load_runs(str(tmp_path))) == before
    assert len(load_runs(str(tmp_path))) == 2, "still recorded, just not counted"


def test_records_predating_variants_are_flagged_not_absorbed(cfg, tmp_path):
    """A missing field must not be read as 'one trial' in silence.

    Two real records predate `variants`. The dataclass default filled the hole,
    so a 23-cell robustness battery counted as a single look — an undercount, in
    the flattering direction, invisible in every output.
    """
    from nullres.runlog import count_trials, unrecorded_variants

    record = record_run(cfg, "robust", runs_dir=str(tmp_path))
    path = next(tmp_path.glob("*.json"))
    raw = json.loads(path.read_text())
    del raw["variants"]                       # the old on-disk schema
    path.write_text(json.dumps(raw))

    runs = load_runs(str(tmp_path))
    assert len(runs) == 1, "an old record must still load"
    assert runs[0].variants is None, "absent is not the same as one"
    assert unrecorded_variants(runs) == 1
    assert count_trials(runs) == 1, "counted as a single look, but reported"
    assert record.variants == 1, "newly written records always declare it"


def test_deflation_strengthens_as_trials_accumulate():
    """The whole point: looking more times must lower the surviving Sharpe.

    Measured on the project's best result (wide cross-sectional k=2, Sharpe
    1.61 over 6,909 bars): 0.88 at the six trials that were previously
    reported, 0.05 at the ~200 actually explored.
    """
    from nullres.backtest.metrics import deflated_sharpe

    values = [deflated_sharpe(1.61, 6909, 2190, n) for n in (1, 6, 25, 200, 1000)]
    assert values == sorted(values, reverse=True), "more trials must deflate more"
    assert values[1] > 0.8, "six trials barely penalises it"
    assert abs(values[3]) < 0.15, "two hundred trials leaves nothing"
    assert values[4] < 0


def test_prior_trials_does_not_change_config_identity(cfg):
    """Declaring past exposure is metadata, not a different experiment."""
    other = copy.deepcopy(cfg)
    other.prior_trials = 500
    assert config_hash(other) == config_hash(cfg)
    assert config_distance(cfg, other)[0] == 0


def test_cli_record_helper_accepts_every_field_commands_pass(cfg, monkeypatch, tmp_path):
    """The CLI helper and `record_run` must not drift apart.

    They did: `variants` was added to `record_run` but not to `_record`, so
    every `run` crashed with a TypeError *after* printing its results. The
    failure was invisible in normal output because the traceback went to stderr.
    """
    import inspect

    from nullres import cli
    from nullres.runlog import record_run as real_record

    helper = set(inspect.signature(cli._record).parameters) - {"cfg", "command"}
    backend = set(inspect.signature(real_record).parameters) - {
        "cfg", "command", "runs_dir", "repo"}
    assert helper == backend, (
        f"cli._record and record_run have drifted: "
        f"helper-only={helper - backend}, backend-only={backend - helper}"
    )

    monkeypatch.chdir(tmp_path)
    cli._record(cfg, "run", metrics={"a": 1}, verdict="KILLED",
                notes="n", variants=7)
    written = list((tmp_path / "runs").glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["variants"] == 7


def test_the_ledger_does_not_dirty_its_own_repo(cfg, tmp_path, monkeypatch):
    """`git_dirty` must mean "the CODE was uncommitted", nothing else.

    Writing a record puts a file in `runs/`, which shows up in `git status`. So
    the first run of a session recorded clean and every one after it recorded
    dirty — against completely pristine code. A flag that is true almost always
    carries no information, and this one exists to tell you whether the recorded
    SHA reproduces the result.
    """
    import subprocess

    from nullres.runlog import _git_state

    repo = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "code.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    assert _git_state(repo)[1] is False, "a clean checkout is not dirty"

    (repo / "runs").mkdir()
    (repo / "runs" / "20260101-000000-abcd1234.json").write_text("{}")
    assert _git_state(repo)[1] is False, "the ledger's own output is not a code change"

    (repo / "code.py").write_text("x = 2\n")
    assert _git_state(repo)[1] is True, "an actual code change must still register"


def test_a_logging_failure_never_kills_a_completed_run(cfg, monkeypatch, capsys):
    """Bookkeeping must not destroy a result that already computed."""
    from nullres import cli

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("nullres.runlog.record_run", boom)
    cli._record(cfg, "run")                     # must not raise
    assert "run log write failed" in capsys.readouterr().out


def test_corrupt_records_are_skipped_not_fatal(tmp_path):
    (tmp_path / "20260101-000000-deadbeef.json").write_text("{not json")
    (tmp_path / "20260101-000001-cafe0000.json").write_text('{"unexpected": 1}')
    assert load_runs(str(tmp_path)) == []


def test_near_miss_of_a_killed_run_is_flagged(cfg, tmp_path):
    """The capability markdown cannot provide: recognising a dead end."""
    record_run(cfg, "robust", verdict="KILLED", runs_dir=str(tmp_path))
    runs = load_runs(str(tmp_path))

    tweaked = copy.deepcopy(cfg)
    tweaked.sizing.min_hold = cfg.sizing.min_hold + 6      # tuning around it

    hits = find_similar(tweaked, runs, max_distance=3)
    assert hits, "re-running a killed config with one tweak was not flagged"
    distance, differing, record = hits[0]
    assert distance == 1 and differing == ["sizing.min_hold"]
    assert "KILLED" in format_warning(hits)


def test_a_different_asset_is_a_new_experiment_not_a_near_miss(cfg, tmp_path):
    """Killing a rule on BTC says nothing about SOL.

    Distance counted every key equally, so `data.symbol` scored the same as
    `sizing.min_hold` off by one — and testing a dead rule on a different asset
    got flagged as re-treading a dead end. The warning must stay rare enough to
    keep being read.
    """
    record_run(cfg, "robust", verdict="KILLED", runs_dir=str(tmp_path))
    runs = load_runs(str(tmp_path))

    other_asset = copy.deepcopy(cfg)
    other_asset.data.symbol = "SOLUSDT"
    assert not find_similar(other_asset, runs), "a different asset is a new experiment"

    other_period = copy.deepcopy(cfg)
    other_period.data.start = "2018-01"
    assert not find_similar(other_period, runs), "a different era is a new experiment"

    # ...but tuning the same experiment on the same data still warns.
    tuned = copy.deepcopy(cfg)
    tuned.sizing.min_hold += 6
    assert find_similar(tuned, runs)


def test_a_genuinely_different_config_is_not_flagged(cfg, tmp_path):
    """The warning must stay quiet, or it will be ignored."""
    record_run(cfg, "robust", verdict="KILLED", runs_dir=str(tmp_path))
    runs = load_runs(str(tmp_path))

    different = copy.deepcopy(cfg)
    different.data.symbol = "ETHUSDT"
    different.data.interval = "1d"
    different.label.horizon = 5
    different.label.kind = "fwd_return"
    different.sizing.min_hold = 3
    different.model.kind = "logistic"

    assert not find_similar(different, runs, max_distance=3)
    assert format_warning([]) == ""


def test_only_killed_runs_trigger_the_warning(cfg, tmp_path):
    record_run(cfg, "robust", verdict="SURVIVED", runs_dir=str(tmp_path))
    runs = load_runs(str(tmp_path))
    assert not find_similar(cfg, runs, max_distance=3, verdict="KILLED")
    assert find_similar(cfg, runs, max_distance=3, verdict="SURVIVED")


def test_killing_one_strategy_does_not_condemn_another(cfg, tmp_path):
    """`robust` pins its strategy into the logged config.

    Without that, the verdict attaches to the data/label/model settings alone,
    and killing donchian on BTCUSDT 4h would warn you off every other strategy
    on the same bars — which is both wrong and the fastest way to teach someone
    to ignore the warning.
    """
    killed = copy.deepcopy(cfg)
    killed.strategies = ["donchian"]
    record_run(killed, "robust", verdict="KILLED", runs_dir=str(tmp_path))
    runs = load_runs(str(tmp_path))

    another = copy.deepcopy(cfg)
    another.strategies = ["ml_meta"]
    assert not find_similar(another, runs, max_distance=0)

    same = copy.deepcopy(cfg)
    same.strategies = ["donchian"]
    assert find_similar(same, runs, max_distance=0)


def test_exact_rerun_reports_zero_distance(cfg, tmp_path):
    record_run(cfg, "robust", verdict="KILLED", runs_dir=str(tmp_path))
    hits = find_similar(cfg, load_runs(str(tmp_path)), max_distance=0)
    assert hits and hits[0][0] == 0
    assert "exact re-run" in format_warning(hits)
