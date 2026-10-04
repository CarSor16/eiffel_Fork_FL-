"""Tests for deterministic campaign sweep expansion."""

import csv
from pathlib import Path
from types import SimpleNamespace

from eiffel.campaign_runner import build_variants, main


def _base_profile():
    return {
        "experiment": {
            "name": "campaign-test",
            "seed": 2026,
            "num_clients": 10,
            "rounds": 30,
        },
        "dataset": {
            "name": "mirage_app3",
            "task": "multiclass",
            "num_classes": 3,
        },
        "partition": {"type": "preassigned"},
        "model": {"name": "p4p_mlp"},
        "attack": {
            "mechanism": "sign_flip",
            "malicious_fraction": 0.2,
            "strength": 3.0,
        },
        "aggregation": {"name": "fedavg"},
    }


def test_all_two_client_combinations_produce_45_variants():
    variants = build_variants(
        _base_profile(),
        malicious_combination_size=2,
    )

    assert len(variants) == 45
    observed = {
        tuple(profile["attack"]["malicious_client_ids"])
        for profile in variants
    }
    assert (0, 1) in observed
    assert (8, 9) in observed
    assert all("malicious_fraction" not in profile["attack"] for profile in variants)


def test_seed_and_parameter_sweeps_form_cartesian_product():
    variants = build_variants(
        _base_profile(),
        seeds=[2026, 2027, 2028],
        sweep_specs=[
            ("attack.strength", [1.0, 3.0]),
            ("attack.schedule.type", ["continuous", "late"]),
        ],
    )

    assert len(variants) == 12
    assert {profile["experiment"]["seed"] for profile in variants} == {
        2026,
        2027,
        2028,
    }
    assert {profile["attack"]["strength"] for profile in variants} == {1.0, 3.0}
    assert {
        profile["attack"]["schedule"]["type"]
        for profile in variants
    } == {"continuous", "late"}


def test_campaign_checkpoints_manifest_before_training_and_after_each_run(tmp_path, monkeypatch):
    output = tmp_path / "campaign"
    monkeypatch.setattr("eiffel.campaign_runner.load_profile", lambda path: _base_profile())
    observed = []

    def run(command, **kwargs):
        with (output / "campaign_manifest.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        observed.append(rows)
        assert rows[-1]["return_code"] == "running"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("eiffel.campaign_runner.subprocess.run", run)
    assert main(["unused.toml", "--output-root", str(output), "--seeds", "2026,2027"]) == 0
    assert len(observed) == 2
    assert observed[1][0]["return_code"] == "0"


def test_campaign_interrupt_preserves_completed_and_interrupted_runs(tmp_path, monkeypatch):
    output = tmp_path / "campaign"
    monkeypatch.setattr("eiffel.campaign_runner.load_profile", lambda path: _base_profile())
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("eiffel.campaign_runner.subprocess.run", run)
    assert main(["unused.toml", "--output-root", str(output), "--seeds", "2026,2027"]) == 130
    with (output / "campaign_manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["return_code"] for row in rows] == ["0", "interrupted"]
    assert all((Path(row["output_dir"]) / "campaign_profile.json").is_file() for row in rows)
