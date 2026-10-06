"""Tests for the file-based runtime control plane and terminal status view."""

import json
from pathlib import Path

import pytest

from eiffel.control import (
    ControlPlane,
    RuntimeControlError,
    apply_changes,
    validate_runtime_changes,
    write_control,
)
from eiffel.tui import publish_changes, status_text


def test_runtime_controls_reject_static_experiment_fields():
    with pytest.raises(RuntimeControlError, match="static during a run"):
        validate_runtime_changes({"dataset.name": "cesnet"})
    with pytest.raises(RuntimeControlError, match="static during a run"):
        validate_runtime_changes({"model.name": "cnn1d"})


@pytest.mark.parametrize(
    "key",
    [
        "attack.malicious_fraction",
        "attack.malicious_client_ids",
        "attack.poison_rate",
    ],
)
def test_runtime_controls_reject_client_or_data_assignment_changes(key):
    with pytest.raises(RuntimeControlError, match="cannot be changed"):
        validate_runtime_changes({key: 0.5})


def test_control_revision_is_applied_only_at_requested_round(tmp_path: Path):
    path = tmp_path / "control.json"
    write_control(
        path,
        revision=4,
        apply_from_round=3,
        changes={
            "attack.strength": 2.0,
            "aggregation.name": "median",
        },
    )
    plane = ControlPlane(path, enabled=True)
    assert plane.read_for_round(1) is None
    assert plane.read_for_round(2) is None
    revision, changes = plane.read_for_round(3)
    assert revision == 4
    assert changes["attack.strength"] == 2.0
    assert changes["aggregation.name"] == "median"
    assert plane.read_for_round(4) is None


def test_apply_changes_updates_only_runtime_sections():
    attack = {"mechanism": "sign_flip", "strength": 1.0}
    aggregation = {"name": "fedavg"}
    defense = {"name": "none"}
    apply_changes(
        attack=attack,
        aggregation=aggregation,
        defense=defense,
        changes={
            "attack.strength": 3.0,
            "aggregation.name": "trimmed_mean",
            "aggregation.trim_ratio": 0.2,
            "defense.name": "norm_clipping",
            "defense.max_norm": 5.0,
        },
    )
    assert attack["strength"] == 3.0
    assert aggregation == {"name": "trimmed_mean", "trim_ratio": 0.2}
    assert defense == {"name": "norm_clipping", "max_norm": 5.0}


def test_tui_publish_increments_revision_and_parses_values(tmp_path: Path):
    publish_changes(
        tmp_path,
        ["attack.strength=2.5", "aggregation.name=median"],
        apply_from_round=2,
    )
    first = json.loads((tmp_path / "control.json").read_text(encoding="utf-8"))
    assert first["revision"] == 0
    assert first["set"]["attack.strength"] == 2.5

    publish_changes(
        tmp_path,
        ["defense.name=norm_clipping", "defense.max_norm=4"],
        apply_from_round=3,
    )
    second = json.loads((tmp_path / "control.json").read_text(encoding="utf-8"))
    assert second["revision"] == 1
    assert second["apply_from_round"] == 3
    assert second["set"]["defense.max_norm"] == 4


def test_status_view_reads_resolved_profile_and_latest_event(tmp_path: Path):
    (tmp_path / "resolved_profile.json").write_text(
        json.dumps(
            {
                "experiment": {"rounds": 10},
                "dataset": {"registry": "synthetic/stress", "task": "binary"},
                "model": {"name": "cnn1d", "task": "binary"},
                "attack": {"mechanism": "sign_flip"},
                "aggregation": {"name": "fedavg"},
                "defense": {"name": "none"},
            }
        ),
        encoding="utf-8",
    )
    events = [
        {
            "event": "round_start",
            "round": 4,
            "total_rounds": 10,
            "attack": "sign_flip",
            "aggregation": "median",
            "defense": "norm_clipping",
        },
        {
            "event": "round_complete",
            "round": 3,
            "total_rounds": 10,
            "loss": 0.4,
            "metrics": {"accuracy": 0.9, "macro_f1": 0.8},
            "malicious_clients": 2,
            "attack_multiplier": 1.0,
        },
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    rendered = status_text(tmp_path)
    assert "Round        : 3/10" in rendered
    assert "Model        : cnn1d" in rendered
    assert "Aggregation  : median" in rendered
    assert "Defense      : norm_clipping" in rendered
    assert "Accuracy     : 0.900000" in rendered


def test_tui_honors_custom_control_paths_and_disabled_guard(tmp_path: Path):
    (tmp_path / "resolved_profile.json").write_text(
        json.dumps(
            {
                "control": {
                    "enabled": True,
                    "path": "runtime/live-control.json",
                    "events_path": "runtime/live-events.jsonl",
                }
            }
        ),
        encoding="utf-8",
    )
    path = publish_changes(
        tmp_path,
        ["aggregation.name=median"],
        apply_from_round=2,
    )
    assert path == tmp_path / "runtime" / "live-control.json"
    assert path.exists()

    (tmp_path / "resolved_profile.json").write_text(
        json.dumps({"control": {"enabled": False}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeControlError, match="disabled"):
        publish_changes(
            tmp_path,
            ["aggregation.name=fedavg"],
            apply_from_round=3,
        )
