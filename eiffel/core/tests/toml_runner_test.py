"""Tests for the direct TOML configuration layer."""

from pathlib import Path

import pytest

from eiffel.direct_runner import (
    ExperimentConfigError,
    load_profile,
    resolve_profile,
)

ROOT = Path(__file__).resolve().parents[3]


def test_sign_flip_resolves_to_plain_runtime_config():
    resolved = resolve_profile({
        "experiment": {"seed": 2026, "num_clients": 10, "rounds": 20},
        "dataset": {"name": "cicids"},
        "partition": {"type": "dirichlet", "dirichlet_alpha": 0.5},
        "model": {"name": "mlp"},
        "training": {"local_epochs": 1, "batch_size": 128},
        "attack": {
            "mechanism": "sign_flip",
            "malicious_fraction": 0.2,
            "strength": 3.0,
        },
        "aggregation": {"name": "fedavg"},
    })

    assert resolved["dataset"]["registry"] == "nfv2/sampled/cicids"
    assert resolved["experiment"]["num_benign"] == 8
    assert resolved["experiment"]["num_attackers"] == 2
    assert resolved["partition"]["type"] == "dirichlet"
    assert resolved["model"]["name"] == "popoola"
    assert resolved["attack"]["model_attack"]["mechanism"] == "sign_flip"
    assert resolved["attack"]["model_attack"]["strength"] == 3.0
    assert resolved["attack"]["data_poisoning"] is None


def test_synthetic_multiclass_keeps_generated_client_assignment():
    resolved = resolve_profile({
        "experiment": {"num_clients": 10, "rounds": 5},
        "dataset": {
            "name": "synthetic_stress",
            "task": "multiclass",
            "num_classes": 6,
        },
        "partition": {"type": "dirichlet", "dirichlet_alpha": 0.5},
        "model": {"name": "stress_mlp"},
        "attack": {"mechanism": "none", "malicious_fraction": 0.0},
    })

    assert resolved["dataset"]["partition_mode"] == "dirichlet"
    assert resolved["partition"]["type"] == "preassigned"
    assert resolved["model"]["task"] == "multiclass"
    assert resolved["model"]["num_classes"] == 6


def test_fixed_preprocessed_dataset_rejects_repartitioning():
    with pytest.raises(ExperimentConfigError, match="preassigned"):
        resolve_profile({
            "experiment": {"num_clients": 10, "rounds": 5},
            "dataset": {
                "name": "mirage_app3",
                "task": "multiclass",
                "num_classes": 3,
            },
            "partition": {"type": "dirichlet"},
            "model": {"name": "p4p_mlp"},
            "attack": {"mechanism": "none"},
        })


def test_multiclass_label_flip_resolves_explicit_mapping():
    resolved = resolve_profile({
        "experiment": {"num_clients": 10, "rounds": 5},
        "dataset": {
            "name": "ciciot_family",
            "task": "multiclass",
            "num_classes": 8,
        },
        "partition": {"type": "preassigned"},
        "model": {"name": "p4p_mlp"},
        "attack": {
            "mechanism": "label_flip",
            "malicious_fraction": 0.2,
            "poison_rate": 1.0,
            "source_class": 1,
            "destination_class": 2,
        },
    })

    poison = resolved["attack"]["data_poisoning"]
    assert poison["type"] == "targeted"
    assert poison["source_class"] == 1
    assert poison["destination_class"] == 2
    assert resolved["attack"]["model_attack"]["mechanism"] == "none"


def test_targeted_family_requires_probe_capture():
    with pytest.raises(ExperimentConfigError, match="capture_inference"):
        resolve_profile({
            "experiment": {"num_clients": 6, "rounds": 5},
            "dataset": {"name": "synthetic_stress"},
            "partition": {"type": "iid"},
            "model": {"name": "stress_mlp"},
            "attack": {
                "mechanism": "targeted_family_poisoning",
                "malicious_fraction": 0.33,
                "targeted": {"target_family": "Botnet"},
            },
            "storage": {"enabled": True, "capture_inference": False},
        })


def test_window_label_flip_becomes_stateful_selector():
    resolved = resolve_profile({
        "experiment": {"num_clients": 4, "rounds": 3},
        "dataset": {"name": "synthetic_stress"},
        "partition": {"type": "iid"},
        "model": {"name": "stress_mlp"},
        "attack": {
            "mechanism": "label_flip",
            "malicious_fraction": 0.25,
            "poison_rate": 0.5,
            "schedule": {"type": "window", "start_round": 2, "end_round": 2},
        },
    })
    assert resolved["attack"]["data_poisoning"]["profile"] == "0.0+0.5{2}-0.5{3}"


@pytest.mark.parametrize("profile_name", [
    "synthetic_50k_quick_clean.toml",
    "synthetic_50k_quick_sign_flip.toml",
    "synthetic_50k_min_max.toml",
    "mirage_app3_quick_clean.toml",
    "cesnet_top50_clean.toml",
    "ciciot_binary_clean.toml",
    "ciciot_family_label_flip_targeted.toml",
    "dc_cicids_clean.toml",
])
def test_committed_profiles_resolve_without_hydra(profile_name):
    profile = load_profile(ROOT / "experiments" / "toml" / profile_name)
    resolved = resolve_profile(profile)
    assert resolved["experiment"]["rounds"] >= 1
    assert resolved["dataset"]["registry"]
    assert "model_attack" in resolved["attack"]


def test_source_tree_has_no_hydra_or_omegaconf_imports():
    offenders = []
    for path in (ROOT / "eiffel").rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "import hydra" in text or "from hydra" in text or "omegaconf" in text:
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, offenders
    assert not (ROOT / "eiffel" / "conf").exists()
