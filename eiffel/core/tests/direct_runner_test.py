"""Tests for the direct TOML configuration runtime."""

from pathlib import Path

import pytest

from eiffel.direct_runner import (
    DATASETS,
    DATASET_PRESETS,
    MODEL_ATTACKS,
    ExperimentConfigError,
    apply_overrides,
    load_profile,
    resolve_profile,
)

ROOT = Path(__file__).resolve().parents[3]




def _portable_profile(dataset_name: str = "synthetic_stress") -> dict:
    return {
        "experiment": {"seed": 2026, "num_clients": 10, "rounds": 5},
        "dataset": {"name": dataset_name},
        "training": {"local_epochs": 1},
        "attack": {
            "mechanism": "sign_flip",
            "malicious_fraction": 0.2,
        },
        "storage": {"enabled": True, "capture_inference": True},
    }


def test_every_dataset_alias_has_a_structural_preset():
    canonical = set(DATASETS.values())
    assert canonical == set(DATASET_PRESETS)


@pytest.mark.parametrize(
    ("dataset_name", "task", "partition", "model_name", "num_classes"),
    [
        ("cicids", "binary", "iid", "popoola", 2),
        ("cicids_datacenter", "family_aware", "dirichlet", "popoola", 2),
        ("nb15", "binary", "iid", "popoola", 2),
        ("nb15_datacenter", "family_aware", "dirichlet", "popoola", 2),
        ("toniot", "binary", "iid", "popoola", 2),
        ("botiot", "binary", "iid", "popoola", 2),
        ("synthetic_stress", "binary", "preassigned", "stress_mlp", 6),
        ("mirage_app3", "multiclass", "preassigned", "p4p_mlp", 3),
        ("cesnet", "multiclass", "preassigned", "p4p_mlp", 50),
        ("ciciot_binary", "binary", "preassigned", "popoola", 2),
        ("ciciot_family", "multiclass", "preassigned", "p4p_mlp", 8),
        ("ciciot_fine", "multiclass", "preassigned", "p4p_mlp", 34),
    ],
)
def test_portable_profile_infers_dataset_structure(
    dataset_name: str,
    task: str,
    partition: str,
    model_name: str,
    num_classes: int,
):
    resolved = resolve_profile(_portable_profile(dataset_name))
    assert resolved["dataset"]["task"] == task
    assert resolved["dataset"]["num_classes"] == num_classes
    assert resolved["partition"]["type"] == partition
    assert resolved["model"]["name"] == model_name
    assert resolved["model"]["num_classes"] == num_classes


@pytest.mark.parametrize(
    "mechanism",
    sorted(set(MODEL_ATTACKS.values()) - {"none"}),
)
@pytest.mark.parametrize(
    "dataset_name",
    ["cicids", "synthetic_stress", "mirage_app3", "cesnet", "ciciot_binary", "ciciot_family", "ciciot_fine"],
)
def test_portable_model_attack_resolves_across_dataset_families(
    dataset_name: str,
    mechanism: str,
):
    profile = _portable_profile(dataset_name)
    profile["attack"]["mechanism"] = mechanism
    if mechanism == "targeted_family_poisoning":
        profile["attack"]["targeted"] = {"target_family": "configured-target"}
    resolved = resolve_profile(profile)
    assert resolved["attack"]["model_attack"]["mechanism"] == mechanism
    assert resolved["experiment"]["num_attackers"] == 2


def test_dataset_override_reuses_same_profile_without_dataset_specific_edits():
    base = _portable_profile("synthetic_stress")
    switched = apply_overrides(base, ["dataset=cesnet"])
    resolved = resolve_profile(switched)
    assert resolved["dataset"]["registry"] == "cesnet/quicext25_top50"
    assert resolved["dataset"]["task"] == "multiclass"
    assert resolved["partition"]["type"] == "preassigned"
    assert resolved["model"]["name"] == "p4p_mlp"
    assert resolved["model"]["num_classes"] == 50


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


@pytest.mark.parametrize(
    "profile_path",
    sorted((ROOT / "experiments" / "toml").glob("*.toml")),
    ids=lambda path: path.name,
)
def test_every_committed_profile_resolves_without_hydra(profile_path):
    profile = load_profile(profile_path)
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
