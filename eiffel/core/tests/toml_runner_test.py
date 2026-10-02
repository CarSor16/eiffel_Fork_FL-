"""Tests for the TOML compatibility translation layer."""

from pathlib import Path

import pytest

from eiffel.toml_runner import TomlExperimentError, load_profile, profile_to_overrides


def test_sign_flip_translation():
    profile = {
        "experiment": {"seed": 2026, "num_clients": 10, "rounds": 20},
        "dataset": {"name": "cicids"},
        "partition": {"type": "dirichlet", "dirichlet_alpha": 0.5},
        "model": {"name": "mlp"},
        "training": {"local_epochs": 1, "batch_size": 128},
        "attack": {
            "mechanism": "sign_flip",
            "malicious_fraction": 0.2,
            "strength": 3.0,
            "schedule": {"type": "continuous"},
        },
        "aggregation": {"name": "fedavg"},
    }

    overrides = profile_to_overrides(profile)

    assert "num_clients=8" in overrides
    assert "num_attackers=2" in overrides
    assert "+datasets=nfv2/sampled/cicids" in overrides
    assert "partitioner=dirichlet" in overrides
    assert "partitioner.alpha=0.5" in overrides
    assert "model=popoola" in overrides
    assert "model_attack=sign_flip" in overrides
    assert "model_attack.strength=3.0" in overrides
    assert "poisoning/profile=clean" in overrides


def test_on_off_translation():
    profile = {
        "experiment": {"num_clients": 10},
        "dataset": {"name": "nb15"},
        "attack": {
            "mechanism": "colluding_sign_flip",
            "malicious_fraction": 0.2,
            "schedule": {
                "type": "on_off",
                "start_round": 1,
                "on_rounds": 3,
                "off_rounds": 3,
            },
        },
        "aggregation": {"name": "fedavg"},
    }

    overrides = profile_to_overrides(profile)

    assert "+datasets=nfv2/sampled/nb15" in overrides
    assert "model_attack=colluding_sign_flip" in overrides
    assert "++model_attack.schedule.period=6" in overrides
    assert "++model_attack.schedule.active_rounds=3" in overrides


def test_model_scaling_uses_scale_factor():
    profile = {
        "experiment": {"num_clients": 5},
        "dataset": {"name": "cicids"},
        "attack": {
            "mechanism": "model_scaling",
            "malicious_fraction": 0.2,
            "strength": 12.0,
        },
        "aggregation": {"name": "fedavg"},
    }

    overrides = profile_to_overrides(profile)
    assert "model_attack=scaling" in overrides
    assert "model_attack.scale_factor=12.0" in overrides



def test_synthetic_50k_translation():
    profile = {
        "experiment": {"seed": 2026, "num_clients": 10, "rounds": 5},
        "dataset": {
            "name": "synthetic_stress",
            "samples_per_client": 5000,
            "central_test_size": 12000,
            "num_features": 32,
        },
        "partition": {"type": "dirichlet", "dirichlet_alpha": 0.5},
        "model": {"name": "stress_mlp", "hidden1": 64, "hidden2": 32},
        "training": {"local_epochs": 1, "learning_rate": 0.001},
        "attack": {"mechanism": "none", "malicious_fraction": 0.0},
        "aggregation": {"name": "fedavg"},
    }

    overrides = profile_to_overrides(profile)

    assert "+datasets=synthetic/stress" in overrides
    assert "datasets.synthetic_stress.num_clients=10" in overrides
    assert "datasets.synthetic_stress.samples_per_client=5000" in overrides
    assert "datasets.synthetic_stress.central_test_size=12000" in overrides
    assert "datasets.synthetic_stress.partition_mode=dirichlet" in overrides
    assert "datasets.synthetic_stress.dirichlet_alpha=0.5" in overrides
    assert "partitioner=preassigned" in overrides
    assert "model=stress_mlp" in overrides



def test_synthetic_multiclass_translation():
    profile = {
        "experiment": {"seed": 2026, "num_clients": 10, "rounds": 5},
        "dataset": {
            "name": "synthetic_stress",
            "task": "multiclass",
            "num_classes": 6,
        },
        "partition": {"type": "dirichlet", "dirichlet_alpha": 0.5},
        "model": {"name": "stress_mlp"},
        "attack": {"mechanism": "sign_flip", "malicious_fraction": 0.2},
        "aggregation": {"name": "fedavg"},
    }

    overrides = profile_to_overrides(profile)

    assert "datasets.synthetic_stress.task=multiclass" in overrides
    assert "++model.task=multiclass" in overrides
    assert "++model.num_classes=6" in overrides
    assert "model_attack=sign_flip" in overrides
    assert "num_clients=8" in overrides
    assert "num_attackers=2" in overrides


def test_mirage_multiclass_translation_preserves_preassigned_clients():
    profile = {
        "experiment": {"seed": 2026, "num_clients": 10, "rounds": 5},
        "dataset": {
            "name": "mirage_app3",
            "task": "multiclass",
            "num_classes": 3,
        },
        "partition": {"type": "preassigned"},
        "model": {"name": "p4p_mlp"},
        "training": {"local_epochs": 1, "batch_size": 128},
        "attack": {"mechanism": "none", "malicious_fraction": 0.0},
        "aggregation": {"name": "fedavg"},
    }

    overrides = profile_to_overrides(profile)

    assert "+datasets=mirage/app3" in overrides
    assert "partitioner=preassigned" in overrides
    assert "++model.task=multiclass" in overrides
    assert "++model.num_classes=3" in overrides
    assert "model=p4p_mlp" in overrides


def test_mirage_rejects_repartitioning():
    profile = {
        "experiment": {"num_clients": 10, "rounds": 5},
        "dataset": {
            "name": "mirage_app3",
            "task": "multiclass",
            "num_classes": 3,
        },
        "partition": {"type": "dirichlet", "dirichlet_alpha": 0.5},
        "model": {"name": "p4p_mlp"},
        "attack": {"mechanism": "none", "malicious_fraction": 0.0},
        "aggregation": {"name": "fedavg"},
    }

    with pytest.raises(TomlExperimentError, match="preassigned"):
        profile_to_overrides(profile)


@pytest.mark.parametrize(
    "profile_name",
    (
        "mirage_app3_quick_clean.toml",
        "mirage_app3_quick_sign_flip.toml",
    ),
)
def test_committed_mirage_profiles_translate(profile_name):
    root = Path(__file__).resolve().parents[3]
    profile = load_profile(root / "experiments" / "toml" / profile_name)
    overrides = profile_to_overrides(profile)

    assert "+datasets=mirage/app3" in overrides
    assert "partitioner=preassigned" in overrides
    assert "++model.task=multiclass" in overrides
    assert "++model.num_classes=3" in overrides


def test_multiclass_label_flip_requires_explicit_class_mapping():
    profile = {
        "experiment": {"num_clients": 10, "rounds": 5},
        "dataset": {
            "name": "synthetic_stress",
            "task": "multiclass",
            "num_classes": 6,
        },
        "attack": {
            "mechanism": "label_flip",
            "malicious_fraction": 0.2,
            "poison_rate": 0.5,
        },
        "aggregation": {"name": "fedavg"},
    }

    with pytest.raises(TomlExperimentError, match="source_class"):
        profile_to_overrides(profile)


def test_multiclass_is_not_silently_enabled_for_real_nfv2():
    profile = {
        "experiment": {"num_clients": 10, "rounds": 5},
        "dataset": {"name": "cicids", "task": "multiclass"},
        "attack": {"mechanism": "none", "malicious_fraction": 0.0},
        "aggregation": {"name": "fedavg"},
    }

    with pytest.raises(TomlExperimentError, match="synthetic_stress"):
        profile_to_overrides(profile)



@pytest.mark.parametrize(
    ("mechanism", "nested_key", "expected_group", "expected_override"),
    [
        (
            "min_max",
            "min_max",
            "min_max",
            "++model_attack.max_lambda=7.5",
        ),
        (
            "min_sum",
            "min_sum",
            "min_sum",
            "++model_attack.search_steps=18",
        ),
        (
            "adaptive_stealth",
            "adaptive",
            "adaptive_stealth",
            "++model_attack.l2_quantile=0.9",
        ),
        (
            "heterogeneity_aware_mimicry",
            "heterogeneity",
            "heterogeneity_aware_mimicry",
            "++model_attack.neighbors=2",
        ),
    ],
)
def test_advanced_attack_translation_is_dataset_portable(
    mechanism,
    nested_key,
    expected_group,
    expected_override,
):
    profile = {
        "experiment": {"num_clients": 8, "rounds": 6},
        # Deliberately use a non-built-in dataset name. hydra_group is the portable
        # escape hatch for any Eiffel dataset configuration group.
        "dataset": {
            "name": "my_future_dataset",
            "hydra_group": "custom/my_future_dataset",
            "task": "binary",
        },
        "partition": {"type": "iid"},
        "model": {"name": "mlp"},
        "attack": {
            "mechanism": mechanism,
            "malicious_fraction": 0.25,
            nested_key: {},
        },
        "aggregation": {"name": "fedavg"},
    }
    if mechanism == "min_max":
        profile["attack"][nested_key] = {
            "direction": "sign",
            "max_lambda": 7.5,
        }
    elif mechanism == "min_sum":
        profile["attack"][nested_key] = {"search_steps": 18}
    elif mechanism == "adaptive_stealth":
        profile["attack"][nested_key] = {"l2_quantile": 0.9}
    else:
        profile["attack"][nested_key] = {
            "neighbors": 2,
            "similarity": "l2",
            "mimicry_lambda": 0.7,
        }

    overrides = profile_to_overrides(profile)

    assert "+datasets=custom/my_future_dataset" in overrides
    assert f"model_attack={expected_group}" in overrides
    assert expected_override in overrides
    assert "num_attackers=2" in overrides
    assert "poisoning/profile=clean" in overrides


def test_targeted_family_translation_uses_configurable_dataset_metadata():
    profile = {
        "experiment": {"num_clients": 8, "rounds": 8},
        "dataset": {
            "name": "another_dataset",
            "hydra_group": "custom/another_dataset",
            "task": "binary",
        },
        "partition": {"type": "iid"},
        "model": {"name": "mlp"},
        "attack": {
            "mechanism": "targeted_family_poisoning",
            "malicious_fraction": 0.25,
            "targeted": {
                "target_family": "Arbitrary-Threat-Name",
                "target_amplification": 2.25,
                "target_mimicry_lambda": 0.2,
            },
            "schedule": {"type": "late", "start_round": 4},
        },
        "aggregation": {"name": "fedavg"},
        "storage": {"capture_inference": True, "probe_size": 128},
    }

    overrides = profile_to_overrides(profile)

    assert "+datasets=custom/another_dataset" in overrides
    assert "model_attack=targeted_family_poisoning" in overrides
    assert "++model_attack.target_family=Arbitrary-Threat-Name" in overrides
    assert "++model_attack.target_amplification=2.25" in overrides
    assert "++model_attack.target_mimicry_lambda=0.2" in overrides
    assert "storage.capture_inference=true" in overrides
    assert "model_attack.schedule.type=late" in overrides
    assert "model_attack.schedule.start_round=4" in overrides


def test_targeted_family_requires_probe_capture():
    profile = {
        "experiment": {"num_clients": 6, "rounds": 5},
        "dataset": {
            "name": "portable",
            "hydra_group": "custom/portable",
        },
        "model": {"name": "mlp"},
        "attack": {
            "mechanism": "targeted_family_poisoning",
            "malicious_fraction": 0.33,
            "targeted": {"target_family": "Threat-A"},
        },
        "aggregation": {"name": "fedavg"},
        "storage": {"capture_inference": False},
    }

    with pytest.raises(TomlExperimentError, match="capture_inference"):
        profile_to_overrides(profile)


@pytest.mark.parametrize("mechanism", ["min_max", "min_sum", "adaptive_stealth"])
def test_optimized_attacks_require_two_benign_clients(mechanism):
    profile = {
        "experiment": {"num_clients": 3, "rounds": 5},
        "dataset": {
            "name": "portable",
            "hydra_group": "custom/portable",
        },
        "model": {"name": "mlp"},
        "attack": {
            "mechanism": mechanism,
            "malicious_fraction": 0.67,
        },
        "aggregation": {"name": "fedavg"},
    }

    with pytest.raises(TomlExperimentError, match="two benign"):
        profile_to_overrides(profile)



ADVANCED_PROFILE_NAMES = (
    "synthetic_50k_min_max.toml",
    "synthetic_50k_min_sum.toml",
    "synthetic_50k_adaptive_stealth.toml",
    "synthetic_50k_adaptive_stealth_gradual.toml",
    "synthetic_50k_heterogeneity_aware_mimicry.toml",
    "synthetic_50k_targeted_family_poisoning.toml",
)


@pytest.mark.parametrize("profile_name", ADVANCED_PROFILE_NAMES)
def test_committed_advanced_profiles_translate(profile_name):
    root = Path(__file__).resolve().parents[3]
    profile = load_profile(root / "experiments" / "toml" / profile_name)
    overrides = profile_to_overrides(profile)

    assert "strategy=instrumented_fedavg" in overrides
    assert "poisoning/profile=clean" in overrides
    assert any(value.startswith("model_attack=") for value in overrides)
    assert any(value.startswith("model_attack.schedule.type=") for value in overrides)

def test_storage_posthoc_inference_flags_translate():
    profile = {
        "experiment": {"num_clients": 2, "rounds": 2},
        "dataset": {"name": "synthetic_stress", "task": "binary"},
        "partition": {"type": "iid"},
        "model": {"name": "stress_mlp"},
        "attack": {"mechanism": "none", "malicious_fraction": 0.0},
        "aggregation": {"name": "fedavg"},
        "storage": {
            "capture_inference": True,
            "capture_logits": False,
            "capture_probe_features": False,
            "capture_global_inference": False,
            "probe_size": 64,
        },
    }

    overrides = profile_to_overrides(profile)

    assert "storage.capture_inference=true" in overrides
    assert "storage.capture_logits=false" in overrides
    assert "storage.capture_probe_features=false" in overrides
    assert "storage.capture_global_inference=false" in overrides
    assert "storage.probe_size=64" in overrides



def _targeted_probe_profile(storage):
    return {
        "experiment": {"num_clients": 6, "rounds": 5},
        "dataset": {
            "name": "portable",
            "hydra_group": "custom/portable",
        },
        "model": {"name": "mlp"},
        "attack": {
            "mechanism": "targeted_family_poisoning",
            "malicious_fraction": 0.33,
            "targeted": {"target_family": "Threat-A"},
        },
        "aggregation": {"name": "fedavg"},
        "storage": storage,
    }


def test_targeted_family_requires_enabled_storage():
    with pytest.raises(TomlExperimentError, match="storage.enabled=true"):
        profile_to_overrides(
            _targeted_probe_profile(
                {"enabled": False, "capture_inference": True, "probe_size": 64}
            )
        )


def test_targeted_family_requires_nonempty_probe():
    with pytest.raises(TomlExperimentError, match="probe_size > 0"):
        profile_to_overrides(
            _targeted_probe_profile(
                {"enabled": True, "capture_inference": True, "probe_size": 0}
            )
        )



def test_label_flip_temporal_selector_is_hydra_quoted():
    profile = {
        "experiment": {"num_clients": 4, "rounds": 3},
        "dataset": {"name": "synthetic_stress", "task": "binary"},
        "partition": {"type": "iid"},
        "model": {"name": "stress_mlp"},
        "attack": {
            "mechanism": "label_flip",
            "malicious_fraction": 0.25,
            "poison_rate": 0.5,
            "objective": "untargeted",
            "schedule": {
                "type": "window",
                "start_round": 2,
                "end_round": 2,
            },
        },
        "aggregation": {"name": "fedavg"},
    }

    overrides = profile_to_overrides(profile)

    assert "attacks.0.profile='0.0+0.5{2}-0.5{3}'" in overrides

@pytest.mark.parametrize(
    ("poison_rate", "expected"),
    [
        (0.0, "attacks.0.profile='0.0'"),
        (0.5, "attacks.0.profile='0.5'"),
        (1.0, "attacks.0.profile='1'"),
    ],
)
def test_label_flip_scalar_selector_is_always_hydra_string(
    poison_rate,
    expected,
):
    profile = {
        "experiment": {"num_clients": 4, "rounds": 3},
        "dataset": {"name": "synthetic_stress", "task": "binary"},
        "partition": {"type": "iid"},
        "model": {"name": "stress_mlp"},
        "attack": {
            "mechanism": "label_flip",
            "malicious_fraction": 0.25,
            "poison_rate": poison_rate,
            "objective": "untargeted",
            "schedule": {"type": "continuous"},
        },
        "aggregation": {"name": "fedavg"},
    }

    overrides = profile_to_overrides(profile)

    assert expected in overrides



@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("cicids_full", "+datasets=nfv2/full/cicids"),
        ("cicids_datacenter", "+datasets=nfv2/datacenter/cicids"),
        ("nb15_full", "+datasets=nfv2/full/nb15"),
        ("nb15_datacenter", "+datasets=nfv2/datacenter/nb15"),
    ],
)
def test_full_and_datacenter_dataset_aliases(name, expected):
    profile = {
        "experiment": {"num_clients": 10, "rounds": 30},
        "dataset": {"name": name, "task": "family_aware"},
        "partition": {"type": "dirichlet", "dirichlet_alpha": 0.5},
        "model": {"name": "popoola"},
        "training": {"local_epochs": 1, "batch_size": 512},
        "attack": {"mechanism": "none", "malicious_fraction": 0.0},
        "aggregation": {"name": "fedavg"},
    }
    overrides = profile_to_overrides(profile)
    assert expected in overrides
    assert "partitioner=dirichlet" in overrides
    assert "partitioner.alpha=0.5" in overrides


DATACENTER_PROFILE_NAMES = tuple(
    f"dc_{dataset}_{attack}.toml"
    for dataset in ("cicids", "nb15")
    for attack in (
        "clean",
        "label_flip_targeted",
        "sign_flip",
        "model_scaling",
        "gaussian_noise",
        "lie",
        "gradient_mimicry",
        "colluding_sign_flip",
        "min_max",
        "min_sum",
        "adaptive_stealth",
        "heterogeneity_aware_mimicry",
        "targeted_family_poisoning",
    )
)


@pytest.mark.parametrize("profile_name", DATACENTER_PROFILE_NAMES)
def test_committed_datacenter_profiles_translate(profile_name):
    root = Path(__file__).resolve().parents[3]
    profile = load_profile(root / "experiments" / "toml" / profile_name)
    overrides = profile_to_overrides(profile)
    assert "strategy=instrumented_fedavg" in overrides
    assert "num_rounds=30" in overrides
    assert any(value.startswith("+datasets=nfv2/datacenter/") for value in overrides)
