"""Tests for the TOML compatibility translation layer."""

from eiffel.toml_runner import profile_to_overrides


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
