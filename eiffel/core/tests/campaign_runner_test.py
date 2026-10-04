"""Tests for deterministic campaign sweep expansion."""

from eiffel.campaign_runner import build_variants


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
