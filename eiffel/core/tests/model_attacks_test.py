"""Tests for procedural model poisoning attacks and temporal schedules."""

import numpy as np
import pytest

from eiffel.attacks.model import apply_round_attack, attack_strength_for_round


def _updates():
    return [
        [np.array([1.0, 2.0], dtype=np.float32), np.array([1.0], dtype=np.float32)],
        [np.array([3.0, 4.0], dtype=np.float32), np.array([3.0], dtype=np.float32)],
        [np.array([5.0, 6.0], dtype=np.float32), np.array([5.0], dtype=np.float32)],
        [np.array([7.0, 8.0], dtype=np.float32), np.array([7.0], dtype=np.float32)],
    ]


def _cfg(mechanism, **kwargs):
    config = {
        "enabled": True,
        "mechanism": mechanism,
        "strength": 1.0,
        "schedule": {"type": "continuous", "start_round": 1},
    }
    config.update(kwargs)
    return config


def _assert_layout_and_finite(original, attacked):
    assert len(attacked) == len(original)
    for before_client, after_client in zip(original, attacked):
        assert len(before_client) == len(after_client)
        for before, after in zip(before_client, after_client):
            assert before.shape == after.shape
            assert after.dtype == np.float32
            assert np.all(np.isfinite(after))


def test_none_keeps_every_update_unchanged():
    original = _updates()
    attacked, multiplier = apply_round_attack(
        original,
        [False, False, True, True],
        _cfg("none"),
        server_round=1,
        total_rounds=5,
        seed=2026,
    )

    assert multiplier == 1.0
    _assert_layout_and_finite(original, attacked)
    for before_client, after_client in zip(original, attacked):
        for before, after in zip(before_client, after_client):
            np.testing.assert_array_equal(after, before)


def test_sign_flip_changes_only_malicious_clients():
    original = _updates()
    attacked, _ = apply_round_attack(
        original,
        [False, False, True, True],
        _cfg("sign_flip", strength=2.0),
        server_round=1,
        total_rounds=5,
        seed=2026,
    )

    _assert_layout_and_finite(original, attacked)
    for idx in (0, 1):
        for before, after in zip(original[idx], attacked[idx]):
            np.testing.assert_array_equal(after, before)
    for idx in (2, 3):
        for before, after in zip(original[idx], attacked[idx]):
            np.testing.assert_allclose(after, -2.0 * before)


def test_scaling_changes_only_malicious_clients():
    original = _updates()
    attacked, _ = apply_round_attack(
        original,
        [False, False, True, True],
        _cfg("scaling", scale_factor=3.0),
        server_round=1,
        total_rounds=5,
        seed=2026,
    )

    _assert_layout_and_finite(original, attacked)
    for idx in (0, 1):
        for before, after in zip(original[idx], attacked[idx]):
            np.testing.assert_array_equal(after, before)
    for idx in (2, 3):
        for before, after in zip(original[idx], attacked[idx]):
            np.testing.assert_allclose(after, 3.0 * before)


def test_gaussian_noise_is_deterministic_for_seed_round_and_malicious_only():
    original = _updates()
    args = dict(
        updates=original,
        malicious_mask=[False, False, True, True],
        config=_cfg("gaussian_noise", noise_std=0.5),
        server_round=3,
        total_rounds=5,
        seed=2026,
    )
    first, _ = apply_round_attack(**args)
    second, _ = apply_round_attack(**args)

    _assert_layout_and_finite(original, first)
    for idx in (0, 1):
        for before, after in zip(original[idx], first[idx]):
            np.testing.assert_array_equal(after, before)
    for idx in (2, 3):
        changed = False
        for before, after, repeated in zip(original[idx], first[idx], second[idx]):
            np.testing.assert_array_equal(after, repeated)
            changed = changed or not np.array_equal(after, before)
        assert changed


def test_lie_uses_same_round_benign_mean_and_std():
    original = _updates()
    attacked, _ = apply_round_attack(
        original,
        [False, False, True, True],
        _cfg("lie", lie_z=1.0),
        server_round=1,
        total_rounds=5,
        seed=2026,
    )

    expected = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ]
    _assert_layout_and_finite(original, attacked)
    for idx in (2, 3):
        for observed, wanted in zip(attacked[idx], expected):
            np.testing.assert_allclose(observed, wanted)


def test_gradient_mimicry_blends_with_benign_reference():
    original = _updates()
    attacked, _ = apply_round_attack(
        original,
        [False, False, True, True],
        _cfg("gradient_mimicry", mimicry_lambda=0.5),
        server_round=1,
        total_rounds=5,
        seed=2026,
    )

    benign_reference = [
        np.array([2.0, 3.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
    ]
    _assert_layout_and_finite(original, attacked)
    for idx in (2, 3):
        for observed, ref, raw in zip(attacked[idx], benign_reference, original[idx]):
            np.testing.assert_allclose(observed, 0.5 * ref + 0.5 * raw)


def test_colluding_sign_flip_submits_same_crafted_centroid():
    original = _updates()
    attacked, _ = apply_round_attack(
        original,
        [False, False, True, True],
        _cfg("colluding_sign_flip", strength=2.0),
        server_round=1,
        total_rounds=5,
        seed=2026,
    )

    expected = [
        np.array([-12.0, -14.0], dtype=np.float32),
        np.array([-12.0], dtype=np.float32),
    ]
    _assert_layout_and_finite(original, attacked)
    for idx in (2, 3):
        for observed, wanted in zip(attacked[idx], expected):
            np.testing.assert_allclose(observed, wanted)


def test_inactive_schedule_keeps_malicious_update_unchanged():
    original = _updates()
    config = _cfg(
        "sign_flip",
        schedule={"type": "late", "start_round": 4, "end_round": 6},
    )
    attacked, multiplier = apply_round_attack(
        original,
        [False, False, True, True],
        config,
        server_round=2,
        total_rounds=6,
        seed=2026,
    )

    assert multiplier == 0.0
    for before_client, after_client in zip(original, attacked):
        for before, after in zip(before_client, after_client):
            np.testing.assert_array_equal(after, before)


def test_schedule_multipliers():
    continuous = _cfg("sign_flip")["schedule"] | {"end_round": 6}
    late = {"type": "late", "start_round": 4, "end_round": 6}
    window = {"type": "window", "start_round": 2, "end_round": 3}
    on_off = {
        "type": "on_off",
        "start_round": 1,
        "end_round": 8,
        "period": 4,
        "active_rounds": 2,
    }
    gradual = {
        "type": "gradual",
        "start_round": 2,
        "end_round": 6,
        "ramp_rounds": 5,
        "gradual_start_strength": 0.2,
        "gradual_end_strength": 1.0,
    }

    def strength(schedule, round_number):
        return attack_strength_for_round(
            {"enabled": True, "schedule": schedule},
            round_number,
            total_rounds=8,
        )

    assert strength(continuous, 1) == 1.0
    assert strength(late, 3) == 0.0
    assert strength(late, 4) == 1.0
    assert strength(window, 1) == 0.0
    assert strength(window, 2) == 1.0
    assert strength(window, 4) == 0.0
    assert [strength(on_off, r) for r in range(1, 7)] == [
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
    ]
    np.testing.assert_allclose(
        [strength(gradual, r) for r in (2, 4, 6)],
        [0.2, 0.6, 1.0],
    )



def _alternate_layout_updates():
    """Different tensor layout to catch feature/model-shape assumptions."""
    return [
        [
            np.array([[0.20, -0.10], [0.05, 0.30]], dtype=np.float32),
            np.array([0.02, -0.03], dtype=np.float32),
        ],
        [
            np.array([[0.16, -0.08], [0.08, 0.27]], dtype=np.float32),
            np.array([0.01, -0.01], dtype=np.float32),
        ],
        [
            np.array([[0.22, -0.13], [0.04, 0.34]], dtype=np.float32),
            np.array([0.03, -0.04], dtype=np.float32),
        ],
        [
            np.array([[0.60, 0.55], [-0.40, -0.30]], dtype=np.float32),
            np.array([0.30, 0.20], dtype=np.float32),
        ],
        [
            np.array([[0.50, 0.45], [-0.35, -0.20]], dtype=np.float32),
            np.array([0.25, 0.15], dtype=np.float32),
        ],
    ]


@pytest.mark.parametrize("mechanism", ["min_max", "min_sum", "adaptive_stealth"])
def test_optimized_attacks_are_layout_agnostic_and_modify_only_malicious(mechanism):
    original = _alternate_layout_updates()
    config = _cfg(
        mechanism,
        direction="sign",
        search_steps=20,
        max_lambda=6.0,
        max_strength=6.0,
        constraint_margin=1.05,
        stealth_margin=1.10,
    )
    attacked, multiplier = apply_round_attack(
        original,
        [False, False, False, True, True],
        config,
        server_round=1,
        total_rounds=5,
        seed=2026,
    )

    assert multiplier == 1.0
    _assert_layout_and_finite(original, attacked)
    for idx in (0, 1, 2):
        for before, after in zip(original[idx], attacked[idx]):
            np.testing.assert_array_equal(after, before)

    # Coordinated optimized attacks submit one common crafted vector.
    for left, right in zip(attacked[3], attacked[4]):
        np.testing.assert_allclose(left, right)


def test_min_max_and_min_sum_require_enough_benign_reference_updates():
    original = _updates()
    for mechanism in ("min_max", "min_sum", "adaptive_stealth"):
        with pytest.raises(ValueError, match="at least two benign"):
            apply_round_attack(
                original,
                [False, True, True, True],
                _cfg(mechanism),
                server_round=1,
                total_rounds=5,
                seed=2026,
            )


def test_heterogeneity_aware_mimicry_uses_nearest_benign_neighborhood():
    original = _alternate_layout_updates()
    attacked, _ = apply_round_attack(
        original,
        [False, False, False, True, True],
        _cfg(
            "heterogeneity_aware_mimicry",
            strength=2.5,
            neighbors=2,
            similarity="cosine",
            mimicry_lambda=0.8,
        ),
        server_round=1,
        total_rounds=5,
        seed=2026,
    )

    _assert_layout_and_finite(original, attacked)
    for idx in (0, 1, 2):
        for before, after in zip(original[idx], attacked[idx]):
            np.testing.assert_array_equal(after, before)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(original[3], attacked[3])
    )


def test_targeted_family_poisoning_uses_configured_metadata_not_hardcoded_family():
    original = _alternate_layout_updates()
    families = [
        ["Normal", "Threat-Z", "Threat-Z", "Other"],
        ["Normal", "Threat-Z", "Threat-Z", "Other"],
        ["Normal", "Threat-Z", "Threat-Z", "Other"],
        ["Normal", "Threat-Z", "Threat-Z", "Other"],
        ["Normal", "Threat-Z", "Threat-Z", "Other"],
    ]
    labels = [np.array([0, 1, 1, 1], dtype=np.int16) for _ in original]
    inferences = [
        np.array([[0.1], [0.9], [0.8], [0.7]], dtype=np.float32),
        np.array([[0.1], [0.85], [0.80], [0.7]], dtype=np.float32),
        np.array([[0.1], [0.88], [0.82], [0.7]], dtype=np.float32),
        # Malicious candidate 3 is deliberately worst on the configured family.
        np.array([[0.1], [0.20], [0.25], [0.7]], dtype=np.float32),
        np.array([[0.1], [0.65], [0.60], [0.7]], dtype=np.float32),
    ]

    attacked, _ = apply_round_attack(
        original,
        [False, False, False, True, True],
        _cfg(
            "targeted_family_poisoning",
            target_family="Threat-Z",
            target_amplification=2.0,
            target_mimicry_lambda=0.1,
        ),
        server_round=1,
        total_rounds=5,
        seed=2026,
        probe_inferences=inferences,
        probe_labels=labels,
        probe_families=families,
    )

    _assert_layout_and_finite(original, attacked)
    for idx in (0, 1, 2):
        for before, after in zip(original[idx], attacked[idx]):
            np.testing.assert_array_equal(after, before)
    for left, right in zip(attacked[3], attacked[4]):
        np.testing.assert_allclose(left, right)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(original[3], attacked[3])
    )


def test_targeted_family_poisoning_supports_multiclass_probe_probabilities():
    original = _alternate_layout_updates()
    labels = [np.array([0, 2, 2, 1], dtype=np.int16) for _ in original]
    families = [["Clean-X", "Rare-X", "Rare-X", "Other-X"] for _ in original]
    inferences = [
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.2, 0.7],
                [0.1, 0.2, 0.7],
                [0.1, 0.8, 0.1],
            ],
            dtype=np.float32,
        )
        for _ in original
    ]
    inferences[3] = inferences[3].copy()
    inferences[3][1:3, 2] = 0.10
    inferences[4] = inferences[4].copy()
    inferences[4][1:3, 2] = 0.50

    attacked, _ = apply_round_attack(
        original,
        [False, False, False, True, True],
        _cfg(
            "targeted_family_poisoning",
            target_family="Rare-X",
            target_amplification=1.5,
        ),
        server_round=1,
        total_rounds=5,
        seed=7,
        probe_inferences=inferences,
        probe_labels=labels,
        probe_families=families,
    )
    _assert_layout_and_finite(original, attacked)


def test_targeted_family_poisoning_fails_fast_for_missing_dataset_metadata():
    original = _alternate_layout_updates()
    labels = [np.array([0, 1], dtype=np.int16) for _ in original]
    families = [["Clean-X", "Other-X"] for _ in original]
    inferences = [
        np.array([[0.1], [0.8]], dtype=np.float32) for _ in original
    ]

    with pytest.raises(ValueError, match="Available probe families"):
        apply_round_attack(
            original,
            [False, False, False, True, True],
            _cfg(
                "targeted_family_poisoning",
                target_family="Missing-Family",
            ),
            server_round=1,
            total_rounds=5,
            seed=7,
            probe_inferences=inferences,
            probe_labels=labels,
            probe_families=families,
        )


def test_targeted_family_poisoning_supports_class_id_selector():
    original = _alternate_layout_updates()
    labels = [np.array([0, 2, 2, 1], dtype=np.int16) for _ in original]
    families = [["dynamic-a", "dynamic-b", "dynamic-b", "dynamic-c"] for _ in original]
    inferences = [
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.2, 0.7],
                [0.1, 0.2, 0.7],
                [0.1, 0.8, 0.1],
            ],
            dtype=np.float32,
        )
        for _ in original
    ]
    inferences[3] = inferences[3].copy()
    inferences[3][1:3, 2] = 0.10
    inferences[4] = inferences[4].copy()
    inferences[4][1:3, 2] = 0.50

    attacked, _ = apply_round_attack(
        original,
        [False, False, False, True, True],
        _cfg(
            "targeted_family_poisoning",
            target_family="class_id:2",
            target_amplification=1.5,
        ),
        server_round=1,
        total_rounds=5,
        seed=7,
        probe_inferences=inferences,
        probe_labels=labels,
        probe_families=families,
    )

    _assert_layout_and_finite(original, attacked)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(original[3], attacked[3])
    )
