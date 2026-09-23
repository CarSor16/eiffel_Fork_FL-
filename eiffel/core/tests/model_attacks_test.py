"""Tests for procedural model poisoning attacks and temporal schedules."""

import numpy as np

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
