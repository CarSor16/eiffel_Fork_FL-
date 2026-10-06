"""Tests for modular FL defenses."""

import numpy as np
import pytest

from eiffel.strategy.defense import (
    apply_update_defense,
    canonical_defense_name,
    distill_global_weights,
    norm_clip_updates,
    update_l2_norm,
)
from eiffel.models.advanced import mk_stress_mlp


def _updates():
    return [
        [np.asarray([3.0, 4.0], dtype=np.float32)],
        [np.asarray([0.3, 0.4], dtype=np.float32)],
    ]


def test_norm_clipping_enforces_global_update_bound():
    clipped = norm_clip_updates(_updates(), max_norm=1.0)
    assert update_l2_norm(clipped[0]) == pytest.approx(1.0, rel=1e-6)
    assert update_l2_norm(clipped[1]) == pytest.approx(0.5, rel=1e-6)


def test_none_defense_preserves_updates():
    original = _updates()
    defended = apply_update_defense(original, {"name": "none"})
    for before, after in zip(original, defended):
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)
            assert left is not right


def test_probe_distillation_does_not_pre_transform_updates():
    original = _updates()
    defended = apply_update_defense(original, {"name": "probe_distillation"})
    for before, after in zip(original, defended):
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("none", "none"),
        ("clip", "norm_clipping"),
        ("clipping", "norm_clipping"),
        ("distillation", "probe_distillation"),
        ("knowledge_distillation", "probe_distillation"),
    ],
)
def test_defense_aliases(alias, canonical):
    assert canonical_defense_name(alias) == canonical


def test_probe_distillation_changes_global_weights_on_shared_probe():
    model = mk_stress_mlp(
        4,
        hidden1=8,
        hidden2=4,
        weight_decay=0.0,
        task="binary",
        num_classes=2,
    )
    initial = [np.asarray(value, dtype=np.float32) for value in model.get_weights()]
    features = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.2, 0.1, 0.0, -0.1],
            [1.0, 0.9, 0.8, 0.7],
            [0.8, 0.9, 1.0, 1.1],
        ],
        dtype=np.float32,
    )
    client_logits = [
        np.asarray([[-2.0], [-1.5], [2.0], [1.5]], dtype=np.float32),
        np.asarray([[-1.8], [-1.3], [1.8], [1.4]], dtype=np.float32),
        np.asarray([[-2.2], [-1.7], [2.2], [1.6]], dtype=np.float32),
    ]
    refined = distill_global_weights(
        initial,
        model_fn=lambda n: mk_stress_mlp(
            n,
            hidden1=8,
            hidden2=4,
            weight_decay=0.0,
            task="binary",
            num_classes=2,
        ),
        probe_features=[features, features.copy(), features.copy()],
        probabilities=[None, None, None],
        logits=client_logits,
        config={
            "name": "probe_distillation",
            "temperature": 2.0,
            "learning_rate": 1e-3,
            "alpha": 1.0,
            "epochs": 2,
        },
        seed=2026,
    )
    assert [value.shape for value in refined] == [value.shape for value in initial]
    assert any(
        not np.allclose(before, after)
        for before, after in zip(initial, refined)
    )


def test_probe_distillation_rejects_mismatched_client_probes():
    model = mk_stress_mlp(
        4,
        hidden1=8,
        hidden2=4,
        weight_decay=0.0,
        task="binary",
        num_classes=2,
    )
    initial = model.get_weights()
    a = np.zeros((4, 4), dtype=np.float32)
    b = a.copy()
    b[0, 0] = 1.0
    logits = [np.zeros((4, 1), dtype=np.float32)] * 2
    with pytest.raises(ValueError, match="same deterministic probe"):
        distill_global_weights(
            initial,
            model_fn=lambda n: mk_stress_mlp(
                n,
                hidden1=8,
                hidden2=4,
                weight_decay=0.0,
                task="binary",
                num_classes=2,
            ),
            probe_features=[a, b],
            probabilities=[None, None],
            logits=logits,
            config={"temperature": 2.0, "learning_rate": 1e-3, "epochs": 1},
            seed=7,
        )
