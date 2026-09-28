"""Tests for Eiffel binary label-flipping data poisoning."""

import numpy as np
import pandas as pd

from eiffel.datasets.nfv2 import NFV2Dataset
from eiffel.datasets.poisoning import (
    PoisonIns,
    PoisonOp,
    poisoning_fraction_at_round,
    poisoning_is_configured,
)


def _dataset():
    return NFV2Dataset(
        X=pd.DataFrame(
            {
                "f0": np.arange(6, dtype=np.float32),
                "f1": np.linspace(0.0, 1.0, 6, dtype=np.float32),
            }
        ),
        y=pd.Series([0, 1, 1, 1, 0, 1], name="Label", dtype=np.int64),
        m=pd.DataFrame(
            {
                "Attack": [
                    "Benign",
                    "Scan",
                    "Botnet",
                    "Botnet",
                    "Benign",
                    "DDoS",
                ]
            }
        ),
        key="unit",
        _default_target=["Botnet"],
    )


def test_targeted_label_flip_changes_only_target_family_and_is_reversible():
    dataset = _dataset()
    original = dataset.y.copy()

    changed = dataset.poison(
        1.0,
        PoisonOp.INC,
        seed=2026,
        target_classes=["Botnet"],
    )

    assert changed == 2
    botnet = dataset.m["Attack"] == "Botnet"
    np.testing.assert_array_equal(
        dataset.y.loc[botnet].to_numpy(),
        1 - original.loc[botnet].to_numpy(),
    )
    np.testing.assert_array_equal(
        dataset.y.loc[~botnet].to_numpy(),
        original.loc[~botnet].to_numpy(),
    )
    assert dataset.m.loc[botnet, "Poisoned"].all()
    assert not dataset.m.loc[~botnet, "Poisoned"].any()

    restored = dataset.poison(
        1.0,
        PoisonOp.DEC,
        seed=2026,
        target_classes=["Botnet"],
    )
    assert restored == 2
    np.testing.assert_array_equal(dataset.y.to_numpy(), original.to_numpy())


def test_untargeted_label_flip_can_flip_the_whole_local_dataset():
    dataset = _dataset()
    original = dataset.y.copy()

    changed = dataset.poison(
        1.0,
        PoisonOp.INC,
        seed=2026,
        target_classes=None,
    )

    assert changed == len(dataset)
    np.testing.assert_array_equal(
        dataset.y.to_numpy(),
        1 - original.to_numpy(),
    )
    assert dataset.m["Poisoned"].all()


def test_poisoning_profile_state_is_reconstructed_by_round():
    late = PoisonIns(
        profile="0.0+0.4{2}-0.4{4}",
        n_rounds=5,
        target=None,
    )
    assert poisoning_is_configured(late)
    assert [
        poisoning_fraction_at_round(late, round_number)
        for round_number in range(1, 6)
    ] == [0.0, 0.4, 0.4, 0.0, 0.0]

    gradual = PoisonIns(
        profile="0.0+0.2[2:4]",
        n_rounds=5,
        target=None,
    )
    np.testing.assert_allclose(
        [
            poisoning_fraction_at_round(gradual, round_number)
            for round_number in range(1, 6)
        ],
        [0.0, 0.2, 0.4, 0.6, 0.6],
    )

    clean = PoisonIns(profile="0.0", n_rounds=3, target=None)
    assert not poisoning_is_configured(clean)
    assert poisoning_fraction_at_round(clean, 3) == 0.0
