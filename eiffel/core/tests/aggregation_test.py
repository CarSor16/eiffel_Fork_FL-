"""Unit tests for pluggable server-side aggregation backends."""

import numpy as np
import pytest

from eiffel.strategy.aggregation import (
    aggregate_updates,
    coordinate_median,
    fedavg,
    krum,
    krum_scores,
    multi_krum,
    trimmed_mean,
)


def _updates(values):
    return [
        [
            np.asarray([value, value + 1.0], dtype=np.float32),
            np.asarray([[value]], dtype=np.float32),
        ]
        for value in values
    ]


def test_fedavg_matches_sample_weighted_mean():
    updates = _updates([0.0, 2.0])
    result = fedavg(updates, [1, 3])
    np.testing.assert_allclose(result[0], [1.5, 2.5])
    np.testing.assert_allclose(result[1], [[1.5]])
    assert all(layer.dtype == np.float32 for layer in result)


def test_coordinate_median_rejects_extreme_outlier():
    result = coordinate_median(_updates([0.0, 0.1, 0.2, 100.0, 0.15]))
    np.testing.assert_allclose(result[0], [0.15, 1.15], atol=1e-6)
    np.testing.assert_allclose(result[1], [[0.15]], atol=1e-6)


def test_trimmed_mean_removes_symmetric_extremes():
    result = trimmed_mean(
        _updates([-100.0, 0.0, 1.0, 2.0, 100.0]),
        trim_ratio=0.2,
    )
    np.testing.assert_allclose(result[0], [1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(result[1], [[1.0]], atol=1e-6)


def test_trimmed_mean_zero_trim_is_plain_unweighted_mean():
    updates = _updates([0.0, 2.0, 4.0])
    result = trimmed_mean(updates, trim_ratio=0.1)
    np.testing.assert_allclose(result[0], [2.0, 3.0])


def test_krum_selects_member_of_tight_benign_cluster():
    updates = _updates([0.0, 0.05, 0.10, 0.15, 50.0])
    scores = krum_scores(updates, num_byzantine=1)
    assert int(np.argmin(scores)) in {0, 1, 2, 3}
    result = krum(updates, num_byzantine=1)
    assert float(result[0][0]) < 1.0


def test_multi_krum_averages_low_score_cluster():
    updates = _updates([0.0, 0.05, 0.10, 0.15, 50.0])
    result = multi_krum(
        updates,
        num_byzantine=1,
        num_selected=2,
    )
    assert float(result[0][0]) < 1.0
    assert float(result[1][0, 0]) < 1.0


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("fedavg", {}),
        ("median", {}),
        ("trimmed_mean", {"trim_ratio": 0.2}),
        ("krum", {"num_byzantine": 1}),
        ("multi_krum", {"num_byzantine": 1, "num_selected": 2}),
    ],
)
def test_dispatcher_preserves_model_layout(name, extra):
    updates = _updates([0.0, 0.1, 0.2, 0.3, 10.0])
    result = aggregate_updates(
        updates,
        num_examples=[10, 20, 30, 40, 50],
        config={"name": name, **extra},
    )
    assert [layer.shape for layer in result] == [(2,), (1, 1)]
    assert all(layer.dtype == np.float32 for layer in result)
    assert all(np.all(np.isfinite(layer)) for layer in result)


def test_krum_rejects_invalid_byzantine_bound():
    with pytest.raises(ValueError, match="2 \* num_byzantine \+ 3"):
        krum(_updates([0.0, 0.1, 0.2, 0.3]), num_byzantine=1)


@pytest.mark.parametrize("ratio", [-0.1, 0.5, 1.0])
def test_trimmed_mean_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="trim_ratio"):
        trimmed_mean(_updates([0.0, 1.0, 2.0]), trim_ratio=ratio)


def test_aggregation_rejects_non_finite_updates():
    updates = _updates([0.0, 1.0, 2.0])
    updates[1][0][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        coordinate_median(updates)
