"""Tests for quantitative client-heterogeneity analysis."""

import math

import pandas as pd

from eiffel.analysis.client_heterogeneity import analyse_distribution


def test_heterogeneity_metrics_distinguish_iid_and_specialists():
    iid = pd.DataFrame(
        [
            {"client_id": client, "class_id": cls, "count": 10}
            for client in range(3)
            for cls in range(3)
        ]
    )
    _, _, iid_summary = analyse_distribution(iid)

    noniid = pd.DataFrame(
        [
            {"client_id": 0, "class_id": 0, "count": 90},
            {"client_id": 0, "class_id": 1, "count": 5},
            {"client_id": 0, "class_id": 2, "count": 5},
            {"client_id": 1, "class_id": 0, "count": 5},
            {"client_id": 1, "class_id": 1, "count": 90},
            {"client_id": 1, "class_id": 2, "count": 5},
            {"client_id": 2, "class_id": 0, "count": 5},
            {"client_id": 2, "class_id": 1, "count": 5},
            {"client_id": 2, "class_id": 2, "count": 90},
        ]
    )
    _, _, noniid_summary = analyse_distribution(noniid)

    assert math.isclose(iid_summary["mean_pairwise_js_divergence"], 0.0, abs_tol=1e-12)
    assert iid_summary["mean_client_normalized_entropy"] > 0.99
    assert noniid_summary["mean_pairwise_js_divergence"] > 0.3
    assert noniid_summary["mean_client_normalized_entropy"] < 0.5
    assert noniid_summary["mean_class_client_count_cv"] > iid_summary[
        "mean_class_client_count_cv"
    ]


def test_target_concentration_is_reported_for_malicious_clients():
    frame = pd.DataFrame(
        [
            {"client_id": 0, "class_id": 0, "count": 40},
            {"client_id": 0, "class_id": 1, "count": 10},
            {"client_id": 1, "class_id": 0, "count": 40},
            {"client_id": 1, "class_id": 1, "count": 10},
            {"client_id": 2, "class_id": 0, "count": 20},
            {"client_id": 2, "class_id": 1, "count": 30},
        ]
    )

    _, _, summary = analyse_distribution(
        frame,
        target_class=0,
        malicious_ids=[0, 1],
    )

    assert summary["target_total_samples"] == 100
    assert summary["target_malicious_samples"] == 80
    assert math.isclose(summary["target_malicious_concentration"], 0.8)
