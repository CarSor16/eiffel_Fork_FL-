"""Regression tests for Eiffel client resource allocation."""

import pytest

import eiffel.core.experiment as experiment


def test_compute_client_resources_respects_two_client_cap(monkeypatch):
    monkeypatch.setattr(experiment.psutil, "cpu_count", lambda: 6)
    monkeypatch.setattr(
        experiment.tf.config,
        "list_physical_devices",
        lambda kind: [],
    )

    resources = experiment.compute_client_resources(2)

    assert resources["num_cpus"] == pytest.approx(3.0)
    # A six-CPU Ray node can schedule exactly two actors requiring 3 CPUs each,
    # but never a third one.
    assert 2 * resources["num_cpus"] <= 6
    assert 3 * resources["num_cpus"] > 6
    assert resources["num_gpus"] == 0.0
