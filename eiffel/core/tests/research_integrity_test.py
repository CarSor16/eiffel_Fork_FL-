"""Regression tests for probe isolation and fixed evaluation domains."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import h5py

from eiffel.core.client import EiffelClient
from eiffel.datasets.preprocessed_network import PreprocessedNetworkDataset
from eiffel.storage.round_store import RoundStore


def _dataset(labels, names):
    return PreprocessedNetworkDataset(
        X=pd.DataFrame({"x": np.arange(len(labels), dtype=np.float32)}),
        y=pd.Series(labels, dtype="int64"),
        m=pd.DataFrame({"ClassName": names, "ClassId": labels}),
        key="research-integrity", _default_target=["*"],
    )


def _client(monkeypatch, train, test, predictions):
    model = Mock()
    model.metrics_names = ["loss", "accuracy"]
    model.evaluate.return_value = [0.5, 1 / 3]
    model.predict.return_value = np.asarray(predictions)
    model.fit.return_value = SimpleNamespace(history={"accuracy": [0.5], "loss": [0.5]})
    model.get_weights.return_value = []
    requested = []

    def get(split):
        requested.append(split)
        return {"train": train, "test": test}[split]

    client = object.__new__(EiffelClient)
    client.model = model
    client.cid = "malicious:0"
    client.seed = 2026
    client.verbose = 0
    client.poison_ins = None
    client.is_malicious = True
    client.eval_fit = False
    client.data_holder = SimpleNamespace(get=SimpleNamespace(remote=get))
    monkeypatch.setattr("eiffel.core.client.ray.get", lambda obj: obj)
    return client, requested


def test_fit_probe_uses_local_training_data_without_reading_test(monkeypatch):
    train = _dataset([0, 1], ["A", "B"])
    test = _dataset([1], ["B"])
    client, requested = _client(monkeypatch, train, test, [[0.1, 0.9]])
    captured = []
    client._capture_probe_payload = lambda dataset, config: captured.append(dataset) or {}
    client.fit([], {"batch_size": 2, "num_epochs": 1, "capture_inference": True})
    assert captured == [train]
    assert requested == ["train"]


def test_fit_evaluation_cannot_replace_attack_probe_with_test_payload(monkeypatch):
    train = _dataset([0, 1], ["A", "B"])
    test = _dataset([1], ["B"])
    client, _ = _client(monkeypatch, train, test, [[0.1, 0.9]])
    client.eval_fit = True
    client._capture_probe_payload = lambda dataset, config: {"_eiffel_probabilities": "local-train"}
    configurations = []

    def evaluate(weights, config):
        configurations.append(config)
        return 0.5, 1, {"global": "{}"}

    client.evaluate = evaluate
    _, _, payload = client.fit([], {"batch_size": 2, "num_epochs": 1, "capture_inference": True})
    assert configurations[0]["capture_inference"] is False
    assert payload["_eiffel_probabilities"] == "local-train"
    assert payload["_eiffel_probe_source"] == "train"


def test_macro_f1_domain_does_not_change_when_absent_class_is_predicted(monkeypatch):
    test = _dataset([0, 1, 1], ["A", "B", "B"])
    client, _ = _client(monkeypatch, test, test, [[0, 0, 1], [0, 1, 0], [0, 0, 1]])
    _, _, payload = client.evaluate([], {"batch_size": 2})
    metrics = json.loads(payload["global"])
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["macro_f1_all_model_classes"] == pytest.approx(2 / 9)
    assert metrics["test_class_coverage"] == pytest.approx(2 / 3)
    assert json.loads(payload["unobserved_test_class_ids"]) == [2]


def test_binary_metrics_include_both_class_recalls_and_full_confusion_matrix(monkeypatch):
    test = _dataset([0, 0, 1, 1], ["Benign", "Benign", "Attack", "Attack"])
    client, _ = _client(monkeypatch, test, test, [[0.1], [0.9], [0.9], [0.9]])
    _, n, payload = client.evaluate([], {"batch_size": 2})
    metrics = json.loads(payload["global"])
    assert metrics["macro_class_recall"] == pytest.approx(0.75)
    assert metrics["min_class_recall"] == pytest.approx(0.5)
    assert json.loads(payload["Benign"])["recall"] == pytest.approx(0.5)
    assert json.loads(payload["Attack"])["precision"] == pytest.approx(2 / 3)
    matrix = np.asarray(json.loads(payload["confusion_matrix"]))
    assert matrix.tolist() == [[1, 1], [0, 2]]
    assert matrix.sum() == n


def test_training_and_evaluation_probes_are_persisted_separately(tmp_path):
    path = tmp_path / "separate_probes.h5"
    with RoundStore(path) as store:
        store.save_client(
            1, "malicious_0", submitted_update=[np.ones(1, dtype=np.float32)],
            probe_features=np.array([[1.], [2.]]), probe_labels=np.array([0, 1]),
            probe_families=["A", "B"], probe_split="train",
        )
        store.save_probe(
            "malicious_0", features=np.array([[9.]]), labels=np.array([1]),
            families=["B"],
        )
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5["fit_probe/clients/malicious_0/features"], [[1.], [2.]])
        np.testing.assert_array_equal(h5["probe/clients/malicious_0/features"], [[9.]])
        assert h5["fit_probe"].attrs["split"] == "train"
        assert h5["probe"].attrs["split"] == "test"
