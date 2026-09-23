"""Small end-to-end client smoke test for the synthetic FL-NIDS path."""

import json

import ray

from eiffel.core.client import EiffelClient
from eiffel.datasets.dataset import DatasetHandle
from eiffel.datasets.synthetic_stress import load_data
from eiffel.models.advanced import mk_stress_mlp


def _split_synthetic(dataset):
    train_mask = dataset.m["Split"] == "train"
    test_mask = dataset.m["Split"] == "test"

    train = dataset.copy()
    train.X = dataset.X.loc[train_mask].copy()
    train.y = dataset.y.loc[train_mask].copy()
    train.m = dataset.m.loc[train_mask].copy()

    test = dataset.copy()
    test.X = dataset.X.loc[test_mask].copy()
    test.y = dataset.y.loc[test_mask].copy()
    test.m = dataset.m.loc[test_mask].copy()
    return train, test


def test_synthetic_client_can_fit_and_evaluate_one_round():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        num_cpus=1,
        local_mode=True,
        include_dashboard=False,
        ignore_reinit_error=True,
        logging_level="ERROR",
    )
    try:
        dataset = load_data(
            seed=2026,
            num_clients=1,
            samples_per_client=128,
            central_test_size=256,
            num_features=8,
            num_classes=4,
            rare_class_id=3,
            rare_class_probability=0.05,
            latent_dim=4,
            informative_features=6,
            redundant_features=2,
            partition_mode="iid",
            train_label_noise=0.0,
        )
        train, test = _split_synthetic(dataset)
        holder = DatasetHandle.remote({"train": train, "test": test})
        model = mk_stress_mlp(
            8,
            hidden1=16,
            hidden2=8,
            learning_rate=0.001,
        )
        client = EiffelClient(
            "smoke_benign_0",
            holder,
            model,
            seed=2026,
        )

        parameters, examples, metrics = client.fit(
            model.get_weights(),
            {
                "batch_size": 32,
                "num_epochs": 1,
                "round": 1,
                "capture_inference": True,
                "probe_size": 32,
            },
        )

        assert examples == 128
        assert len(parameters) == len(model.get_weights())
        assert "_cid" in metrics
        assert "global" in metrics
        assert "_eiffel_inference" in metrics
        decoded = json.loads(metrics["global"])
        assert 0.0 <= float(decoded["accuracy"]) <= 1.0
        assert "macro_f1" in decoded
        assert "mcc" in decoded
    finally:
        ray.shutdown()
