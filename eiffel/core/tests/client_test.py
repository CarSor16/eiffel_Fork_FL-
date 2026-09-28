"""Tests for eiffel.core.client construction and compatibility."""

from functools import partial

import ray

from eiffel.core.client import EiffelClient, mk_client
from eiffel.datasets.dataset import DatasetHandle
from eiffel.datasets.synthetic_stress import load_data
from eiffel.models.advanced import mk_stress_mlp

SEED = 1138


def _synthetic_holder():
    dataset = load_data(
        seed=SEED,
        num_clients=1,
        samples_per_client=32,
        central_test_size=64,
        num_features=8,
        num_classes=2,
        latent_dim=4,
        informative_features=6,
        redundant_features=2,
        partition_mode="iid",
        train_label_noise=0.0,
    )
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

    return DatasetHandle.remote({"train": train, "test": test})


def _model_factory():
    return partial(
        mk_stress_mlp,
        hidden1=8,
        hidden2=4,
        learning_rate=0.001,
    )


def test_mk_client_accepts_historical_three_tuple_mapping():
    """Historical mappings remain usable and default to a benign role."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        num_cpus=1,
        num_gpus=0,
        local_mode=True,
        include_dashboard=False,
        ignore_reinit_error=True,
        logging_level="ERROR",
    )
    try:
        holder = _synthetic_holder()
        client = mk_client(
            cid="opaque_client",
            mappings={
                "opaque_client": (
                    holder,
                    None,
                    _model_factory(),
                )
            },
            seed=SEED,
        )

        assert isinstance(client, EiffelClient)
        assert client.cid == "opaque_client"
        assert client.poison_ins is None
        assert client.is_malicious is False
        assert len(client.get_parameters({})) > 0
    finally:
        ray.shutdown()


def test_mk_client_uses_explicit_malicious_role_without_poisoning():
    """Pure model attackers do not require data-poisoning instructions."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        num_cpus=1,
        num_gpus=0,
        local_mode=True,
        include_dashboard=False,
        ignore_reinit_error=True,
        logging_level="ERROR",
    )
    try:
        holder = _synthetic_holder()
        client = mk_client(
            cid="client_without_role_in_name",
            mappings={
                "client_without_role_in_name": (
                    holder,
                    None,
                    _model_factory(),
                    True,
                )
            },
            seed=SEED,
        )

        assert isinstance(client, EiffelClient)
        assert client.poison_ins is None
        assert client.is_malicious is True
    finally:
        ray.shutdown()
