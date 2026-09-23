"""Tests for the Eiffel-compatible synthetic stress dataset."""

from eiffel.datasets.synthetic_stress import CLASS_NAMES, load_data
from eiffel.datasets.partitioners import PreassignedPartitioner


def test_synthetic_stress_sizes_and_preassigned_shards():
    dataset = load_data(
        seed=2026,
        num_clients=2,
        samples_per_client=20,
        central_test_size=10,
        num_features=8,
        num_classes=4,
        latent_dim=4,
        informative_features=4,
        redundant_features=2,
        dirichlet_alpha=0.5,
    )

    assert len(dataset) == 50
    assert (dataset.m["Split"] == "train").sum() == 40
    assert (dataset.m["Split"] == "test").sum() == 10
    assert set(dataset.y.unique()).issubset({0, 1})

    train_mask = dataset.m["Split"] == "train"
    train = dataset.copy()
    train.X = dataset.X.loc[train_mask].copy()
    train.y = dataset.y.loc[train_mask].copy()
    train.m = dataset.m.loc[train_mask].copy()

    partitioner = PreassignedPartitioner(n_partitions=2, seed=2026)
    partitioner.load(train)
    shards = partitioner.all()

    assert len(shards) == 2
    assert [len(shard) for shard in shards] == [20, 20]



def test_synthetic_stress_multiclass_targets_keep_family_ids():
    dataset = load_data(
        seed=2026,
        num_clients=3,
        samples_per_client=100,
        central_test_size=60,
        num_features=8,
        num_classes=4,
        latent_dim=4,
        informative_features=4,
        redundant_features=2,
        dirichlet_alpha=0.5,
        task="multiclass",
    )

    labels = set(int(value) for value in dataset.y.unique())
    assert min(labels) >= 0
    assert max(labels) < 4
    assert len(labels) >= 3

    # Label noise can introduce a small number of mismatches in train data, but the
    # held-out test split remains an exact family-name -> family-id target.
    test_mask = dataset.m["Split"] == "test"
    test_pairs = set(
        zip(
            dataset.m.loc[test_mask, "Attack"].astype(str),
            dataset.y.loc[test_mask].astype(int),
        )
    )
    assert len({family for family, _ in test_pairs}) >= 3
    for family, class_id in test_pairs:
        assert class_id == CLASS_NAMES.index(family)
