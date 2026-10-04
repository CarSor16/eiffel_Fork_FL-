"""Tests for generic leakage-safe preprocessed network datasets."""

from pathlib import Path

import pandas as pd

from eiffel.datasets.poisoning import PoisonOp
from eiffel.datasets.preprocessed_network import (
    PreprocessedNetworkDataset,
    load_preprocessed_data,
)


def _write_fixture(root: Path) -> None:
    clients = root / "clients"
    clients.mkdir(parents=True)

    rows = []
    for client_id in range(2):
        frame = pd.DataFrame(
            {
                "feature_a": [0.1 + client_id, 0.2 + client_id],
                "feature_b": [0.3, 0.4],
                "family_target": ["Benign", "DDoS"],
                "family_target_id": [0, 1],
                "binary_target": ["Benign", "Attack"],
                "binary_target_id": [0, 1],
                "fine_target": ["BenignTraffic", "DDoS-SYN_Flood"],
                "fine_target_id": [0, 1],
                "source_file": [f"client_{client_id}.csv"] * 2,
                "client_id": [client_id] * 2,
            }
        )
        frame.to_parquet(clients / f"client_{client_id:02d}.parquet", index=False)
        rows.append(frame.drop(columns=["client_id"]))

    pd.concat(rows, ignore_index=True).to_parquet(
        root / "train_scaled.parquet", index=False
    )
    pd.DataFrame(
        {
            "feature_a": [0.5, 0.6],
            "feature_b": [0.7, 0.8],
            "family_target": ["Benign", "DDoS"],
            "family_target_id": [0, 1],
            "binary_target": ["Benign", "Attack"],
            "binary_target_id": [0, 1],
            "fine_target": ["BenignTraffic", "DDoS-SYN_Flood"],
            "fine_target_id": [0, 1],
            "source_file": ["test_a.csv", "test_b.csv"],
        }
    ).to_parquet(root / "test_scaled.parquet", index=False)


def test_preprocessed_loader_preserves_clients_and_excludes_target_ids(tmp_path):
    _write_fixture(tmp_path)

    dataset = load_preprocessed_data(
        tmp_path,
        seed=2026,
        target_name_column="family_target",
        target_id_column="family_target_id",
        metadata_columns=[
            "source_file",
            "family_target",
            "family_target_id",
            "binary_target",
            "binary_target_id",
            "fine_target",
            "fine_target_id",
        ],
        exclude_columns=[
            "source_file",
            "family_target",
            "family_target_id",
            "binary_target",
            "binary_target_id",
            "fine_target",
            "fine_target_id",
        ],
        expected_num_clients=2,
        expected_num_classes=2,
        key="fixture",
        _default_target=["DDoS"],
    )

    assert isinstance(dataset, PreprocessedNetworkDataset)
    assert list(dataset.X.columns) == ["feature_a", "feature_b"]
    train = dataset.m["Split"] == "train"
    test = dataset.m["Split"] == "test"
    assert set(dataset.m.loc[train, "ClientHint"]) == {0, 1}
    assert set(dataset.m.loc[test, "ClientHint"]) == {-1}
    assert set(dataset.m["ClassName"]) == {"Benign", "DDoS"}


def test_preprocessed_binary_target_keeps_family_attack_metadata(tmp_path):
    _write_fixture(tmp_path)

    dataset = load_preprocessed_data(
        tmp_path,
        seed=2026,
        target_name_column="binary_target",
        target_id_column="binary_target_id",
        attack_metadata_column="family_target",
        metadata_columns=["family_target", "binary_target"],
        exclude_columns=[
            "source_file",
            "family_target",
            "family_target_id",
            "binary_target",
            "binary_target_id",
            "fine_target",
            "fine_target_id",
        ],
        expected_num_clients=2,
        expected_num_classes=2,
        key="binary-fixture",
        _default_target=["DDoS"],
    )

    assert "Attack" in dataset.m
    assert set(dataset.m["Attack"]) == {"Benign", "DDoS"}


def test_preprocessed_multiclass_label_flip_is_reversible():
    dataset = PreprocessedNetworkDataset(
        X=pd.DataFrame({"x": [0.1, 0.2, 0.3]}),
        y=pd.Series([0, 1, 1], dtype="int64"),
        m=pd.DataFrame(
            {
                "ClassName": ["A", "B", "B"],
                "ClassId": [0, 1, 1],
            }
        ),
        key="multiclass",
        _default_target=["*"],
    )

    changed = dataset.poison(
        1.0,
        PoisonOp.INC,
        seed=2026,
        source_class=1,
        destination_class=0,
    )
    assert changed == 2
    assert dataset.y.tolist() == [0, 0, 0]

    restored = dataset.poison(
        1.0,
        PoisonOp.DEC,
        seed=2026,
        source_class=1,
        destination_class=0,
    )
    assert restored == 2
    assert dataset.y.tolist() == [0, 1, 1]
