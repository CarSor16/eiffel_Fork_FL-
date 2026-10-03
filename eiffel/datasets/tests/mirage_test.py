"""Tests for the preprocessed MIRAGE dataset adapter."""

import pandas as pd

from eiffel.datasets.mirage import MirageDataset, load_data
from eiffel.datasets.poisoning import PoisonOp


def test_mirage_loader_preserves_features_split_and_client_hints(tmp_path):
    task_dir = tmp_path / "app_3class"
    task_dir.mkdir()

    train = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "capture_id": ["a", "b", "c", "d"],
            "target_name": ["ChatGPT", "Copilot", "Gemini", "ChatGPT"],
            "target_id": [0, 1, 2, 0],
            "client_id": [0, 0, 1, 1],
            "f__a": [0.1, 0.2, 0.3, 0.4],
            "f__b": [0.4, 0.3, 0.2, 0.1],
        }
    )
    test = pd.DataFrame(
        {
            "row_id": [5, 6, 7],
            "capture_id": ["e", "f", "g"],
            "target_name": ["ChatGPT", "Copilot", "Gemini"],
            "target_id": [0, 1, 2],
            "f__a": [0.5, 0.6, 0.7],
            "f__b": [0.7, 0.6, 0.5],
        }
    )
    train.to_parquet(task_dir / "train_scaled_with_clients.parquet", index=False)
    test.to_parquet(task_dir / "test_scaled.parquet", index=False)

    dataset = load_data(
        task_dir,
        seed=2026,
        expected_num_clients=2,
        expected_num_classes=3,
        key="mirage-test",
        _default_target=["ChatGPT"],
    )

    assert isinstance(dataset, MirageDataset)
    assert list(dataset.X.columns) == ["f__a", "f__b"]
    assert len(dataset) == 7
    assert set(dataset.m["Split"]) == {"train", "test"}
    assert set(dataset.m.loc[dataset.m["Split"] == "train", "ClientHint"]) == {0, 1}
    assert set(dataset.m.loc[dataset.m["Split"] == "test", "ClientHint"]) == {-1}
    assert set(dataset.m["ClassName"]) == {"ChatGPT", "Copilot", "Gemini"}
    assert "target_id" not in dataset.X
    assert "capture_id" not in dataset.X


def test_mirage_multiclass_label_flip_is_explicit_and_reversible():
    dataset = MirageDataset(
        X=pd.DataFrame({"f__a": [0.1, 0.2, 0.3, 0.4]}),
        y=pd.Series([2, 2, 1, 0], dtype="int64"),
        m=pd.DataFrame(
            {
                "ClassName": ["Gemini", "Gemini", "Copilot", "ChatGPT"],
                "ClassId": [2, 2, 1, 0],
            }
        ),
        key="mirage-test",
        _default_target=["ChatGPT"],
    )

    changed = dataset.poison(
        1.0,
        PoisonOp.INC,
        seed=2026,
        source_class=2,
        destination_class=1,
    )
    assert changed == 2
    assert dataset.y.tolist() == [1, 1, 1, 0]
    assert dataset.m["ClassId"].tolist() == [2, 2, 1, 0]
    assert dataset.m["Poisoned"].tolist() == [True, True, False, False]

    restored = dataset.poison(
        1.0,
        PoisonOp.DEC,
        seed=2026,
        source_class=2,
        destination_class=1,
    )
    assert restored == 2
    assert dataset.y.tolist() == [2, 2, 1, 0]
    assert not dataset.m["Poisoned"].any()
