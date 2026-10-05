"""MIRAGE-GenAI-2025 dataset adapter for preprocessed FL experiments.

This loader consumes the Parquet files produced by the MIRAGE preprocessing kit.
Preprocessing (imputation/scaling and capture-aware splitting) is intentionally not
repeated here. Eiffel only selects model features, preserves the preassigned client
partition, and exposes a common held-out test set.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from eiffel.datasets import Dataset
from eiffel.datasets.poisoning import PoisonOp


FEATURE_PREFIX = "f__"
TARGET_ID = "target_id"
TARGET_NAME = "target_name"
CLIENT_ID = "client_id"


class MirageDataset(Dataset):
    """Dataset wrapper for multiclass MIRAGE traffic classification."""

    _stratify_column: ClassVar[str] = "ClassName"

    def poison(
        self,
        ratio: float,
        op: PoisonOp,
        *,
        seed: int,
        target_classes: list[str] | None = None,
        source_class: int | None = None,
        destination_class: int | None = None,
    ) -> int:
        """Apply an explicit source -> destination multiclass label flip.

        The semantic metadata (ClassName/ClassId) remains unchanged so the original
        traffic class is still available for auditing. Only the supervised label y is
        modified. This also makes scheduled decrement operations reversible.
        """
        del target_classes
        if source_class is None or destination_class is None:
            raise ValueError(
                "MIRAGE label flipping requires source_class and destination_class."
            )
        source_class = int(source_class)
        destination_class = int(destination_class)
        if source_class == destination_class:
            raise ValueError(
                "MIRAGE source_class and destination_class must be different."
            )
        if not 0.0 <= float(ratio) <= 1.0:
            raise ValueError("MIRAGE poisoning ratio must be in [0, 1].")

        d = self.copy()
        if "ClassId" not in d.m.columns:
            raise ValueError("MIRAGE metadata is missing ClassId.")
        known_ids = set(int(v) for v in d.m["ClassId"].unique())
        if source_class not in known_ids:
            # Non-IID clients can legitimately have no local samples of a class.
            return 0
        if destination_class < 0:
            raise ValueError("destination_class must be non-negative.")

        if "Poisoned" not in d.m.columns:
            d.m["Poisoned"] = False

        original_source = d.m["ClassId"].astype(int) == source_class
        n = int(np.ceil(int(original_source.sum()) * float(ratio)))
        if op == PoisonOp.INC:
            candidates = original_source & ~d.m["Poisoned"].astype(bool)
        elif op == PoisonOp.DEC:
            candidates = original_source & d.m["Poisoned"].astype(bool)
        else:
            raise ValueError(f"Unsupported poisoning operation: {op}")

        n = min(n, int(candidates.sum()))
        if n <= 0:
            return 0

        idx = d.y[candidates].sample(n=n, random_state=int(seed)).index
        if op == PoisonOp.INC:
            d.y.loc[idx] = destination_class
            d.m.loc[idx, "Poisoned"] = True
        else:
            d.y.loc[idx] = source_class
            d.m.loc[idx, "Poisoned"] = False

        self.X = d.X
        self.y = d.y.astype(np.int64)
        self.m = d.m
        return int(len(idx))


def _check_numeric_features(df: pd.DataFrame, feature_cols: list[str], path: Path) -> None:
    bad = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise TypeError(
            f"MIRAGE model features must be numeric; non-numeric columns in {path}: {bad}"
        )
    values = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(
            f"MIRAGE model features contain NaN or infinite values in {path}. "
            "Use the scaled/imputed preprocessing outputs."
        )


def _validate_targets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    expected_num_classes: int | None,
) -> None:
    for split_name, df in (("train", train_df), ("test", test_df)):
        missing = {TARGET_ID, TARGET_NAME} - set(df.columns)
        if missing:
            raise ValueError(
                f"MIRAGE {split_name} split is missing required columns: "
                f"{sorted(missing)}"
            )

    mapping = train_df[[TARGET_ID, TARGET_NAME]].drop_duplicates()
    if mapping[TARGET_ID].duplicated().any() or mapping[TARGET_NAME].duplicated().any():
        raise ValueError(
            "MIRAGE train split contains an inconsistent target_id/target_name mapping."
        )

    train_ids = sorted(int(v) for v in mapping[TARGET_ID].unique())
    if expected_num_classes is not None:
        expected_ids = list(range(int(expected_num_classes)))
        if train_ids != expected_ids:
            raise ValueError(
                "MIRAGE train class IDs do not match the expected contiguous mapping: "
                f"expected {expected_ids}, got {train_ids}."
            )

    train_pairs = {
        (int(row[TARGET_ID]), str(row[TARGET_NAME]))
        for _, row in mapping.iterrows()
    }
    test_pairs = {
        (int(row[TARGET_ID]), str(row[TARGET_NAME]))
        for _, row in test_df[[TARGET_ID, TARGET_NAME]].drop_duplicates().iterrows()
    }
    unknown = test_pairs - train_pairs
    if unknown:
        raise ValueError(
            "MIRAGE test split contains target mappings not present in train: "
            f"{sorted(unknown)}"
        )


def _metadata(
    df: pd.DataFrame,
    *,
    split_name: str,
    require_client_id: bool,
) -> pd.DataFrame:
    metadata_columns = [
        "row_id",
        "capture_id",
        "scenario",
        "device_id",
        "bf_labeling_type",
        "app_class",
        "activity",
        TARGET_NAME,
    ]
    available = [c for c in metadata_columns if c in df.columns]
    m = df[available].copy()
    m["ClassName"] = df[TARGET_NAME].astype(str).to_numpy()
    m["ClassId"] = df[TARGET_ID].astype(np.int64).to_numpy()
    m["Split"] = split_name

    if require_client_id:
        if CLIENT_ID not in df.columns:
            raise ValueError(
                "MIRAGE preassigned training data requires a client_id column. "
                "Use train_scaled_with_clients.parquet from the preprocessing kit."
            )
        m["ClientHint"] = df[CLIENT_ID].astype(np.int64).to_numpy()
    else:
        m["ClientHint"] = -1

    return m


def load_data(
    task_dir: str | Path,
    *,
    seed: int,
    train_file: str = "train_scaled_with_clients.parquet",
    test_file: str = "test_scaled.parquet",
    expected_num_clients: int | None = 10,
    expected_num_classes: int | None = 3,
    **kwargs,
) -> MirageDataset:
    """Load the preprocessed MIRAGE app-classification task.

    Only columns whose names start with f__ are exposed to the model. Identity,
    capture, scenario, application-name and target columns remain metadata and cannot
    become shortcut features.

    The train file must already contain client_id assignments. They are copied to
    the generic ClientHint metadata field used by Eiffel's PreassignedPartitioner.
    The test file is kept as a common held-out test set.
    """
    del seed

    task_path = Path(task_dir)
    train_path = task_path / train_file
    test_path = task_path / test_file
    if not train_path.is_file():
        raise FileNotFoundError(
            f"MIRAGE train file not found: {train_path}. "
            "Expected preprocessing output train_scaled_with_clients.parquet."
        )
    if not test_path.is_file():
        raise FileNotFoundError(f"MIRAGE test file not found: {test_path}")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    _validate_targets(
        train_df,
        test_df,
        expected_num_classes=expected_num_classes,
    )

    feature_cols = [c for c in train_df.columns if c.startswith(FEATURE_PREFIX)]
    if not feature_cols:
        raise ValueError(
            f"No MIRAGE model features found. Expected columns prefixed by {FEATURE_PREFIX!r}."
        )
    missing_test_features = [c for c in feature_cols if c not in test_df.columns]
    if missing_test_features:
        raise ValueError(
            "MIRAGE test split is missing train features: "
            f"{missing_test_features}"
        )

    extra_test_features = [
        c for c in test_df.columns
        if c.startswith(FEATURE_PREFIX) and c not in feature_cols
    ]
    if extra_test_features:
        raise ValueError(
            "MIRAGE test split contains feature columns absent from train: "
            f"{extra_test_features}"
        )

    _check_numeric_features(train_df, feature_cols, train_path)
    _check_numeric_features(test_df, feature_cols, test_path)

    client_ids = (
        sorted(int(v) for v in train_df[CLIENT_ID].unique())
        if CLIENT_ID in train_df
        else []
    )
    if expected_num_clients is not None and len(client_ids) != int(expected_num_clients):
        raise ValueError(
            "MIRAGE preassigned client count mismatch: "
            f"expected {expected_num_clients}, found {len(client_ids)} ({client_ids})."
        )

    train_x = train_df[feature_cols].astype(np.float32)
    test_x = test_df[feature_cols].astype(np.float32)
    train_y = train_df[TARGET_ID].astype(np.int64)
    test_y = test_df[TARGET_ID].astype(np.int64)
    train_m = _metadata(train_df, split_name="train", require_client_id=True)
    test_m = _metadata(test_df, split_name="test", require_client_id=False)

    X = pd.concat([train_x, test_x], ignore_index=True)
    y = pd.concat([train_y, test_y], ignore_index=True)
    m = pd.concat([train_m, test_m], ignore_index=True)

    return MirageDataset(X=X, y=y, m=m, **kwargs)
