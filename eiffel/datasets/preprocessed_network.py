"""Generic adapter for preprocessed network datasets.

The preprocessing kits used by this project already perform:
- leakage-safe train/validation/test splitting;
- imputation/scaling fitted on train only;
- fixed logical FL client assignment.

This module preserves those decisions inside Eiffel instead of repartitioning or
refitting preprocessing.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import ClassVar, Iterable

import numpy as np
import pandas as pd

from eiffel.datasets import Dataset
from eiffel.datasets.poisoning import PoisonOp


class PreprocessedNetworkDataset(Dataset):
    """Dataset wrapper for leakage-safe preprocessed network tasks."""

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
        """Apply reversible binary or explicit multiclass label flipping."""
        ratio = float(ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Poisoning ratio must be in [0, 1].")

        d = self.copy()
        if "ClassId" not in d.m.columns:
            raise ValueError("Preprocessed dataset metadata is missing ClassId.")
        if "Poisoned" not in d.m.columns:
            d.m["Poisoned"] = False

        original_ids = d.m["ClassId"].astype(int)
        poisoned = d.m["Poisoned"].astype(bool)

        explicit_multiclass = (
            source_class is not None or destination_class is not None
        )
        if explicit_multiclass:
            if source_class is None or destination_class is None:
                raise ValueError(
                    "source_class and destination_class must be configured together."
                )
            source_class = int(source_class)
            destination_class = int(destination_class)
            if source_class == destination_class:
                raise ValueError(
                    "source_class and destination_class must be different."
                )
            if source_class not in set(int(v) for v in original_ids.unique()):
                # Non-IID clients can legitimately have no samples of the source
                # class; this is an effective no-op, not a configuration failure.
                return 0

            target = original_ids == source_class
            replacement = destination_class
        else:
            unique = set(int(v) for v in original_ids.unique())
            if not unique.issubset({0, 1}):
                raise ValueError(
                    "Multiclass label flipping requires explicit "
                    "source_class/destination_class."
                )

            if target_classes is None:
                target = pd.Series(True, index=d.y.index)
            elif list(target_classes) == ["*"]:
                target = original_ids == 1
            else:
                class_column = (
                    "Attack"
                    if "Attack" in d.m.columns
                    else "ClassName"
                )
                target = d.m[class_column].astype(str).isin(
                    [str(value) for value in target_classes]
                )
            replacement = None

        base_count = int(target.sum())
        n = int(math.ceil(base_count * ratio))
        if op == PoisonOp.INC:
            candidates = target & ~poisoned
        elif op == PoisonOp.DEC:
            candidates = target & poisoned
        else:  # pragma: no cover - current PoisonOp has INC/DEC only
            raise ValueError(f"Unsupported poisoning operation: {op}")

        n = min(n, int(candidates.sum()))
        if n <= 0:
            return 0

        idx = d.y[candidates].sample(n=n, random_state=int(seed)).index
        if op == PoisonOp.INC:
            if explicit_multiclass:
                d.y.loc[idx] = int(replacement)
            else:
                d.y.loc[idx] = 1 - original_ids.loc[idx].to_numpy()
            d.m.loc[idx, "Poisoned"] = True
        else:
            d.y.loc[idx] = original_ids.loc[idx].to_numpy()
            d.m.loc[idx, "Poisoned"] = False

        self.X = d.X
        self.y = d.y.astype(np.int64)
        self.m = d.m
        return int(len(idx))


def _client_id_from_path(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot infer client id from filename: {path.name}")
    return int(match.group(1))


def _load_client_train(task_dir: Path, clients_subdir: str) -> pd.DataFrame:
    clients_dir = task_dir / clients_subdir
    paths = sorted(clients_dir.glob("client_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No preprocessed client_*.parquet files found in {clients_dir}"
        )

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        inferred = _client_id_from_path(path)
        if "client_id" not in frame.columns:
            frame = frame.copy()
            frame["client_id"] = inferred
        ids = set(int(v) for v in frame["client_id"].unique())
        if ids != {inferred}:
            raise ValueError(
                f"{path} contains client_id values {sorted(ids)}, expected {inferred}."
            )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _validate_target_mapping(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_name_column: str,
    target_id_column: str,
    expected_num_classes: int | None,
) -> None:
    for split_name, frame in (("train", train_df), ("test", test_df)):
        missing = {
            target_name_column,
            target_id_column,
        } - set(frame.columns)
        if missing:
            raise ValueError(
                f"{split_name} split is missing required target columns: "
                f"{sorted(missing)}"
            )

    mapping = train_df[
        [target_id_column, target_name_column]
    ].drop_duplicates()
    if (
        mapping[target_id_column].duplicated().any()
        or mapping[target_name_column].duplicated().any()
    ):
        raise ValueError("Train split contains an inconsistent id/name target mapping.")

    train_ids = sorted(int(v) for v in mapping[target_id_column].unique())
    if train_ids != list(range(len(train_ids))):
        raise ValueError(
            "Train class IDs must be contiguous from zero; got "
            f"{train_ids}."
        )
    if (
        expected_num_classes is not None
        and len(train_ids) != int(expected_num_classes)
    ):
        raise ValueError(
            "Class-count mismatch: expected "
            f"{expected_num_classes}, found {len(train_ids)}."
        )

    known = {
        (int(row[target_id_column]), str(row[target_name_column]))
        for _, row in mapping.iterrows()
    }
    test_pairs = {
        (int(row[target_id_column]), str(row[target_name_column]))
        for _, row in test_df[
            [target_id_column, target_name_column]
        ].drop_duplicates().iterrows()
    }
    unknown = test_pairs - known
    if unknown:
        raise ValueError(
            "Test split contains target mappings not present in train: "
            f"{sorted(unknown)}"
        )


def _feature_columns(
    train_df: pd.DataFrame,
    *,
    excluded: Iterable[str],
) -> list[str]:
    excluded_set = set(str(value) for value in excluded)
    columns = [
        column
        for column in train_df.columns
        if column not in excluded_set
        and pd.api.types.is_numeric_dtype(train_df[column])
    ]
    if not columns:
        raise ValueError("No numeric model feature columns were found.")
    return columns


def _check_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    split_name: str,
) -> None:
    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{split_name} split is missing model features: {missing}"
        )
    matrix = frame[feature_cols].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{split_name} model features contain NaN or infinite values."
        )


def _metadata(
    frame: pd.DataFrame,
    *,
    split: str,
    target_name_column: str,
    target_id_column: str,
    metadata_columns: Iterable[str],
    attack_metadata_column: str | None,
    require_client_id: bool,
) -> pd.DataFrame:
    available = [
        column
        for column in metadata_columns
        if column in frame.columns
    ]
    m = frame[available].copy()
    m["ClassName"] = frame[target_name_column].astype(str).to_numpy()
    m["ClassId"] = frame[target_id_column].astype(np.int64).to_numpy()
    if attack_metadata_column:
        if attack_metadata_column not in frame.columns:
            raise ValueError(
                "Configured attack_metadata_column is missing: "
                f"{attack_metadata_column}"
            )
        m["Attack"] = frame[attack_metadata_column].astype(str).to_numpy()
    m["Split"] = split

    if require_client_id:
        if "client_id" not in frame.columns:
            raise ValueError("Preassigned training rows require client_id.")
        m["ClientHint"] = frame["client_id"].astype(np.int64).to_numpy()
    else:
        m["ClientHint"] = -1
    return m


def load_preprocessed_data(
    task_dir: str | Path,
    *,
    seed: int,
    target_name_column: str,
    target_id_column: str,
    metadata_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    attack_metadata_column: str | None = None,
    test_file: str = "test_scaled.parquet",
    train_file: str = "train_scaled.parquet",
    clients_subdir: str = "clients",
    expected_num_clients: int | None = 10,
    expected_num_classes: int | None = None,
    **kwargs,
) -> PreprocessedNetworkDataset:
    """Load fixed preprocessed client shards and a common test split."""
    del seed  # Client assignment/splits were fixed during preprocessing.

    root = Path(task_dir)
    test_path = root / test_file
    if not test_path.is_file():
        raise FileNotFoundError(f"Preprocessed test file not found: {test_path}")

    train_df = _load_client_train(root, clients_subdir)
    test_df = pd.read_parquet(test_path)

    reference_train = root / train_file
    if reference_train.is_file():
        expected_rows = len(pd.read_parquet(reference_train, columns=[target_id_column]))
        if len(train_df) != expected_rows:
            raise ValueError(
                "Client shard rows do not reconstruct train_scaled.parquet: "
                f"{len(train_df)} != {expected_rows}."
            )

    client_ids = sorted(int(v) for v in train_df["client_id"].unique())
    if expected_num_clients is not None:
        expected_ids = list(range(int(expected_num_clients)))
        if client_ids != expected_ids:
            raise ValueError(
                "Preassigned client IDs mismatch: expected "
                f"{expected_ids}, got {client_ids}."
            )

    _validate_target_mapping(
        train_df,
        test_df,
        target_name_column=target_name_column,
        target_id_column=target_id_column,
        expected_num_classes=expected_num_classes,
    )

    metadata_columns = list(metadata_columns or [])
    excluded = set(exclude_columns or [])
    excluded.update(metadata_columns)
    excluded.update(
        {
            target_name_column,
            target_id_column,
            "client_id",
        }
    )
    feature_cols = _feature_columns(train_df, excluded=excluded)
    _check_features(train_df, feature_cols, split_name="train")
    _check_features(test_df, feature_cols, split_name="test")

    train_x = train_df[feature_cols].astype(np.float32)
    test_x = test_df[feature_cols].astype(np.float32)
    train_y = train_df[target_id_column].astype(np.int64)
    test_y = test_df[target_id_column].astype(np.int64)
    train_m = _metadata(
        train_df,
        split="train",
        target_name_column=target_name_column,
        target_id_column=target_id_column,
        metadata_columns=metadata_columns,
        attack_metadata_column=attack_metadata_column,
        require_client_id=True,
    )
    test_m = _metadata(
        test_df,
        split="test",
        target_name_column=target_name_column,
        target_id_column=target_id_column,
        metadata_columns=metadata_columns,
        attack_metadata_column=attack_metadata_column,
        require_client_id=False,
    )

    return PreprocessedNetworkDataset(
        X=pd.concat([train_x, test_x], ignore_index=True),
        y=pd.concat([train_y, test_y], ignore_index=True),
        m=pd.concat([train_m, test_m], ignore_index=True),
        **kwargs,
    )
