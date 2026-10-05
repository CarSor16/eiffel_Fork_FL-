"""Create a deterministic CESNET-shaped preprocessed fixture for CI.

This is not real CESNET traffic. It only validates the full fixed-client,
50-class Eiffel pipeline and attack/analysis stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path("data/cesnet")
    clients_dir = root / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)
    n_clients = 10
    n_classes = 50
    n_features = 24
    train_per_class_per_client = 6
    test_per_class = 12

    # Well-separated class centroids plus a small client-specific covariate shift.
    centroids = rng.normal(0.0, 3.0, size=(n_classes, n_features)).astype(np.float32)
    feature_names = [f"f_{idx:03d}" for idx in range(n_features)]

    all_train = []
    for client_id in range(n_clients):
        parts = []
        client_shift = rng.normal(0.0, 0.20, size=n_features).astype(np.float32)
        for class_id in range(n_classes):
            x = (
                centroids[class_id]
                + client_shift
                + rng.normal(
                    0.0,
                    0.45,
                    size=(train_per_class_per_client, n_features),
                )
            ).astype(np.float32)
            frame = pd.DataFrame(x, columns=feature_names)
            frame["source_day"] = f"train-day-{client_id:02d}"
            frame["target_name"] = f"domain_{class_id:02d}.example"
            frame["target_id"] = class_id
            frame["client_id"] = client_id
            parts.append(frame)
        client = pd.concat(parts, ignore_index=True)
        client = client.sample(frac=1.0, random_state=2026 + client_id).reset_index(drop=True)
        client.to_parquet(
            clients_dir / f"client_{client_id:02d}.parquet",
            index=False,
        )
        all_train.append(client.drop(columns=["client_id"]))

    train = pd.concat(all_train, ignore_index=True)
    train.to_parquet(root / "train_scaled.parquet", index=False)

    # Validation is present to match the real preprocessing contract but is not
    # consumed by federated training.
    validation = train.groupby("target_id", group_keys=False).head(2).reset_index(drop=True)
    validation.to_parquet(root / "validation_scaled.parquet", index=False)

    test_parts = []
    for class_id in range(n_classes):
        x = (
            centroids[class_id]
            + rng.normal(0.0, 0.45, size=(test_per_class, n_features))
        ).astype(np.float32)
        frame = pd.DataFrame(x, columns=feature_names)
        frame["source_day"] = "test-day"
        frame["target_name"] = f"domain_{class_id:02d}.example"
        frame["target_id"] = class_id
        test_parts.append(frame)
    test = pd.concat(test_parts, ignore_index=True)
    test = test.sample(frac=1.0, random_state=2026).reset_index(drop=True)
    test.to_parquet(root / "test_scaled.parquet", index=False)

    manifest = {
        "fixture": True,
        "seed": 2026,
        "n_clients": n_clients,
        "n_classes": n_classes,
        "n_features": n_features,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "class_mapping": {
            str(idx): f"domain_{idx:02d}.example" for idx in range(n_classes)
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
