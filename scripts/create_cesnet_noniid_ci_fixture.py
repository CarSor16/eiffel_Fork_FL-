"""Create a deterministic non-IID CESNET-shaped fixture for adversarial CI.

This is not real CESNET traffic. It stress-tests the fixed-client 50-class
pipeline under heterogeneous client distributions and makes classes 0/1
deliberately confusable so targeted poisoning has a measurable opportunity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _client_count(client_id: int, class_id: int) -> int:
    """Deterministic heterogeneous class counts with target specialists."""
    # Clients 0 and 1 are the default malicious logical clients for a 20% attack
    # fraction. Make them specialists for the targeted source class 0.
    if client_id in {0, 1} and class_id == 0:
        return 36
    if client_id in {0, 1} and class_id == 1:
        return 12

    # Every client has five strong local classes, emulating non-IID traffic domains.
    preferred = {(client_id * 5 + offset) % 50 for offset in range(5)}
    if class_id in preferred:
        return 14

    # Keep a small support for all classes so every local model remains trainable.
    return 2


def main() -> None:
    root = Path("data/cesnet")
    clients_dir = root / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)

    for old in clients_dir.glob("client_*.parquet"):
        old.unlink()

    rng = np.random.default_rng(20261005)
    n_clients = 10
    n_classes = 50
    n_features = 24
    test_per_class = 18

    centroids = rng.normal(0.0, 3.0, size=(n_classes, n_features)).astype(np.float32)
    # Create one deliberately difficult boundary: class 1 is close to class 0.
    centroids[1] = (
        centroids[0]
        + rng.normal(0.0, 0.35, size=n_features).astype(np.float32)
    )
    feature_names = [f"f_{idx:03d}" for idx in range(n_features)]

    all_train = []
    class_counts: dict[str, dict[str, int]] = {}

    for client_id in range(n_clients):
        parts = []
        client_shift = rng.normal(0.0, 0.45, size=n_features).astype(np.float32)
        class_counts[str(client_id)] = {}

        for class_id in range(n_classes):
            count = _client_count(client_id, class_id)
            class_counts[str(client_id)][str(class_id)] = count
            noise_std = 0.85 if class_id in {0, 1} else 0.65
            x = (
                centroids[class_id]
                + client_shift
                + rng.normal(0.0, noise_std, size=(count, n_features))
            ).astype(np.float32)

            frame = pd.DataFrame(x, columns=feature_names)
            frame["source_day"] = f"train-domain-{client_id:02d}"
            frame["target_name"] = f"domain_{class_id:02d}.example"
            frame["target_id"] = class_id
            frame["client_id"] = client_id
            parts.append(frame)

        client = pd.concat(parts, ignore_index=True)
        client = client.sample(
            frac=1.0,
            random_state=20261005 + client_id,
        ).reset_index(drop=True)
        client.to_parquet(
            clients_dir / f"client_{client_id:02d}.parquet",
            index=False,
        )
        all_train.append(client.drop(columns=["client_id"]))

    train = pd.concat(all_train, ignore_index=True)
    train.to_parquet(root / "train_scaled.parquet", index=False)

    validation = (
        train.groupby("target_id", group_keys=False)
        .head(3)
        .reset_index(drop=True)
    )
    validation.to_parquet(root / "validation_scaled.parquet", index=False)

    test_parts = []
    for class_id in range(n_classes):
        noise_std = 0.85 if class_id in {0, 1} else 0.65
        x = (
            centroids[class_id]
            + rng.normal(0.0, noise_std, size=(test_per_class, n_features))
        ).astype(np.float32)
        frame = pd.DataFrame(x, columns=feature_names)
        frame["source_day"] = "test-domain"
        frame["target_name"] = f"domain_{class_id:02d}.example"
        frame["target_id"] = class_id
        test_parts.append(frame)

    test = pd.concat(test_parts, ignore_index=True)
    test = test.sample(frac=1.0, random_state=20261005).reset_index(drop=True)
    test.to_parquet(root / "test_scaled.parquet", index=False)

    manifest = {
        "fixture": True,
        "fixture_mode": "noniid-hard",
        "seed": 20261005,
        "n_clients": n_clients,
        "n_classes": n_classes,
        "n_features": n_features,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "default_malicious_ids": [0, 1],
        "target_source_class": 0,
        "target_destination_class": 1,
        "class_mapping": {
            str(idx): f"domain_{idx:02d}.example" for idx in range(n_classes)
        },
        "client_class_counts": class_counts,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "client_id": client_id,
                "class_id": class_id,
                "count": count,
            }
            for client_id, counts in class_counts.items()
            for class_id, count in counts.items()
        ]
    ).to_csv(root / "class_distribution.csv", index=False)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
