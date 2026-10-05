"""Create a calibrated non-IID CESNET-shaped fixture for baseline validation.

This is synthetic CESNET-shaped data, not real CESNET traffic. Unlike the harder
stress fixture, every class has enough global support for the clean model to learn
it, while clients remain strongly heterogeneous and clients 0/1 specialize in the
target source class used by targeted poisoning tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _client_count(client_id: int, class_id: int) -> int:
    if client_id in {0, 1} and class_id == 0:
        return 42
    if client_id in {0, 1} and class_id == 1:
        return 18

    preferred = {(client_id * 5 + offset) % 50 for offset in range(5)}
    if class_id in preferred:
        return 16

    return 6


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
    test_per_class = 20

    centroids = rng.normal(0.0, 3.1, size=(n_classes, n_features)).astype(np.float32)
    # Deliberately make the targeted destination class nearby, but not so close that
    # the clean model cannot separate them.
    centroids[1] = (
        centroids[0]
        + rng.normal(0.0, 0.55, size=n_features).astype(np.float32)
    )
    feature_names = [f"f_{idx:03d}" for idx in range(n_features)]

    all_train = []
    distribution = []
    for client_id in range(n_clients):
        client_shift = rng.normal(0.0, 0.30, size=n_features).astype(np.float32)
        parts = []
        for class_id in range(n_classes):
            count = _client_count(client_id, class_id)
            noise_std = 0.72 if class_id in {0, 1} else 0.55
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
            distribution.append(
                {"client_id": client_id, "class_id": class_id, "count": count}
            )

        client = pd.concat(parts, ignore_index=True).sample(
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
        .head(5)
        .reset_index(drop=True)
    )
    validation.to_parquet(root / "validation_scaled.parquet", index=False)

    test_parts = []
    for class_id in range(n_classes):
        noise_std = 0.72 if class_id in {0, 1} else 0.55
        x = (
            centroids[class_id]
            + rng.normal(0.0, noise_std, size=(test_per_class, n_features))
        ).astype(np.float32)
        frame = pd.DataFrame(x, columns=feature_names)
        frame["source_day"] = "test-domain"
        frame["target_name"] = f"domain_{class_id:02d}.example"
        frame["target_id"] = class_id
        test_parts.append(frame)

    test = pd.concat(test_parts, ignore_index=True).sample(
        frac=1.0,
        random_state=20261005,
    ).reset_index(drop=True)
    test.to_parquet(root / "test_scaled.parquet", index=False)

    dist = pd.DataFrame(distribution)
    dist.to_csv(root / "class_distribution.csv", index=False)

    totals = dist.groupby("class_id")["count"].sum().to_dict()
    manifest = {
        "fixture": True,
        "fixture_mode": "noniid-calibrated",
        "seed": 20261005,
        "n_clients": n_clients,
        "n_classes": n_classes,
        "n_features": n_features,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "default_malicious_ids": [0, 1],
        "target_source_class": 0,
        "target_destination_class": 1,
        "global_train_class_counts": {str(k): int(v) for k, v in totals.items()},
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
