"""Quantify statistical heterogeneity of fixed federated client partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64, copy=False)
    q = q.astype(np.float64, copy=False)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def analyse_distribution(
    frame: pd.DataFrame,
    *,
    target_class: int | None = None,
    malicious_ids: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {"client_id", "class_id", "count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing distribution columns: {sorted(missing)}")

    data = frame.copy()
    data["client_id"] = data["client_id"].astype(int)
    data["class_id"] = data["class_id"].astype(int)
    data["count"] = data["count"].astype(float)
    if (data["count"] < 0).any():
        raise ValueError("Class counts must be non-negative.")

    clients = sorted(data["client_id"].unique())
    classes = sorted(data["class_id"].unique())
    matrix = (
        data.pivot_table(
            index="client_id",
            columns="class_id",
            values="count",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=clients, columns=classes, fill_value=0.0)
        .astype(float)
    )
    totals = matrix.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Every client must contain at least one sample.")

    probs = matrix.div(totals, axis=0)
    max_entropy = np.log(max(2, len(classes)))

    client_rows = []
    for client_id in clients:
        p = probs.loc[client_id].to_numpy(dtype=float)
        client_rows.append(
            {
                "client_id": int(client_id),
                "samples": int(totals.loc[client_id]),
                "active_classes": int((matrix.loc[client_id] > 0).sum()),
                "normalized_label_entropy": _entropy(p) / max_entropy,
                "max_class_share": float(p.max()),
            }
        )
    per_client = pd.DataFrame(client_rows)

    class_rows = []
    for class_id in classes:
        counts = matrix[class_id].to_numpy(dtype=float)
        mean = float(np.mean(counts))
        std = float(np.std(counts))
        class_rows.append(
            {
                "class_id": int(class_id),
                "samples": int(counts.sum()),
                "clients_present": int((counts > 0).sum()),
                "client_count_cv": std / mean if mean > 0 else np.nan,
                "max_client_share": float(counts.max() / counts.sum())
                if counts.sum() > 0
                else np.nan,
            }
        )
    per_class = pd.DataFrame(class_rows)

    js_values = []
    for left_idx in range(len(clients)):
        for right_idx in range(left_idx + 1, len(clients)):
            js_values.append(
                _js_divergence(
                    probs.iloc[left_idx].to_numpy(dtype=float),
                    probs.iloc[right_idx].to_numpy(dtype=float),
                )
            )

    summary = {
        "n_clients": len(clients),
        "n_classes": len(classes),
        "total_samples": int(matrix.to_numpy().sum()),
        "mean_client_normalized_entropy": float(
            per_client["normalized_label_entropy"].mean()
        ),
        "min_client_normalized_entropy": float(
            per_client["normalized_label_entropy"].min()
        ),
        "mean_client_max_class_share": float(per_client["max_class_share"].mean()),
        "mean_pairwise_js_divergence": float(np.mean(js_values)) if js_values else 0.0,
        "max_pairwise_js_divergence": float(np.max(js_values)) if js_values else 0.0,
        "mean_class_client_count_cv": float(per_class["client_count_cv"].mean()),
    }

    if target_class is not None:
        target_class = int(target_class)
        if target_class not in matrix.columns:
            raise ValueError(f"Target class {target_class} is not present.")
        target_counts = matrix[target_class]
        total_target = float(target_counts.sum())
        bad = set(int(value) for value in (malicious_ids or []))
        malicious_target = float(
            target_counts.loc[
                [idx for idx in target_counts.index if int(idx) in bad]
            ].sum()
        )
        summary["target_class"] = target_class
        summary["target_total_samples"] = int(total_target)
        summary["target_malicious_samples"] = int(malicious_target)
        summary["target_malicious_concentration"] = (
            malicious_target / total_target if total_target > 0 else np.nan
        )

    return per_client, per_class, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("distribution", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-class", type=int)
    parser.add_argument("--malicious-ids", default="")
    args = parser.parse_args(argv)

    malicious_ids = [
        int(value.strip())
        for value in args.malicious_ids.split(",")
        if value.strip()
    ]
    frame = pd.read_csv(args.distribution)
    per_client, per_class, summary = analyse_distribution(
        frame,
        target_class=args.target_class,
        malicious_ids=malicious_ids,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_client.to_csv(args.output_dir / "client_heterogeneity.csv", index=False)
    per_class.to_csv(args.output_dir / "class_heterogeneity.csv", index=False)
    (args.output_dir / "heterogeneity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
