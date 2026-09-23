"""Compact per-run analysis for FL-security experiments.

The default thesis workflow focuses on one attack/run at a time.  Each run produces
two plots and two CSV files:
- performance_by_round.png / round_performance.csv
- updates_by_round.png / round_updates.csv

The plots deliberately keep attack context visible without producing one image per
metric: total/malicious client counts are shown in titles and rounds with active model
poisoning are shaded according to the recorded attack multiplier.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eiffel.analysis.compare_metrics import RunSpec, discover_runs

PERFORMANCE_METRICS = (
    "accuracy",
    "macro_f1",
    "macro_attack_recall",
    "min_attack_recall",
)
AUDIT_METRICS = (
    "l2",
    "cosine_to_mean",
    "distance_to_mean",
    "sign_agreement_mean",
)


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    ).strip("_.")
    return cleaned or "run"


def _metric_value(client: h5py.Group, metric: str) -> float:
    for phase in ("evaluate", "fit"):
        try:
            attrs = client["metrics"][phase].attrs
        except KeyError:
            continue
        for key in (f"global.{metric}", metric):
            if key in attrs:
                value = float(attrs[key])
                if math.isfinite(value):
                    return value
    return math.nan


def _audit_value(client: h5py.Group, metric: str) -> float:
    try:
        value = float(client["audit"].attrs[metric])
    except KeyError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def _difference_l2(
    submitted: h5py.Group | None,
    pre_attack: h5py.Group | None,
) -> float:
    if submitted is None or pre_attack is None:
        return math.nan
    submitted_names = sorted(submitted.keys())
    pre_attack_names = sorted(pre_attack.keys())
    if submitted_names != pre_attack_names:
        return math.nan
    squared = 0.0
    found = False
    for name in submitted_names:
        submitted_array = np.asarray(submitted[name], dtype=np.float64)
        pre_attack_array = np.asarray(pre_attack[name], dtype=np.float64)
        if submitted_array.shape != pre_attack_array.shape:
            return math.nan
        squared += float(np.square(submitted_array - pre_attack_array).sum())
        found = True
    return float(math.sqrt(squared)) if found else math.nan


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else math.nan


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if abs(denominator) <= 1e-12:
        return math.nan
    return numerator / denominator


def read_compact_run(run: RunSpec) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    performance: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []

    with h5py.File(run.path, "r") as h5:
        clients_root = h5.get("clients")
        if clients_root is None:
            return performance, updates

        rounds_root = h5.get("rounds")
        for round_name in sorted(clients_root.keys()):
            round_number = int(round_name.rsplit("_", 1)[-1])
            clients = clients_root[round_name]
            client_groups = list(clients.values())
            total_clients = len(client_groups)
            malicious_clients = sum(
                bool(int(client.attrs.get("malicious", 0)))
                for client in client_groups
            )
            active_malicious = sum(
                bool(int(client.attrs.get("malicious", 0)))
                and bool(int(client.attrs.get("attack_active", 0)))
                for client in client_groups
            )

            mechanism = run.label
            attack_multiplier = 0.0
            if rounds_root is not None and round_name in rounds_root:
                metadata = rounds_root[round_name].attrs
                attack_multiplier = float(metadata.get("attack_multiplier", 0.0))
                recorded_malicious = int(
                    metadata.get("malicious_clients", malicious_clients)
                )
                if recorded_malicious != malicious_clients:
                    malicious_clients = recorded_malicious

            base = {
                "attack": mechanism,
                "run": str(run.path.parent),
                "round": round_number,
                "total_clients": total_clients,
                "malicious_clients": malicious_clients,
                "active_malicious_clients": active_malicious,
                "malicious_fraction": (
                    malicious_clients / total_clients if total_clients else 0.0
                ),
                "attack_multiplier": attack_multiplier,
            }

            perf_row = dict(base)
            for metric in PERFORMANCE_METRICS:
                perf_row[metric] = _mean(
                    _metric_value(client, metric) for client in client_groups
                )
            performance.append(perf_row)

            benign = [
                client
                for client in client_groups
                if not bool(int(client.attrs.get("malicious", 0)))
            ]
            malicious = [
                client
                for client in client_groups
                if bool(int(client.attrs.get("malicious", 0)))
            ]

            update_row = dict(base)
            for metric in AUDIT_METRICS:
                update_row[f"benign_{metric}_mean"] = _mean(
                    _audit_value(client, metric) for client in benign
                )
                update_row[f"malicious_{metric}_mean"] = _mean(
                    _audit_value(client, metric) for client in malicious
                )

            update_row["malicious_to_benign_l2_ratio"] = _ratio(
                float(update_row["malicious_l2_mean"]),
                float(update_row["benign_l2_mean"]),
            )
            update_row["malicious_transform_l2_mean"] = _mean(
                _difference_l2(
                    client.get("submitted_update"),
                    client.get("pre_attack_update"),
                )
                for client in malicious
                if "pre_attack_update" in client
            )
            updates.append(update_row)

    return performance, updates


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _context_title(rows: Sequence[dict[str, object]], label: str) -> str:
    if not rows:
        return label
    total = int(rows[0]["total_clients"])
    malicious = int(rows[0]["malicious_clients"])
    fraction = 100.0 * float(rows[0]["malicious_fraction"])
    return (
        f"{label} — {malicious}/{total} malicious clients "
        f"({fraction:.0f}%)"
    )


def _shade_attack(ax, rows: Sequence[dict[str, object]]) -> None:
    labelled = False
    for row in rows:
        multiplier = float(row.get("attack_multiplier", 0.0))
        active = int(row.get("active_malicious_clients", 0))
        if multiplier <= 0.0 and active <= 0:
            continue
        round_number = int(row["round"])
        alpha = min(0.22, 0.06 + 0.12 * max(0.0, multiplier))
        ax.axvspan(
            round_number - 0.45,
            round_number + 0.45,
            alpha=alpha,
            label="Attack active" if not labelled else None,
        )
        labelled = True


def _plot_performance(
    rows: Sequence[dict[str, object]],
    output: Path,
    label: str,
) -> None:
    if not rows:
        return
    rounds = [int(row["round"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    plotted = False
    for metric in PERFORMANCE_METRICS:
        values = [float(row.get(metric, math.nan)) for row in rows]
        if not any(math.isfinite(value) for value in values):
            continue
        ax.plot(
            rounds,
            values,
            marker="o",
            linewidth=1.8,
            label=metric.replace("_", " ").title(),
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return

    _shade_attack(ax, rows)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(_context_title(rows, label) + "\nPerformance by round")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_updates(
    rows: Sequence[dict[str, object]],
    output: Path,
    label: str,
) -> None:
    if not rows:
        return
    rounds = [int(row["round"]) for row in rows]
    benign_l2 = [float(row.get("benign_l2_mean", math.nan)) for row in rows]
    malicious_l2 = [float(row.get("malicious_l2_mean", math.nan)) for row in rows]

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    if any(math.isfinite(value) for value in benign_l2):
        ax.plot(
            rounds,
            benign_l2,
            marker="o",
            linewidth=1.8,
            label="Benign mean update L2",
        )
    if any(math.isfinite(value) for value in malicious_l2):
        ax.plot(
            rounds,
            malicious_l2,
            marker="o",
            linewidth=1.8,
            label="Malicious mean update L2",
        )

    _shade_attack(ax, rows)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Mean submitted-update L2 norm")
    ax.set_title(_context_title(rows, label) + "\nUpdate magnitude by round")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _run_output_name(run: RunSpec) -> str:
    parent = run.path.parent
    date = parent.parent.name if parent.parent != parent else ""
    stamp = "_".join(value for value in (date, parent.name) if value)
    return _safe_name(f"{run.label}__{stamp}")


def analyse_compact(runs: Sequence[RunSpec], output_dir: Path) -> int:
    if not runs:
        raise ValueError("No round_state.h5 runs were found or provided.")
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    usable = 0
    for run in runs:
        performance, updates = read_compact_run(run)
        if not performance:
            print(f"Skipping run without compact HDF5 round metrics: {run.path}")
            continue

        usable += 1
        run_output = output_dir / _run_output_name(run)
        run_output.mkdir(parents=True, exist_ok=True)
        _write_csv(run_output / "round_performance.csv", performance)
        _write_csv(run_output / "round_updates.csv", updates)
        _plot_performance(
            performance,
            run_output / "performance_by_round.png",
            run.label,
        )
        _plot_updates(
            updates,
            run_output / "updates_by_round.png",
            run.label,
        )

        first = performance[0]
        index_rows.append(
            {
                "attack": run.label,
                "run": str(run.path.parent),
                "analysis_dir": str(run_output),
                "rounds": len(performance),
                "total_clients": first["total_clients"],
                "malicious_clients": first["malicious_clients"],
                "malicious_fraction": first["malicious_fraction"],
            }
        )

    if not usable:
        raise ValueError(
            "No usable fresh HDF5 runs were found. Run a new experiment with the "
            "current instrumented strategy before analysing."
        )

    _write_csv(output_dir / "runs_index.csv", index_rows)
    print(
        f"Compact analysis complete: {usable} run(s), "
        "2 plots per run plus round CSV tables."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate compact per-run FL-security round analysis."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Recursively discover round_state.h5 files below this directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-results"),
    )
    args = parser.parse_args(argv)
    return analyse_compact(discover_runs(args.runs_root), args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
