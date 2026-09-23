"""Compare Eiffel FL-security metrics across rounds and attacks.

Reads the per-round metrics stored by InstrumentedFedAvg in round_state.h5.
It writes CSV tables, round-by-round line plots, final grouped bar plots, and
clean-baseline deltas. Multiple runs with the same attack label are aggregated
as mean/std, which makes the same script usable for multi-seed experiments.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

DEFAULT_METRICS = (
    "accuracy",
    "f1",
    "macro_f1",
    "weighted_f1",
    "recall",
    "missrate",
    "mcc",
    "macro_attack_recall",
    "min_attack_recall",
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path


def _metric_candidates(metric: str) -> tuple[str, ...]:
    return (metric,) if "." in metric else (f"global.{metric}", metric)


def _decode(value: object) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _infer_attack_label(h5_path: Path) -> str:
    mechanism = "none"
    schedule = "continuous"
    with h5py.File(h5_path, "r") as h5:
        rounds = h5.get("rounds")
        if rounds is not None:
            names = sorted(rounds.keys())
            if names:
                mechanism = _decode(
                    rounds[names[-1]].attrs.get("attack_mechanism", "none")
                )

    config_path = h5_path.parent / ".hydra" / "config.yaml"
    if config_path.exists():
        try:
            cfg = OmegaConf.load(config_path)
            attack_cfg = cfg.get("model_attack", {}) or {}
            mechanism = str(attack_cfg.get("mechanism", mechanism))
            schedule_cfg = attack_cfg.get("schedule", {}) or {}
            schedule = str(schedule_cfg.get("type", "continuous"))
            n_attackers = int(cfg.get("num_attackers", 0))
            poisoning = cfg.get("attacks", []) or []
            if mechanism == "none" and n_attackers > 0 and len(poisoning) > 0:
                first = poisoning[0]
                profile_value = str(first.get("profile", ""))
                poison_type = str(first.get("type", ""))
                if profile_value not in {"", "0", "0.0", "clean"}:
                    mechanism = "label_flip"
                    if poison_type:
                        mechanism += f"_{poison_type}"
        except Exception:
            # HDF5 metadata remains sufficient for model attacks. Explicit --run
            # LABEL=PATH can always be used when a historical Hydra config is absent
            # or cannot be parsed.
            pass

    label = "clean" if mechanism == "none" else mechanism
    if schedule not in {"", "continuous"} and label != "clean":
        label += f"[{schedule}]"
    return label


def discover_runs(root: Path) -> list[RunSpec]:
    return [
        RunSpec(_infer_attack_label(path), path)
        for path in sorted(root.rglob("round_state.h5"))
    ]


def parse_run_arg(value: str) -> RunSpec:
    if "=" not in value:
        path = Path(value).expanduser().resolve()
        return RunSpec(_infer_attack_label(path), path)
    label, raw_path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("--run label cannot be empty")
    return RunSpec(label.strip(), Path(raw_path).expanduser().resolve())


def _read_phase_metrics(client: h5py.Group, phase: str) -> dict[str, float]:
    try:
        attrs = client["metrics"][phase].attrs
    except KeyError:
        return {}
    return {
        str(key): float(value)
        for key, value in attrs.items()
        if np.isscalar(value)
    }


def _client_metrics(client: h5py.Group, phase: str) -> dict[str, float]:
    phases = ("evaluate", "fit") if phase == "auto" else (phase,)
    for candidate in phases:
        decoded = _read_phase_metrics(client, candidate)
        if decoded:
            return decoded
    return {}


def read_run_metrics(
    run: RunSpec, metrics: Sequence[str], phase: str = "auto"
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with h5py.File(run.path, "r") as h5:
        clients_root = h5.get("clients")
        if clients_root is None:
            return rows
        for round_name in sorted(clients_root.keys()):
            round_number = int(round_name.rsplit("_", 1)[-1])
            values: dict[str, list[float]] = {m: [] for m in metrics}
            for client in clients_root[round_name].values():
                decoded = _client_metrics(client, phase)
                for metric in metrics:
                    for key in _metric_candidates(metric):
                        if key in decoded and math.isfinite(decoded[key]):
                            values[metric].append(decoded[key])
                            break
            row: dict[str, object] = {
                "attack": run.label,
                "run": str(run.path.parent),
                "round": round_number,
            }
            for metric, observed in values.items():
                row[metric] = (
                    float(np.mean(observed)) if observed else math.nan
                )
            rows.append(row)
    return rows


def read_per_family(run: RunSpec, phase: str = "auto") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with h5py.File(run.path, "r") as h5:
        clients_root = h5.get("clients")
        if clients_root is None:
            return rows
        for round_name in sorted(clients_root.keys()):
            round_number = int(round_name.rsplit("_", 1)[-1])
            values: dict[tuple[str, str], list[float]] = defaultdict(list)
            for client in clients_root[round_name].values():
                for key, value in _client_metrics(client, phase).items():
                    if "." not in key:
                        continue
                    family, metric = key.rsplit(".", 1)
                    if (
                        family in {"global", "Benign", "fit"}
                        or metric not in {"precision", "recall", "f1", "missrate"}
                    ):
                        continue
                    if math.isfinite(value):
                        values[(family, metric)].append(value)
            for (family, metric), observed in sorted(values.items()):
                rows.append(
                    {
                        "attack": run.label,
                        "run": str(run.path.parent),
                        "round": round_number,
                        "family": family,
                        "metric": metric,
                        "value": float(np.mean(observed)),
                    }
                )
    return rows


def _write_csv(
    path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _plot_round_metric(
    rows: Sequence[dict[str, object]], metric: str, output_dir: Path
) -> None:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row.get(metric, math.nan))
        if math.isfinite(value):
            grouped[(str(row["attack"]), int(row["round"]))].append(value)
    attacks = sorted({attack for attack, _ in grouped})
    if not attacks:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for attack in attacks:
        rounds = sorted(r for label, r in grouped if label == attack)
        means = [float(np.mean(grouped[(attack, r)])) for r in rounds]
        stds = [float(np.std(grouped[(attack, r)])) for r in rounds]
        line = ax.plot(rounds, means, marker="o", linewidth=1.8, label=attack)[0]
        if any(value > 0 for value in stds):
            ax.fill_between(
                rounds,
                np.asarray(means) - np.asarray(stds),
                np.asarray(means) + np.asarray(stds),
                alpha=0.15,
                color=line.get_color(),
            )
    ax.set_xlabel("Communication round")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ').title()} by round")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"round_{metric}.png", dpi=180)
    plt.close(fig)


def _final_rows(
    rows: Sequence[dict[str, object]], metrics: Sequence[str]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["attack"]), str(row["run"]))].append(row)
    finals: list[dict[str, object]] = []
    for (attack, run), run_rows in grouped.items():
        final = max(run_rows, key=lambda item: int(item["round"]))
        out = {"attack": attack, "run": run, "round": final["round"]}
        out.update({metric: final.get(metric, math.nan) for metric in metrics})
        finals.append(out)
    return finals


def _plot_final_bars(
    finals: Sequence[dict[str, object]],
    metrics: Sequence[str],
    output_dir: Path,
) -> None:
    attacks = sorted({str(row["attack"]) for row in finals})
    available = [
        metric
        for metric in metrics
        if any(
            math.isfinite(float(row.get(metric, math.nan))) for row in finals
        )
    ]
    if not attacks or not available:
        return

    x = np.arange(len(attacks), dtype=float)
    width = 0.8 / len(available)
    fig, ax = plt.subplots(figsize=(max(10.0, len(attacks) * 1.15), 6.0))
    for idx, metric in enumerate(available):
        means, stds = [], []
        for attack in attacks:
            values = [
                float(row[metric])
                for row in finals
                if str(row["attack"]) == attack
                and math.isfinite(float(row.get(metric, math.nan)))
            ]
            means.append(float(np.mean(values)) if values else math.nan)
            stds.append(float(np.std(values)) if len(values) > 1 else 0.0)
        offset = (idx - (len(available) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=metric,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=25, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Final-round metrics by attack")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "final_metrics_by_attack.png", dpi=180)
    plt.close(fig)


def _delta_vs_clean(
    finals: Sequence[dict[str, object]],
    metrics: Sequence[str],
    output_dir: Path,
) -> list[dict[str, object]]:
    clean = [row for row in finals if str(row["attack"]) == "clean"]
    if not clean:
        return []

    clean_mean: dict[str, float] = {}
    for metric in metrics:
        values = [
            float(row[metric])
            for row in clean
            if math.isfinite(float(row.get(metric, math.nan)))
        ]
        if values:
            clean_mean[metric] = float(np.mean(values))

    delta_rows: list[dict[str, object]] = []
    attacks = sorted(
        {str(row["attack"]) for row in finals if str(row["attack"]) != "clean"}
    )
    for attack in attacks:
        attack_rows = [row for row in finals if str(row["attack"]) == attack]
        for metric, baseline in clean_mean.items():
            values = [
                float(row[metric])
                for row in attack_rows
                if math.isfinite(float(row.get(metric, math.nan)))
            ]
            if values:
                delta_rows.append(
                    {
                        "attack": attack,
                        "metric": metric,
                        "delta_vs_clean": float(np.mean(values)) - baseline,
                    }
                )

    if not delta_rows:
        return []
    plot_metrics = [
        metric
        for metric in metrics
        if any(row["metric"] == metric for row in delta_rows)
    ]
    x = np.arange(len(attacks), dtype=float)
    width = 0.8 / len(plot_metrics)
    fig, ax = plt.subplots(figsize=(max(10.0, len(attacks) * 1.15), 6.0))
    for idx, metric in enumerate(plot_metrics):
        values = [
            next(
                (
                    float(row["delta_vs_clean"])
                    for row in delta_rows
                    if row["attack"] == attack and row["metric"] == metric
                ),
                math.nan,
            )
            for attack in attacks
        ]
        offset = (idx - (len(plot_metrics) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=metric)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=25, ha="right")
    ax.set_ylabel("Difference from clean")
    ax.set_title("Final-round metric change relative to clean")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "delta_vs_clean.png", dpi=180)
    plt.close(fig)
    return delta_rows


def _plot_final_family_recall(
    rows: Sequence[dict[str, object]], output_dir: Path
) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == "recall":
            grouped[
                (str(row["attack"]), str(row["run"]), str(row["family"]))
            ].append(row)

    finals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (attack, _run, family), values in grouped.items():
        final = max(values, key=lambda item: int(item["round"]))
        finals[(attack, family)].append(float(final["value"]))

    attacks = sorted({attack for attack, _ in finals})
    families = sorted({family for _, family in finals})
    if not attacks or not families:
        return

    x = np.arange(len(attacks), dtype=float)
    width = 0.8 / len(families)
    fig, ax = plt.subplots(figsize=(max(10.0, len(attacks) * 1.15), 6.0))
    for idx, family in enumerate(families):
        means = [
            float(np.mean(finals[(attack, family)]))
            if (attack, family) in finals
            else math.nan
            for attack in attacks
        ]
        offset = (idx - (len(families) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, label=family)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=25, ha="right")
    ax.set_ylabel("Recall")
    ax.set_title("Final per-family recall by attack")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "final_per_family_recall.png", dpi=180)
    plt.close(fig)


def _family_delta_vs_clean(
    rows: Sequence[dict[str, object]], output_dir: Path
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == "recall":
            grouped[
                (str(row["attack"]), str(row["run"]), str(row["family"]))
            ].append(row)

    finals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (attack, _run, family), values in grouped.items():
        final = max(values, key=lambda item: int(item["round"]))
        finals[(attack, family)].append(float(final["value"]))

    clean_families = {
        family: float(np.mean(values))
        for (attack, family), values in finals.items()
        if attack == "clean"
    }
    attacks = sorted(
        {attack for attack, _ in finals if attack != "clean"}
    )
    delta_rows: list[dict[str, object]] = []
    for attack in attacks:
        for family, baseline in sorted(clean_families.items()):
            values = finals.get((attack, family), [])
            if values:
                delta_rows.append(
                    {
                        "attack": attack,
                        "family": family,
                        "recall_delta_vs_clean": float(np.mean(values))
                        - baseline,
                    }
                )

    if not delta_rows:
        return []

    families = sorted({str(row["family"]) for row in delta_rows})
    x = np.arange(len(attacks), dtype=float)
    width = 0.8 / len(families)
    fig, ax = plt.subplots(figsize=(max(10.0, len(attacks) * 1.15), 6.0))
    for idx, family in enumerate(families):
        values = [
            next(
                (
                    float(row["recall_delta_vs_clean"])
                    for row in delta_rows
                    if row["attack"] == attack and row["family"] == family
                ),
                math.nan,
            )
            for attack in attacks
        ]
        offset = (idx - (len(families) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=family)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=25, ha="right")
    ax.set_ylabel("Recall difference from clean")
    ax.set_title("Final per-family recall change relative to clean")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "per_family_recall_delta_vs_clean.png", dpi=180)
    plt.close(fig)
    return delta_rows


def analyse(
    runs: Iterable[RunSpec],
    output_dir: Path,
    metrics: Sequence[str],
    phase: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_list = list(runs)
    if not run_list:
        raise ValueError("No round_state.h5 runs were found or provided.")
    missing = [str(run.path) for run in run_list if not run.path.exists()]
    if missing:
        raise FileNotFoundError("Missing HDF5 run(s): " + ", ".join(missing))

    rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    for run in run_list:
        rows.extend(read_run_metrics(run, metrics, phase=phase))
        family_rows.extend(read_per_family(run, phase=phase))
    if not rows:
        raise ValueError("No client metrics found in the supplied HDF5 files.")

    _write_csv(
        output_dir / "round_metrics.csv",
        rows,
        ("attack", "run", "round", *metrics),
    )
    for metric in metrics:
        if any(
            math.isfinite(float(row.get(metric, math.nan))) for row in rows
        ):
            _plot_round_metric(rows, metric, output_dir)

    finals = _final_rows(rows, metrics)
    _write_csv(
        output_dir / "final_metrics.csv",
        finals,
        ("attack", "run", "round", *metrics),
    )
    _plot_final_bars(finals, metrics, output_dir)

    deltas = _delta_vs_clean(finals, metrics, output_dir)
    if deltas:
        _write_csv(
            output_dir / "delta_vs_clean.csv",
            deltas,
            ("attack", "metric", "delta_vs_clean"),
        )

    if family_rows:
        _write_csv(
            output_dir / "per_family_metrics.csv",
            family_rows,
            ("attack", "run", "round", "family", "metric", "value"),
        )
        _plot_final_family_recall(family_rows, output_dir)
        family_deltas = _family_delta_vs_clean(family_rows, output_dir)
        if family_deltas:
            _write_csv(
                output_dir / "per_family_recall_delta_vs_clean.csv",
                family_deltas,
                ("attack", "family", "recall_delta_vs_clean"),
            )

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Eiffel HDF5 metrics across rounds and attacks."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Recursively discover round_state.h5 files below this directory.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        type=parse_run_arg,
        metavar="[LABEL=]PATH",
        help="Add one explicit run; repeat for multiple attacks/seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-results"),
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metrics to compare (global. prefix is optional).",
    )
    parser.add_argument(
        "--phase",
        choices=("auto", "fit", "evaluate"),
        default="auto",
        help="auto prefers evaluate metrics, then fit metrics.",
    )
    args = parser.parse_args(argv)

    runs: list[RunSpec] = list(args.run)
    if args.runs_root is not None:
        runs.extend(discover_runs(args.runs_root))
    if not runs:
        parser.error("provide --runs-root or at least one --run")
    return analyse(runs, args.output_dir, args.metrics, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
