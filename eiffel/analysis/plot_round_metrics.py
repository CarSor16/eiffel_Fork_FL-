"""Visualise Eiffel metrics and audit signals across communication rounds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GLOBAL_METRICS = (
    "accuracy",
    "f1",
    "precision",
    "recall",
    "missrate",
    "fallout",
    "macro_attack_recall",
    "min_attack_recall",
    "macro_attack_missrate",
)

FAMILY_METRICS = (
    "recall",
    "missrate",
    "false_positive_rate",
    "specificity",
)


def _latest_run(root: Path) -> Path:
    candidates: list[Path] = []
    if root.exists():
        for path in root.rglob("*"):
            if path.is_dir() and (
                (path / "distributed.json").exists()
                or (path / "round_state.h5").exists()
            ):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No Eiffel run containing distributed.json or round_state.h5 under {root}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_run(path: str | Path | None) -> Path:
    if path is None:
        return _latest_run(Path("outputs"))
    candidate = Path(path)
    if candidate.is_file():
        return candidate.parent
    if candidate.is_dir():
        if (candidate / "distributed.json").exists() or (
            candidate / "round_state.h5"
        ).exists():
            return candidate
        return _latest_run(candidate)
    raise FileNotFoundError(candidate)


def _load_distributed(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "distributed.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(values)) if values else float("nan")


def aggregate_distributed(
    distributed: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_values: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    family_values: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for _, rounds in distributed.items():
        if not isinstance(rounds, dict):
            continue
        for round_key, scopes in rounds.items():
            try:
                round_no = int(round_key)
            except (TypeError, ValueError):
                continue
            if not isinstance(scopes, dict):
                continue

            global_scope = scopes.get("global", {})
            if isinstance(global_scope, dict):
                for metric in GLOBAL_METRICS:
                    value = global_scope.get(metric)
                    if isinstance(value, (int, float)):
                        global_values[round_no][metric].append(float(value))

            for family, metrics in scopes.items():
                if family in {"global", "fit", "_cid"} or not isinstance(metrics, dict):
                    continue
                for metric in FAMILY_METRICS:
                    value = metrics.get(metric)
                    if isinstance(value, (int, float)):
                        family_values[(round_no, str(family))][metric].append(
                            float(value)
                        )

    global_rows = []
    for round_no in sorted(global_values):
        row: dict[str, Any] = {"round": round_no}
        for metric, values in global_values[round_no].items():
            row[metric] = _mean(values)
        global_rows.append(row)

    family_rows = []
    for (round_no, family), metrics in sorted(family_values.items()):
        row = {"round": round_no, "family": family}
        for metric, values in metrics.items():
            row[metric] = _mean(values)
        family_rows.append(row)

    return pd.DataFrame(global_rows), pd.DataFrame(family_rows)


def load_audit(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "round_state.h5"
    if not path.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5:
        clients_root = h5.get("clients")
        if clients_root is None:
            return pd.DataFrame()
        for round_name in clients_root:
            try:
                round_no = int(round_name.split("_")[-1])
            except ValueError:
                continue
            round_group = clients_root[round_name]
            for client_key in round_group:
                client_group = round_group[client_key]
                audit = client_group.get("audit")
                if audit is None:
                    continue
                row: dict[str, Any] = {
                    "round": round_no,
                    "cid": str(client_group.attrs.get("cid", client_key)),
                    "malicious": bool(client_group.attrs.get("malicious", 0)),
                    "attack_active": bool(
                        client_group.attrs.get("attack_active", 0)
                    ),
                    "mechanism": str(
                        client_group.attrs.get("mechanism", "none")
                    ),
                }
                for key, value in audit.attrs.items():
                    row[str(key)] = float(value)
                rows.append(row)
    return pd.DataFrame(rows)


def _plot_global(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    if df.empty:
        return
    available = [
        metric
        for metric in (
            "accuracy",
            "f1",
            "recall",
            "precision",
            "macro_attack_recall",
            "min_attack_recall",
        )
        if metric in df.columns
    ]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in available:
        ax.plot(df["round"], df[metric], marker="o", label=metric)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Global metrics by communication round")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "global_metrics_by_round.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def _plot_families(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    show: bool,
) -> None:
    if df.empty or metric not in df.columns:
        return
    subset = df.dropna(subset=[metric])
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for family, family_df in subset.groupby("family"):
        family_df = family_df.sort_values("round")
        ax.plot(
            family_df["round"],
            family_df[metric],
            marker="o",
            label=str(family),
        )
    ax.set_xlabel("Communication round")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Per-family {metric} by communication round")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"family_{metric}_by_round.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def _plot_audit(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    if df.empty:
        return
    for metric in ("l2", "cosine_to_mean", "distance_to_mean"):
        if metric not in df.columns:
            continue
        grouped = (
            df.groupby(["round", "malicious"], as_index=False)[metric]
            .mean()
            .sort_values("round")
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        for malicious, group in grouped.groupby("malicious"):
            label = "malicious" if bool(malicious) else "benign"
            ax.plot(group["round"], group[metric], marker="o", label=label)
        ax.set_xlabel("Communication round")
        ax.set_ylabel(metric)
        ax.set_title(f"Mean client-update {metric} by round")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"audit_{metric}_by_round.png", dpi=160)
        if show:
            plt.show()
        plt.close(fig)


def build_report(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    show: bool = False,
) -> Path:
    out_dir = output_dir or (run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    distributed = _load_distributed(run_dir)
    global_df, family_df = aggregate_distributed(distributed)
    audit_df = load_audit(run_dir)

    if not global_df.empty:
        global_df.to_csv(out_dir / "round_global_metrics.csv", index=False)
    if not family_df.empty:
        family_df.to_csv(out_dir / "round_family_metrics.csv", index=False)
    if not audit_df.empty:
        audit_df.to_csv(out_dir / "round_audit_metrics.csv", index=False)

    _plot_global(global_df, out_dir, show)
    _plot_families(family_df, out_dir, "recall", show)
    _plot_families(family_df, out_dir, "missrate", show)
    _plot_families(family_df, out_dir, "false_positive_rate", show)
    _plot_audit(audit_df, out_dir, show)

    summary = {
        "run_directory": str(run_dir.resolve()),
        "global_rounds": int(global_df["round"].nunique())
        if not global_df.empty
        else 0,
        "families": sorted(family_df["family"].dropna().unique().tolist())
        if not family_df.empty
        else [],
        "audit_rows": int(len(audit_df)),
        "output_directory": str(out_dir.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot global, per-attack-family, and update-audit metrics across "
            "federated communication rounds."
        )
    )
    parser.add_argument(
        "run",
        nargs="?",
        default=None,
        help=(
            "Run directory, outputs root, distributed.json, or round_state.h5. "
            "If omitted, the newest run under ./outputs is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG/CSV analysis artifacts (default: RUN/analysis).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open figures interactively.",
    )
    args = parser.parse_args(argv)

    run_dir = resolve_run(args.run)
    out_dir = build_report(
        run_dir,
        output_dir=args.output_dir,
        show=args.show,
    )
    print(f"Run: {run_dir}")
    print(f"Analysis written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
