"""Tests for the compact thesis-oriented per-run analysis."""

import csv

import numpy as np

from eiffel.analysis.compact_round_analysis import analyse_compact
from eiffel.analysis.compare_metrics import RunSpec
from eiffel.storage.round_store import RoundStore


def _write_compact_run(path, *, mechanism: str, malicious: bool) -> None:
    weights = [np.array([1.0, -1.0], dtype=np.float32)]
    with RoundStore(path) as store:
        store.save_global(0, weights)
        for round_number in (1, 2, 3):
            benign_update = [
                np.array(
                    [0.10 * round_number, -0.05 * round_number],
                    dtype=np.float32,
                )
            ]
            store.save_client(
                round_number,
                "pool_benign_0",
                submitted_update=benign_update,
                audit={
                    "l2": float(np.linalg.norm(benign_update[0])),
                    "cosine_to_mean": 0.95,
                    "distance_to_mean": 0.10,
                    "sign_agreement_mean": 1.0,
                },
                malicious=False,
                attack_active=False,
                mechanism="none",
            )
            store.save_client_metrics(
                round_number,
                "pool_benign_0",
                {
                    "global": {
                        "accuracy": 0.90 - 0.02 * round_number,
                        "macro_f1": 0.88 - 0.02 * round_number,
                        "macro_attack_recall": 0.86 - 0.03 * round_number,
                        "min_attack_recall": 0.80 - 0.04 * round_number,
                    }
                },
                phase="evaluate",
            )

            malicious_count = 0
            active_count = 0
            if malicious:
                malicious_count = 1
                attack_active = round_number >= 2
                active_count = 1 if attack_active else 0
                pre_attack = [
                    np.array(
                        [0.08 * round_number, -0.04 * round_number],
                        dtype=np.float32,
                    )
                ]
                submitted = (
                    [(-2.0 * pre_attack[0]).astype(np.float32)]
                    if attack_active
                    else [pre_attack[0].copy()]
                )
                store.save_client(
                    round_number,
                    "pool_malicious_0",
                    submitted_update=submitted,
                    pre_attack_update=pre_attack if attack_active else None,
                    audit={
                        "l2": float(np.linalg.norm(submitted[0])),
                        "cosine_to_mean": -0.80 if attack_active else 0.90,
                        "distance_to_mean": 0.60 if attack_active else 0.12,
                        "sign_agreement_mean": 0.0 if attack_active else 1.0,
                    },
                    malicious=True,
                    attack_active=attack_active,
                    mechanism=mechanism if attack_active else "none",
                )
                store.save_client_metrics(
                    round_number,
                    "pool_malicious_0",
                    {
                        "global": {
                            "accuracy": 0.82 - 0.05 * round_number,
                            "macro_f1": 0.79 - 0.05 * round_number,
                            "macro_attack_recall": 0.74 - 0.06 * round_number,
                            "min_attack_recall": 0.65 - 0.07 * round_number,
                        }
                    },
                    phase="evaluate",
                )

            store.save_round_metadata(
                round_number,
                attack_mechanism=mechanism,
                attack_multiplier=1.0 if active_count else 0.0,
                malicious_clients=malicious_count,
            )
            store.save_global(
                round_number,
                [weights[0] + benign_update[0]],
            )
            store.mark_round_complete(round_number)


def test_compact_analysis_writes_two_plots_per_run_and_attack_context(tmp_path):
    clean = tmp_path / "outputs" / "2026-09-23" / "clean" / "round_state.h5"
    attacked = (
        tmp_path / "outputs" / "2026-09-23" / "signflip" / "round_state.h5"
    )
    clean.parent.mkdir(parents=True)
    attacked.parent.mkdir(parents=True)
    _write_compact_run(clean, mechanism="none", malicious=False)
    _write_compact_run(attacked, mechanism="sign_flip", malicious=True)

    output = tmp_path / "analysis-results"
    rc = analyse_compact(
        [RunSpec("clean", clean), RunSpec("sign_flip", attacked)],
        output,
    )
    assert rc == 0
    assert (output / "runs_index.csv").exists()

    run_dirs = sorted(path for path in output.iterdir() if path.is_dir())
    assert len(run_dirs) == 2
    expected = {
        "round_performance.csv",
        "round_updates.csv",
        "performance_by_round.png",
        "updates_by_round.png",
    }
    for run_dir in run_dirs:
        assert {path.name for path in run_dir.iterdir()} == expected

    attacked_dir = next(path for path in run_dirs if path.name.startswith("sign_flip"))
    with (attacked_dir / "round_performance.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        performance = list(csv.DictReader(handle))
    assert [row["malicious_clients"] for row in performance] == ["1", "1", "1"]
    assert [row["active_malicious_clients"] for row in performance] == [
        "0",
        "1",
        "1",
    ]
    assert [float(row["attack_multiplier"]) for row in performance] == [
        0.0,
        1.0,
        1.0,
    ]

    with (attacked_dir / "round_updates.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        updates = list(csv.DictReader(handle))
    assert np.isnan(float(updates[0]["malicious_transform_l2_mean"]))
    assert float(updates[1]["malicious_transform_l2_mean"]) > 0.0
    assert float(updates[1]["malicious_to_benign_l2_ratio"]) > 0.0
