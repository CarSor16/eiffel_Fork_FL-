"""Tests for round/attack metric comparison outputs."""

import csv
import json

import numpy as np

from eiffel.analysis.compare_metrics import RunSpec, analyse
from eiffel.storage.round_store import RoundStore


def _write_run(path, *, mechanism, accuracy, macro_f1, botnet_recall):
    weights = [np.array([1.0], dtype=np.float32)]
    with RoundStore(path) as store:
        store.save_global(0, weights)
        for round_number in (1, 2):
            delta = [np.array([0.1 * round_number], dtype=np.float32)]
            store.save_client(
                round_number,
                "benign_0",
                submitted_update=delta,
                malicious=False,
                attack_active=False,
                mechanism="none",
            )
            store.save_client_metrics(
                round_number,
                "benign_0",
                {
                    "global": {
                        "accuracy": accuracy - 0.02 * (2 - round_number),
                        "macro_f1": macro_f1 - 0.02 * (2 - round_number),
                    },
                    "Botnet": {
                        "recall": botnet_recall - 0.02 * (2 - round_number),
                        "missrate": 1.0
                        - (botnet_recall - 0.02 * (2 - round_number)),
                    },
                },
                phase="fit",
            )
            store.save_round_metadata(
                round_number,
                attack_mechanism=mechanism,
                attack_multiplier=0.0 if mechanism == "none" else 1.0,
                malicious_clients=0 if mechanism == "none" else 1,
            )
            store.save_global(
                round_number,
                [weights[0] + delta[0]],
            )
            store.mark_round_complete(round_number)


def test_analysis_writes_round_final_delta_and_family_outputs(tmp_path):
    clean = tmp_path / "clean" / "round_state.h5"
    attacked = tmp_path / "sign_flip" / "round_state.h5"
    clean.parent.mkdir()
    attacked.parent.mkdir()
    _write_run(
        clean,
        mechanism="none",
        accuracy=0.90,
        macro_f1=0.88,
        botnet_recall=0.86,
    )
    _write_run(
        attacked,
        mechanism="sign_flip",
        accuracy=0.70,
        macro_f1=0.62,
        botnet_recall=0.50,
    )

    output = tmp_path / "analysis"
    rc = analyse(
        [RunSpec("clean", clean), RunSpec("sign_flip", attacked)],
        output,
        ("accuracy", "macro_f1"),
        "fit",
    )

    assert rc == 0
    expected = {
        "round_metrics.csv",
        "final_metrics.csv",
        "delta_vs_clean.csv",
        "final_metrics_by_attack.png",
        "delta_vs_clean.png",
        "round_accuracy.png",
        "round_macro_f1.png",
        "round_delta_vs_clean.csv",
        "round_delta_accuracy.png",
        "round_delta_macro_f1.png",
        "per_family_metrics.csv",
        "final_per_family_recall.png",
        "per_family_recall_delta_vs_clean.csv",
        "per_family_recall_delta_vs_clean.png",
        "round_family_recall_botnet.png",
    }
    assert expected.issubset({path.name for path in output.iterdir()})

    with (output / "delta_vs_clean.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    accuracy_delta = next(
        float(row["delta_vs_clean"])
        for row in rows
        if row["attack"] == "sign_flip" and row["metric"] == "accuracy"
    )
    assert np.isclose(accuracy_delta, -0.20)

    with (output / "round_delta_vs_clean.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        round_rows = list(csv.DictReader(handle))
    round_one_delta = next(
        float(row["delta_vs_clean"])
        for row in round_rows
        if row["attack"] == "sign_flip"
        and row["metric"] == "accuracy"
        and row["round"] == "1"
    )
    round_two_delta = next(
        float(row["delta_vs_clean"])
        for row in round_rows
        if row["attack"] == "sign_flip"
        and row["metric"] == "accuracy"
        and row["round"] == "2"
    )
    assert np.isclose(round_one_delta, -0.20)
    assert np.isclose(round_two_delta, -0.20)



def test_analysis_falls_back_to_eiffel_json_for_legacy_hdf5(tmp_path):
    run_dir = tmp_path / "legacy_clean"
    run_dir.mkdir()
    h5_path = run_dir / "round_state.h5"
    weights = [np.array([1.0], dtype=np.float32)]

    # Simulate an older round_state.h5 that stores FL state but no client metrics.
    with RoundStore(h5_path) as store:
        store.save_global(0, weights)
        store.save_client(
            1,
            "benign_0",
            submitted_update=[np.array([0.1], dtype=np.float32)],
            malicious=False,
            attack_active=False,
            mechanism="none",
        )
        store.save_round_metadata(
            1,
            attack_mechanism="none",
            attack_multiplier=0.0,
            malicious_clients=0,
        )
        store.save_global(1, [np.array([1.1], dtype=np.float32)])
        store.mark_round_complete(1)

    distributed = {
        "benign_0": {
            "1": {
                "global": {
                    "accuracy": 0.81,
                    "macro_f1": 0.78,
                },
                "Botnet": {
                    "recall": 0.66,
                    "missrate": 0.34,
                },
            }
        }
    }
    (run_dir / "distributed.json").write_text(
        json.dumps(distributed), encoding="utf-8"
    )

    output = tmp_path / "legacy_analysis"
    rc = analyse(
        [RunSpec("clean", h5_path)],
        output,
        ("accuracy", "macro_f1"),
        "auto",
    )

    assert rc == 0
    with (output / "round_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert np.isclose(float(rows[0]["accuracy"]), 0.81)
    assert np.isclose(float(rows[0]["macro_f1"]), 0.78)

    with (output / "per_family_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        family_rows = list(csv.DictReader(handle))
    botnet_recall = next(
        float(row["value"])
        for row in family_rows
        if row["family"] == "Botnet" and row["metric"] == "recall"
    )
    assert np.isclose(botnet_recall, 0.66)



def test_analysis_accepts_round_first_and_json_encoded_metrics(tmp_path):
    run_dir = tmp_path / "legacy_round_first"
    run_dir.mkdir()
    h5_path = run_dir / "round_state.h5"
    weights = [np.array([1.0], dtype=np.float32)]
    with RoundStore(h5_path) as store:
        store.save_global(0, weights)
        store.save_client(
            1,
            "benign_0",
            submitted_update=[np.array([0.1], dtype=np.float32)],
        )
        store.save_global(1, [np.array([1.1], dtype=np.float32)])
        store.mark_round_complete(1)

    fit = {
        "1": {
            "benign_0": json.dumps(
                {
                    "global": {"accuracy": 0.79, "macro_f1": 0.74},
                    "Botnet": {"recall": 0.61, "missrate": 0.39},
                }
            )
        }
    }
    (run_dir / "fit.json").write_text(json.dumps(fit), encoding="utf-8")

    output = tmp_path / "round_first_analysis"
    rc = analyse(
        [RunSpec("clean", h5_path)],
        output,
        ("accuracy", "macro_f1"),
        "fit",
    )
    assert rc == 0

    with (output / "round_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert np.isclose(float(rows[0]["accuracy"]), 0.79)
    assert np.isclose(float(rows[0]["macro_f1"]), 0.74)
