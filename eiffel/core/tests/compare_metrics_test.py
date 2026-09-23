"""Tests for round/attack metric comparison outputs."""

import csv

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
