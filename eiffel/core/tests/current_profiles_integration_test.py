"""Application-level smoke tests for the current shipped TOML attack profiles."""

from __future__ import annotations

import copy
import csv
import math
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from eiffel.analysis.compact_round_analysis import analyse_compact
from eiffel.analysis.compare_metrics import RunSpec
from eiffel.analysis.validate_round_state import validate
from eiffel.toml_runner import load_profile, profile_to_overrides


REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = REPO_ROOT / "experiments" / "toml"

PROFILE_CASES = (
    "synthetic_50k_clean.toml",
    "synthetic_50k_sign_flip.toml",
    "synthetic_50k_model_scaling.toml",
    "synthetic_50k_gaussian_noise.toml",
    "synthetic_50k_lie.toml",
    "synthetic_50k_gradient_mimicry.toml",
    "synthetic_50k_colluding_sign_flip.toml",
    "synthetic_50k_label_flip.toml",
    "synthetic_50k_min_max.toml",
    "synthetic_50k_min_sum.toml",
    "synthetic_50k_adaptive_stealth.toml",
    "synthetic_50k_adaptive_stealth_gradual.toml",
    "synthetic_50k_heterogeneity_aware_mimicry.toml",
    "synthetic_50k_targeted_family_poisoning.toml",
    "synthetic_50k_multiclass_sign_flip.toml",
)

RAY_STARTUP_TIMEOUT_MARKERS = (
    "Timed out after 60 seconds while waiting for node to startup",
    "The current node timed out during startup",
)


def _run_with_ray_startup_retry(
    command: list[str],
    *,
    timeout: int = 240,
) -> subprocess.CompletedProcess[str]:
    """Retry once only when Ray itself fails to start locally."""
    completed = None
    for attempt in range(2):
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        if completed.returncode == 0:
            return completed
        startup_timeout = any(
            marker in completed.stdout for marker in RAY_STARTUP_TIMEOUT_MARKERS
        )
        if not startup_timeout or attempt == 1:
            return completed
    assert completed is not None  # pragma: no cover
    return completed


def _scale_for_application_smoke(profile: dict) -> dict:
    """Reduce compute while preserving the profile's attack semantics."""
    profile = copy.deepcopy(profile)
    attack = profile.setdefault("attack", {})
    mechanism = str(attack.get("mechanism", "none")).lower()

    # Five clients keep all advanced attacks supplied with >= 2 benign updates.
    # Colluding sign flip keeps ten clients so its default 20% fraction still
    # represents two colluding attackers.
    total_clients = 10 if mechanism == "colluding_sign_flip" else 5
    profile.setdefault("experiment", {})["num_clients"] = total_clients
    profile["experiment"]["rounds"] = 2

    dataset = profile.setdefault("dataset", {})
    dataset["samples_per_client"] = 48
    dataset["central_test_size"] = 96

    training = profile.setdefault("training", {})
    training["local_epochs"] = 1
    training["batch_size"] = min(int(training.get("batch_size", 32)), 32)

    storage = profile.setdefault("storage", {})
    storage["enabled"] = True
    storage["compression_level"] = 1
    storage["flush_each_round"] = True
    storage["probe_size"] = min(int(storage.get("probe_size", 48)), 48)

    schedule = attack.setdefault("schedule", {})
    kind = str(schedule.get("type", "continuous")).lower()
    if kind == "late":
        schedule["start_round"] = 2
        schedule["end_round"] = 2
    else:
        schedule["start_round"] = 1
        schedule["end_round"] = 2

    if kind == "on_off":
        schedule["on_rounds"] = 1
        schedule["off_rounds"] = 1
    elif kind == "gradual":
        schedule["ramp_rounds"] = 2

    return profile


def _scale_for_ten_round_longitudinal(profile: dict) -> dict:
    """Keep a representative E2E run light while preserving 10 FL rounds."""
    profile = copy.deepcopy(profile)
    profile.setdefault("experiment", {})["num_clients"] = 10
    profile["experiment"]["rounds"] = 10

    dataset = profile.setdefault("dataset", {})
    dataset["samples_per_client"] = 128
    dataset["central_test_size"] = 512

    training = profile.setdefault("training", {})
    training["local_epochs"] = 1
    training["batch_size"] = min(int(training.get("batch_size", 64)), 64)

    storage = profile.setdefault("storage", {})
    storage["enabled"] = True
    storage["compression_level"] = 1
    storage["flush_each_round"] = True
    storage["capture_inference"] = True
    storage["probe_size"] = min(int(storage.get("probe_size", 128)), 128)

    attack = profile.setdefault("attack", {})
    schedule = attack.setdefault("schedule", {})
    schedule["type"] = "continuous"

    return profile


def _expected_attackers(profile: dict) -> int:
    attack = profile.get("attack", {})
    ids = str(attack.get("malicious_client_ids", "")).strip()
    if ids:
        return len([item for item in ids.split(",") if item.strip()])

    total = int(profile["experiment"]["num_clients"])
    fraction = float(attack.get("malicious_fraction", 0.0))
    if fraction <= 0.0:
        return 0
    return max(1, int(round(total * fraction)))


@pytest.mark.parametrize("profile_name", PROFILE_CASES)
def test_current_toml_attack_profile_runs_end_to_end(
    tmp_path: Path,
    profile_name: str,
):
    source_path = PROFILE_DIR / profile_name
    assert source_path.exists(), source_path

    profile = _scale_for_application_smoke(load_profile(source_path))
    mechanism = str(profile.get("attack", {}).get("mechanism", "none")).lower()
    expected_attackers = _expected_attackers(profile)

    run_dir = tmp_path / source_path.stem
    command = [
        sys.executable,
        "-m",
        "eiffel",
        *profile_to_overrides(profile),
        f"hydra.run.dir={run_dir.as_posix()}",
        "hydra.output_subdir=.hydra",
    ]

    completed = _run_with_ray_startup_retry(command)
    assert completed.returncode == 0, (
        f"{profile_name} failed through TOML -> Hydra -> Flower:\n"
        f"{completed.stdout}"
    )

    h5_path = run_dir / str(profile["storage"].get("path", "round_state.h5"))
    assert h5_path.exists(), completed.stdout

    validation_errors = validate(h5_path)
    assert not validation_errors, (
        f"{profile_name} produced an invalid round store:\n"
        + "\n".join(validation_errors)
    )

    with h5py.File(h5_path, "r") as h5:
        assert int(h5["meta"].attrs["last_complete_round"]) == 2
        clients = h5["clients"]["round_0002"]
        assert len(clients) == int(profile["experiment"]["num_clients"])

        malicious = [
            client
            for client in clients.values()
            if bool(int(client.attrs.get("malicious", 0)))
        ]
        assert len(malicious) == expected_attackers

        active_malicious = [
            client
            for client in malicious
            if bool(int(client.attrs.get("attack_active", 0)))
        ]

        if mechanism == "none":
            assert not malicious
            assert not active_malicious
        else:
            assert malicious
            assert active_malicious, (
                f"{profile_name} completed but no malicious client was active "
                "in the final smoke round"
            )

        if mechanism == "label_flip":
            assert any(
                float(client.attrs.get("data_poison_fraction", 0.0)) > 0.0
                for client in active_malicious
            )
        elif mechanism != "none":
            assert all("pre_attack_update" in client for client in active_malicious)

        round_meta = h5["rounds"]["round_0002"].attrs
        malicious_count = int(round_meta.get("malicious_clients", 0))
        assert malicious_count == expected_attackers



def test_sign_flip_runs_for_ten_rounds_with_longitudinal_outputs(
    tmp_path: Path,
):
    """Run one representative attack for long enough to test round evolution."""
    source_path = PROFILE_DIR / "synthetic_50k_sign_flip.toml"
    assert source_path.exists(), source_path

    profile = _scale_for_ten_round_longitudinal(load_profile(source_path))
    expected_attackers = _expected_attackers(profile)
    assert expected_attackers == 1

    run_dir = tmp_path / "sign_flip_10_rounds"
    command = [
        sys.executable,
        "-m",
        "eiffel",
        *profile_to_overrides(profile),
        f"hydra.run.dir={run_dir.as_posix()}",
        "hydra.output_subdir=.hydra",
    ]

    completed = _run_with_ray_startup_retry(command, timeout=600)
    assert completed.returncode == 0, (
        "10-round sign-flip run failed through TOML -> Hydra -> Flower:\n"
        f"{completed.stdout}"
    )

    h5_path = run_dir / str(profile["storage"].get("path", "round_state.h5"))
    assert h5_path.exists(), completed.stdout

    validation_errors = validate(h5_path)
    assert not validation_errors, (
        "10-round sign-flip run produced an invalid round store:\n"
        + "\n".join(validation_errors)
    )

    with h5py.File(h5_path, "r") as h5:
        assert int(h5["meta"].attrs["last_complete_round"]) == 10
        assert set(h5["clients"].keys()) == {
            f"round_{round_number:04d}" for round_number in range(1, 11)
        }
        assert set(h5["rounds"].keys()) == {
            f"round_{round_number:04d}" for round_number in range(1, 11)
        }
        assert set(h5["global"].keys()) == {
            f"round_{round_number:04d}" for round_number in range(0, 11)
        }

        for round_number in range(1, 11):
            round_name = f"round_{round_number:04d}"
            clients = h5["clients"][round_name]
            assert len(clients) == 10

            malicious = [
                client
                for client in clients.values()
                if bool(int(client.attrs.get("malicious", 0)))
            ]
            assert len(malicious) == expected_attackers
            assert all(
                bool(int(client.attrs.get("attack_active", 0)))
                for client in malicious
            )
            assert all("pre_attack_update" in client for client in malicious)

            for client in clients.values():
                for dataset in client["submitted_update"].values():
                    assert np.isfinite(np.asarray(dataset)).all()

            round_meta = h5["rounds"][round_name].attrs
            assert int(round_meta.get("malicious_clients", 0)) == expected_attackers
            assert float(round_meta.get("attack_multiplier", 0.0)) > 0.0

    analysis_dir = tmp_path / "sign_flip_10_round_analysis"
    rc = analyse_compact(
        [RunSpec("sign_flip_10_rounds", h5_path)],
        analysis_dir,
    )
    assert rc == 0

    run_outputs = [path for path in analysis_dir.iterdir() if path.is_dir()]
    assert len(run_outputs) == 1
    run_output = run_outputs[0]

    with (run_output / "round_performance.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        performance_rows = list(csv.DictReader(handle))
    with (run_output / "round_updates.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        update_rows = list(csv.DictReader(handle))

    assert [int(row["round"]) for row in performance_rows] == list(range(1, 11))
    assert [int(row["round"]) for row in update_rows] == list(range(1, 11))
    assert all(int(row["active_malicious_clients"]) == 1 for row in performance_rows)
    assert all(math.isfinite(float(row["accuracy"])) for row in performance_rows)
    assert all(math.isfinite(float(row["macro_f1"])) for row in performance_rows)

    transform_l2 = [
        float(row["malicious_transform_l2_mean"]) for row in update_rows
    ]
    assert all(math.isfinite(value) for value in transform_l2)
    assert any(value > 0.0 for value in transform_l2)

    for plot_name in ("performance_by_round.png", "updates_by_round.png"):
        plot = run_output / plot_name
        assert plot.exists()
        assert plot.stat().st_size > 0
