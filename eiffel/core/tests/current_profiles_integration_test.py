"""Representative committed profiles through the direct TOML runtime."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import h5py
import pytest

from eiffel.analysis.validate_round_state import validate
from eiffel.direct_runner import build_command, load_profile

ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = ROOT / "experiments" / "toml"

PROFILE_CASES = (
    "synthetic_50k_quick_clean.toml",
    "synthetic_50k_label_flip.toml",
    "synthetic_50k_sign_flip.toml",
    "synthetic_50k_model_scaling.toml",
    "synthetic_50k_gaussian_noise.toml",
    "synthetic_50k_lie.toml",
    "synthetic_50k_gradient_mimicry.toml",
    "synthetic_50k_colluding_sign_flip.toml",
    "synthetic_50k_min_max.toml",
    "synthetic_50k_min_sum.toml",
    "synthetic_50k_adaptive_stealth.toml",
    "synthetic_50k_heterogeneity_aware_mimicry.toml",
    "synthetic_50k_targeted_family_poisoning.toml",
    "portable_model_attack.toml",
)


def _scaled(profile: dict) -> dict:
    profile = copy.deepcopy(profile)
    profile.setdefault("experiment", {})["num_clients"] = 4
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
    storage["capture_inference"] = True
    storage["probe_size"] = min(int(storage.get("probe_size", 48)), 48)
    schedule = profile.setdefault("attack", {}).setdefault("schedule", {})
    schedule.update(type="continuous", start_round=1, end_round=2)
    return profile


def _expected_attackers(profile: dict) -> int:
    attack = profile.get("attack", {})
    ids = attack.get("malicious_client_ids") or []
    if ids:
        return len(ids)
    fraction = float(attack.get("malicious_fraction", 0.0))
    return 0 if fraction <= 0 else max(
        1, int(round(int(profile["experiment"]["num_clients"]) * fraction))
    )


@pytest.mark.parametrize("profile_name", PROFILE_CASES)
def test_profile_runs_without_hydra(tmp_path: Path, profile_name: str):
    source = PROFILE_DIR / profile_name
    profile = _scaled(load_profile(source))
    expected_attackers = _expected_attackers(profile)
    mechanism = str(profile.get("attack", {}).get("mechanism", "none"))

    profile_path = tmp_path / (source.stem + ".json")
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    run_dir = tmp_path / source.stem
    completed = subprocess.run(
        build_command(profile_path, output_dir=run_dir),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout
    assert ".hydra" not in completed.stdout
    assert (run_dir / "resolved_profile.json").exists()

    h5_path = run_dir / str(profile["storage"].get("path", "round_state.h5"))
    assert h5_path.exists(), completed.stdout
    assert not validate(h5_path)

    with h5py.File(h5_path, "r") as h5:
        assert int(h5["meta"].attrs["last_complete_round"]) == 2
        clients = h5["clients"]["round_0002"]
        assert len(clients) == 4
        malicious = [
            c for c in clients.values()
            if bool(int(c.attrs.get("malicious", 0)))
        ]
        assert len(malicious) == expected_attackers
        if mechanism == "none":
            assert not malicious
        else:
            assert malicious
            assert any(bool(int(c.attrs.get("attack_active", 0))) for c in malicious)
